import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datetime import datetime
import json
import glob
import os
import re
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from gptcache import cache
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from typing import List, Dict

# Load environment variables and set up OpenAI client
load_dotenv()
client = OpenAI()
cache.init()
cache.set_openai_key()

# Define available models
MODEL_OPTIONS = {
    'BERT': 'bert-base-uncased',
    'RoBERTa': 'roberta-base',
    'DistilBERT': 'distilbert-base-uncased',
    'ALBERT': 'albert-base-v2',
    'XLNet': 'xlnet-base-cased'
}

# Simple text chunker
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Load the documents from JSON files
@st.cache_data
def load_documents(directory: str) -> List[Dict[str, str]]:
    documents = []
    for filename in glob.glob(os.path.join(directory, 'incident_*.json')):
        with open(filename, 'r') as f:
            data = json.load(f)
            documents.append({
                'content': data['incident_report'],
                'source': filename
            })
    return documents

# Chunk the documents
def chunk_documents(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    chunked_documents = []
    for doc in documents:
        chunks = chunk_text(doc['content'])
        for chunk in chunks:
            chunked_documents.append({
                'content': chunk,
                'source': doc['source']
            })
    return chunked_documents

# Create embeddings using the selected model
@st.cache_resource
def create_embeddings(docs: List[Dict[str, str]], model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    embeddings = []
    for doc in docs:
        inputs = tokenizer(doc['content'], return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    
    return tokenizer, model, np.array(embeddings)

# Create FAISS index
def create_faiss_index(embeddings: np.ndarray):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Retrieve relevant documents using FAISS
def retrieve_relevant_docs(query: str, top_k: int = 5):
    inputs = st.session_state.tokenizer(query, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = st.session_state.model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    distances, indices = st.session_state.faiss_index.search(np.array([query_embedding]), top_k)
    return [st.session_state.processed_documents[i] for i in indices[0]]

def extract_incident_types(documents, selected_types):
    prompt = f"Analyze the following incident reports and categorize them into the following types: {', '.join(selected_types)}. If an incident doesn't fit into these categories, label it as 'other'. List only the incident types found, separated by commas:\n\n"
    prompt += "\n\n".join([doc['content'] for doc in documents[:5]])  # Use first 5 documents as a sample
    
    gpt_model = st.session_state.gpt_model
    
    response = client.chat.completions.create(
        model=gpt_model,
        messages=[{"role": "user", "content": prompt}]
    )
    incident_types = [t.strip() for t in response.choices[0].message.content.split(',')]
    return incident_types

def count_incident_types(documents, incident_types):
    counts = {incident: sum(1 for doc in documents if incident.lower() in doc['content'].lower()) for incident in incident_types}
    return counts

def extract_date(text):
    # Try to find a date in the format YYYY-MM-DD
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', text)
    if date_match:
        return datetime.strptime(date_match.group(), '%Y-%m-%d')
    
    # If not found, try other common formats
    date_patterns = [
        (r'\d{2}/\d{2}/\d{4}', '%m/%d/%Y'),
        (r'\d{2}-\d{2}-\d{4}', '%m-%d-%Y'),
        (r'\w+ \d{1,2}, \d{4}', '%B %d, %Y')
    ]
    
    for pattern, date_format in date_patterns:
        date_match = re.search(pattern, text)
        if date_match:
            try:
                return datetime.strptime(date_match.group(), date_format)
            except ValueError:
                continue
    
    # If no date found, return None
    return None

def extract_plant(text):
    plant_match = re.search(r'Plant [A-Z]', text)
    return plant_match.group() if plant_match else 'Unknown'

def analyze_patterns(documents, incident_types):
    df = pd.DataFrame([
        {
            'date': extract_date(doc['content']),
            'plant': extract_plant(doc['content']),
            'incident': doc['content']
        }
        for doc in documents
    ])
    
    # Remove rows with no date
    df = df.dropna(subset=['date'])
    
    # Incident rates per plant
    incident_rates = df.groupby('plant')['incident'].count() / len(df)
    
    # Changes over time
    monthly_incidents = df.resample('M', on='date')['incident'].count()
    
    def categorize_incident(text):
        for incident_type in incident_types:
            if incident_type.lower() in text.lower():
                return incident_type
        return 'other'
    
    df['incident_type'] = df['incident'].apply(categorize_incident)
    
    correlation_matrix = pd.crosstab(df['incident_type'], df['incident_type'].shift())
    
    return {
        'incident_rates_per_plant': incident_rates.to_dict(),
        'monthly_incident_counts': monthly_incidents.to_dict(),
        'incident_type_correlations': correlation_matrix.to_dict()
    }

# Streamlit app
st.title("Incident Analysis Dashboard with RAG")

# Sidebar for GPT model selection
gpt_models = ["gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
st.session_state.gpt_model = st.sidebar.selectbox("Select GPT Model", gpt_models)

# Sidebar for vectorization method selection
vectorization_methods = ["TF-IDF", "Embeddings"]
st.session_state.vectorization_method = st.sidebar.selectbox("Select Vectorization Method", vectorization_methods)

# Sidebar for embedding model selection (only shown if Embeddings is selected)
if st.session_state.vectorization_method == "Embeddings":
    st.session_state.embedding_model = st.sidebar.selectbox("Select Embedding Model", list(MODEL_OPTIONS.keys()))

# Sidebar for document processing option
st.session_state.use_chunking = st.sidebar.radio("Document Processing", ["Full Documents", "Chunked Documents"]) == "Chunked Documents"

# Sidebar for incident type selection
default_incident_types = ['slip', 'fire', 'safety violation', 'chemical spill', 'injury', 'near-miss', 
                          'electrical', 'ventilation', 'falling object', 'heat exhaustion', 'other']

selected_incident_types = st.sidebar.multiselect(
    "Select Incident Types to Analyze",
    options=default_incident_types,
    default=default_incident_types
)

# Allow users to add custom incident types
custom_type = st.sidebar.text_input("Add custom incident type")
if custom_type and custom_type not in selected_incident_types:
    selected_incident_types.append(custom_type)

# Main content
if selected_incident_types:
    # Step 1: Load documents
    if 'documents' not in st.session_state:
        if st.button("Load Documents"):
            with st.spinner("Loading documents..."):
                st.session_state.documents = load_documents('data/incidents/')
            st.success("Documents loaded successfully!")
    elif st.session_state.documents:
        st.success("Documents are already loaded.")
    
    # Step 2: Process documents (chunking if selected)
    if 'documents' in st.session_state and 'processed_documents' not in st.session_state:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                if st.session_state.use_chunking:
                    st.session_state.processed_documents = chunk_documents(st.session_state.documents)
                    st.success(f"Documents chunked successfully! Total chunks: {len(st.session_state.processed_documents)}")
                else:
                    st.session_state.processed_documents = st.session_state.documents
                    st.success("Documents processed without chunking.")
    elif 'processed_documents' in st.session_state:
        st.success("Documents are already processed.")
    
    # Step 3: Create embeddings
    if 'processed_documents' in st.session_state and 'embeddings' not in st.session_state:
        if st.button("Create Embeddings"):
            with st.spinner(f"Creating embeddings using {st.session_state.embedding_model}..."):
                model_name = MODEL_OPTIONS[st.session_state.embedding_model]
                st.session_state.tokenizer, st.session_state.model, st.session_state.embeddings = create_embeddings(st.session_state.processed_documents, model_name)
            st.success("Embeddings created successfully!")
    elif 'embeddings' in st.session_state:
        st.success("Embeddings are already created.")
    
    # Step 4: Create FAISS index
    if 'embeddings' in st.session_state and 'faiss_index' not in st.session_state:
        if st.button("Create Index"):
            with st.spinner("Creating an index..."):
                st.session_state.faiss_index = create_faiss_index(st.session_state.embeddings)
            st.success("Index created successfully!")
    elif 'faiss_index' in st.session_state:
        st.success("Index is already created.")
    
    # Button to extract incident types
    if st.button("Extract Incident Types"):
        if 'processed_documents' not in st.session_state:
            st.warning("Please process the documents first.")
        else:
            with st.spinner("Extracting incident types..."):
                st.session_state.incident_types = extract_incident_types(st.session_state.processed_documents, selected_incident_types)
            st.success("Incident types extracted successfully!")
            st.write("Extracted incident types:", st.session_state.incident_types)
    
    # Button to count incidents
    if st.button("Count Incidents"):
        if 'incident_types' not in st.session_state:
            st.warning("Please extract incident types first.")
        else:
            with st.spinner("Counting incidents..."):
                counts = count_incident_types(st.session_state.processed_documents, st.session_state.incident_types)
            st.success("Incidents counted successfully!")
            st.header("Incident Type Counts")
            st.bar_chart(counts)
    
    # Button to perform pattern analysis
    if st.button("Analyze Patterns"):
        if 'incident_types' not in st.session_state:
            st.warning("Please extract incident types first.")
        else:
            with st.spinner("Analyzing patterns..."):
                analysis_results = analyze_patterns(st.session_state.processed_documents, st.session_state.incident_types)
            st.success("Pattern analysis completed successfully!")
            
            st.header("Pattern Analysis")
            
            st.subheader("Incident Rates per Plant")
            st.bar_chart(analysis_results['incident_rates_per_plant'])
            
            st.subheader("Monthly Incident Counts")
            st.line_chart(analysis_results['monthly_incident_counts'])
            
            st.subheader("Incident Type Correlations")
            st.dataframe(analysis_results['incident_type_correlations'])
    
    # Question Answering section with RAG
    st.header("Question Answering with RAG")
    query = st.text_input("Enter your question about the incidents:")
    if st.button("Answer Question"):
        if query:
            if 'faiss_index' not in st.session_state:
                st.warning("Please complete all previous steps before asking a question.")
            else:
                with st.spinner("Answering question..."):
                    relevant_docs = retrieve_relevant_docs(query)
                    context = "\n".join([doc['content'] for doc in relevant_docs])
                    
                    # Use GPT for answering with RAG
                    response = client.chat.completions.create(
                        model=st.session_state.gpt_model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant analyzing incident reports. Use the provided context to answer the question."},
                            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
                        ]
                    )
                    st.subheader("Answer:")
                    st.write(response.choices[0].message.content)
                    
                    st.subheader("Relevant Document Chunks:")
                    for i, doc in enumerate(relevant_docs, 1):
                        st.write(f"{'Chunk' if st.session_state.use_chunking else 'Document'} {i} (Source: {doc['source']}):")
                        st.write(doc['content'])
                        st.write("---")
                st.success("Question answered using RAG!")
        else:
            st.warning("Please enter a question.")

else:
    st.warning("Please select at least one incident type to analyze.")