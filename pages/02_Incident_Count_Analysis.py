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

# Load the documents from JSON files
@st.cache_data
def load_documents(directory):
    documents = []
    for filename in glob.glob(os.path.join(directory, 'incident_*.json')):
        with open(filename, 'r') as f:
            data = json.load(f)
            documents.append(data['incident_report'])
    return documents

# Create a TF-IDF vectorizer
@st.cache_resource
def create_tfidf_matrix(docs):
    vectorizer = TfidfVectorizer()
    return vectorizer, vectorizer.fit_transform(docs)

# Create embeddings using the selected model
@st.cache_resource
def create_embeddings(docs, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    embeddings = []
    for doc in docs:
        inputs = tokenizer(doc, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    
    return tokenizer, model, np.array(embeddings)

def retrieve_relevant_docs(query, top_k=5):
    if st.session_state.vectorization_method == 'TF-IDF':
        query_vec = st.session_state.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, st.session_state.tfidf_matrix).flatten()
    else:  # Embeddings
        inputs = st.session_state.tokenizer(query, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = st.session_state.model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        similarities = cosine_similarity([query_embedding], st.session_state.embeddings).flatten()
    
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [st.session_state.documents[i] for i in top_indices]

def extract_incident_types(documents, selected_types):
    prompt = f"Analyze the following incident reports and categorize them into the following types: {', '.join(selected_types)}. If an incident doesn't fit into these categories, label it as 'other'. List only the incident types found, separated by commas:\n\n"
    prompt += "\n\n".join(documents[:5])  # Use first 5 documents as a sample
    
    gpt_model = st.session_state.gpt_model
    
    response = client.chat.completions.create(
        model=gpt_model,
        messages=[{"role": "user", "content": prompt}]
    )
    incident_types = [t.strip() for t in response.choices[0].message.content.split(',')]
    return incident_types

def count_incident_types(documents, incident_types):
    counts = {incident: sum(1 for doc in documents if incident.lower() in doc.lower()) for incident in incident_types}
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
            'date': extract_date(doc),
            'plant': extract_plant(doc),
            'incident': doc
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
st.title("Incident Analysis Dashboard")

# Sidebar for GPT model selection
gpt_models = ["gpt-4", "gpt-4-turbo", "gpt-4o"]
st.session_state.gpt_model = st.sidebar.selectbox("Select GPT Model", gpt_models)

# Sidebar for vectorization method selection
vectorization_methods = ["TF-IDF", "Embeddings"]
st.session_state.vectorization_method = st.sidebar.selectbox("Select Vectorization Method", vectorization_methods)

# Sidebar for embedding model selection (only shown if Embeddings is selected)
if st.session_state.vectorization_method == "Embeddings":
    st.session_state.embedding_model = st.sidebar.selectbox("Select Embedding Model", list(MODEL_OPTIONS.keys()))

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
    # Load documents
    if 'documents' not in st.session_state:
        with st.spinner("Loading documents..."):
            st.session_state.documents = load_documents('data/incidents/')
        st.success("Documents loaded successfully!")
    
    documents = st.session_state.documents
    
    # Prepare vectorization
    if st.button("Prepare Vectorization"):
        with st.spinner(f"Preparing {st.session_state.vectorization_method} vectorization..."):
            if st.session_state.vectorization_method == 'TF-IDF':
                st.session_state.vectorizer, st.session_state.tfidf_matrix = create_tfidf_matrix(documents)
            else:  # Embeddings
                model_name = MODEL_OPTIONS[st.session_state.embedding_model]
                st.session_state.tokenizer, st.session_state.model, st.session_state.embeddings = create_embeddings(documents, model_name)
        st.success(f"{st.session_state.vectorization_method} vectorization prepared successfully!")
    
    # Button to extract incident types
    if st.button("Extract Incident Types"):
        with st.spinner("Extracting incident types..."):
            st.session_state.incident_types = extract_incident_types(documents, selected_incident_types)
        st.success("Incident types extracted successfully!")
        st.write("Extracted incident types:", st.session_state.incident_types)
    
    # Button to count incidents
    if st.button("Count Incidents"):
        if 'incident_types' not in st.session_state:
            st.warning("Please extract incident types first.")
        else:
            with st.spinner("Counting incidents..."):
                counts = count_incident_types(documents, st.session_state.incident_types)
            st.success("Incidents counted successfully!")
            st.header("Incident Type Counts")
            st.bar_chart(counts)
    
    # Button to perform pattern analysis
    if st.button("Analyze Patterns"):
        if 'incident_types' not in st.session_state:
            st.warning("Please extract incident types first.")
        else:
            with st.spinner("Analyzing patterns..."):
                analysis_results = analyze_patterns(documents, st.session_state.incident_types)
            st.success("Pattern analysis completed successfully!")
            
            st.header("Pattern Analysis")
            
            st.subheader("Incident Rates per Plant")
            st.bar_chart(analysis_results['incident_rates_per_plant'])
            
            st.subheader("Monthly Incident Counts")
            st.line_chart(analysis_results['monthly_incident_counts'])
            
            st.subheader("Incident Type Correlations")
            st.dataframe(analysis_results['incident_type_correlations'])
    
    # Question Answering section
    st.header("Question Answering")
    query = st.text_input("Enter your question about the incidents:")
    if st.button("Answer Question"):
        if query:
            if 'vectorizer' not in st.session_state and 'model' not in st.session_state:
                st.warning("Please prepare vectorization first.")
            else:
                with st.spinner("Answering question..."):
                    relevant_docs = retrieve_relevant_docs(query)
                    for i, doc in enumerate(relevant_docs, 1):
                        st.subheader(f"Relevant Document {i}")
                        st.write(doc)
                        
                        # Use GPT for answering
                        response = client.chat.completions.create(
                            model=st.session_state.gpt_model,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant analyzing incident reports."},
                                {"role": "user", "content": f"Based on this incident report: '{doc}', please answer the following question: {query}"}
                            ]
                        )
                        st.write("Answer:", response.choices[0].message.content)
                st.success("Question answered!")
        else:
            st.warning("Please enter a question.")

else:
    st.warning("Please select at least one incident type to analyze.")