import numpy as np
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
import faiss
from typing import List, Dict
import pickle
from difflib import SequenceMatcher

# Load environment variables and set up OpenAI client
load_dotenv()
client = OpenAI()
cache.init()
cache.set_openai_key()

# Define available models
EMBEDDING_MODELS = [
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large"
]

# Initialize session state variables
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'incident_types' not in st.session_state:
    st.session_state.incident_types = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model' not in st.session_state:
    st.session_state.model = None

ORIG_INCIDENT_TYPES = [
    'slip', 'fire', 'safety_violation', 'chemical_spill', 'injury', 'near_miss',
    'electrical', 'ventilation', 'falling object', 'heat exhaustion'
]

# Add this new function for comparing incident types
def compare_incident_types(original, discovered):
    original_formatted = [o.replace('-', ' ') for o in original]
    
    # Find exact matches
    exact_matches = set(original_formatted) & set(discovered)
    
    # Find close matches for non-exact matches
    close_matches = []
    unmatched_original = [o for o in original_formatted if o not in exact_matches]
    unmatched_discovered = [d for d in discovered if d not in exact_matches]
    
    for o in unmatched_original:
        best_match = None
        best_ratio = 0
        for d in unmatched_discovered:
            ratio = SequenceMatcher(None, o, d).ratio()
            if ratio > best_ratio and ratio > 0.6:  # Adjust threshold as needed
                best_ratio = ratio
                best_match = d
        if best_match:
            close_matches.append((o, best_match))
            unmatched_discovered.remove(best_match)
    
    return {
        'exact_matches': list(exact_matches),
        'close_matches': close_matches,
        'unmatched_original': [o for o in unmatched_original if o not in [m[0] for m in close_matches]],
        'unmatched_discovered': unmatched_discovered
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
    for filename in glob.glob(os.path.join(directory, '*.txt')):
        with open(filename, 'r') as f:
            content = f.read()
            documents.append({
                'content': content,
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
def create_embeddings(docs: List[Dict[str, str]], model: str):
    embeddings = []
    for doc in docs:
        response = client.embeddings.create(
            model=model,
            input=doc['content']
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)

# Create FAISS index
def create_faiss_index(embeddings: np.ndarray):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Retrieve relevant documents using FAISS
def retrieve_relevant_docs(query: str, top_k: int = 5):
    query_embedding = client.embeddings.create(
        model=st.session_state.embedding_model,
        input=query
    ).data[0].embedding
    
    distances, indices = st.session_state.faiss_index.search(np.array([query_embedding]), top_k)
    return [st.session_state.processed_documents[i] for i in indices[0]]
# New function to discover incident types
def discover_incident_types(documents, num_docs):
    prompt = f"""Analyze the following incident reports and list all unique incident types you can identify. 
    List only the incident types found, separated by commas. 
    Format multi-word types as word_anotherword (e.g., 'safety violation' should be 'safety_violation').
    Here are some example formats: slip, fire, safety_violation, chemical_spill, injury, near_miss, 
    electrical, ventilation, falling_object, heat_exhaustion.\n\n
    """
    prompt += "\n\n".join([doc['content'] for doc in documents[:num_docs]])
    
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
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', text)
    if date_match:
        return datetime.strptime(date_match.group(), '%Y-%m-%d')
    
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
    
    df = df.dropna(subset=['date'])
    
    incident_rates = df.groupby('plant')['incident'].count() / len(df)
    
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

# Add this function to load and process the CSV file
def load_incident_frequency():
    df = pd.read_csv('data/incident_type_frequency.csv')
    return dict(zip(df['Incident Type'], df['Frequency']))

# Add these new functions to save and load embeddings and index
def save_embeddings_and_index(embeddings, index, processed_documents):
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    faiss.write_index(index, 'faiss_index.bin')
    with open('processed_documents.pkl', 'wb') as f:
        pickle.dump(processed_documents, f)

def load_embeddings_and_index():
    with open('embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    index = faiss.read_index('faiss_index.bin')
    with open('processed_documents.pkl', 'rb') as f:
        processed_documents = pickle.load(f)
    return embeddings, index, processed_documents

# Streamlit app
st.title("Incident Analysis Dashboard")

# Sidebar for GPT model selection
gpt_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
st.session_state.gpt_model = st.sidebar.selectbox("Select GPT Model", gpt_models)

# Sidebar for vectorization method selection
st.session_state.embedding_model = st.sidebar.selectbox(
    "Select Embedding Model",
    options=EMBEDDING_MODELS,
    index=EMBEDDING_MODELS.index(st.session_state.get('embedding_model', EMBEDDING_MODELS[0]))
)

# Sidebar for document processing option
st.session_state.use_chunking = st.sidebar.radio("Document Processing", ["Full Documents", "Chunked Documents"]) == "Chunked Documents"

# Main content
# Step 1: Load documents
if st.session_state.documents is None:
    if st.button("Load Documents"):
        with st.spinner("Loading documents..."):
            st.session_state.documents = load_documents('data')
        st.success("Documents loaded successfully!")
else:
    st.success("Documents are already loaded.")

# Step 2: Process documents (chunking if selected)
if st.session_state.documents is not None and st.session_state.processed_documents is None:
    if st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            if st.session_state.use_chunking:
                st.session_state.processed_documents = chunk_documents(st.session_state.documents)
                st.success(f"Documents chunked successfully! Total chunks: {len(st.session_state.processed_documents)}")
            else:
                st.session_state.processed_documents = st.session_state.documents
                st.success("Documents processed without chunking.")
elif st.session_state.processed_documents is not None:
    st.success("Documents are already processed.")

# Slider for number of documents to analyze
if st.session_state.processed_documents is not None:
    num_docs_to_analyze = st.slider("Number of documents to analyze", 
                                    min_value=1, 
                                    max_value=len(st.session_state.processed_documents), 
                                    value=min(5, len(st.session_state.processed_documents)), 
                                    step=1)
else:
    st.warning("Please load and process documents before analyzing.")
    num_docs_to_analyze = 5  # Default value

# Step 3: Create embeddings
if st.session_state.processed_documents is not None and st.session_state.embeddings is None:
    if st.button("Create Embeddings"):
        with st.spinner(f"Creating embeddings using {st.session_state.embedding_model}..."):
            st.session_state.embeddings = create_embeddings(st.session_state.processed_documents, st.session_state.embedding_model)
        st.success("Embeddings created successfully!")
elif st.session_state.embeddings is not None:
    st.success("Embeddings are already created.")

# Step 4: Create FAISS index
if st.session_state.embeddings is not None and st.session_state.faiss_index is None:
    if st.button("Create Index"):
        with st.spinner("Creating an index..."):
            st.session_state.faiss_index = create_faiss_index(st.session_state.embeddings)
        st.success("Index created successfully!")
elif st.session_state.faiss_index is not None:
    st.success("Index is already created.")

# Add a button to save embeddings and index
if st.session_state.embeddings is not None and st.session_state.faiss_index is not None:
    if st.button("Save Embeddings and Index"):
        with st.spinner("Saving embeddings and index..."):
            save_embeddings_and_index(st.session_state.embeddings, st.session_state.faiss_index, st.session_state.processed_documents)
        st.success("Embeddings and index saved successfully!")

# Button to discover incident types
if st.button("Discover Incident Types"):
    if st.session_state.processed_documents is None:
        st.warning("Please process the documents first.")
    else:
        with st.spinner(f"Discovering incident types from {num_docs_to_analyze} documents..."):
            st.session_state.incident_types = discover_incident_types(st.session_state.processed_documents, num_docs_to_analyze)
        st.success("Incident types discovered successfully!")
        st.write("Discovered incident types:", st.session_state.incident_types)


if st.button("Compare Incident Types"):
    if st.session_state.incident_types is None:
        st.warning("Please discover incident types first.")
    else:
        comparison = compare_incident_types(ORIG_INCIDENT_TYPES, st.session_state.incident_types)
        
        st.subheader("Incident Type Comparison")
        
        st.write("Exact Matches:")
        st.write(", ".join(comparison['exact_matches']))
        
        st.write("Close Matches:")
        for original, discovered in comparison['close_matches']:
            st.write(f"- {original} ≈ {discovered}")
        
        st.write("Unmatched Original Types:")
        st.write(", ".join(comparison['unmatched_original']))
        
        st.write("Unmatched Discovered Types:")
        st.write(", ".join(comparison['unmatched_discovered']))

# Button to count incidents
if st.button("Count Incidents"):
    if st.session_state.incident_types is None:
        st.warning("Please discover incident types first.")
    else:
        with st.spinner("Counting incidents..."):
            counts = count_incident_types(st.session_state.processed_documents, st.session_state.incident_types)
            st.session_state.counts = counts  # Store counts in session state
        st.success("Incidents counted successfully!")
        st.header("Incident Type Counts")
        st.bar_chart(counts)

# Button to perform pattern analysis
if st.button("Analyze Patterns"):
    if st.session_state.incident_types is None:
        st.warning("Please discover incident types first.")
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
        st.markdown('''
        **Interpretation**:
        
        - Rows represent the initial incident type, and columns represent the subsequent incident type.
        - Each number shows how many times an incident of the row type was followed by an incident of the column type.
        - The diagonal (where row and column are the same) shows incidents of the same type occurring consecutively.
        ''')

    # Question Answering section with RAG
    st.header("Question Answering with RAG")
    query = st.text_input("Enter your question about the incidents:")
    if st.button("Answer Question"):
        if query:
            if st.session_state.faiss_index is None:
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

# Add this new button and functionality
if st.button("Compare Frequencies"):
    if st.session_state.incident_types is None or 'counts' not in st.session_state:
        st.warning("Please discover and count incident types first.")
    else:
        with st.spinner("Loading and comparing frequencies..."):
            csv_frequencies = load_incident_frequency()
            discovered_frequencies = st.session_state.counts

            st.subheader("Frequency Comparison")

            # Create a DataFrame for easy comparison
            comparison_df = pd.DataFrame({
                'CSV Frequency': pd.Series(csv_frequencies),
                'Discovered Frequency': pd.Series(discovered_frequencies)
            }).fillna(0)

            # Calculate the difference
            comparison_df['Difference'] = comparison_df['Discovered Frequency'] - comparison_df['CSV Frequency']

            # Display the comparison
            st.dataframe(comparison_df)

            # Visualize the comparison
            st.bar_chart(comparison_df[['CSV Frequency', 'Discovered Frequency']])

            # Analysis of differences
            st.subheader("Analysis of Differences")
            new_types = set(discovered_frequencies.keys()) - set(csv_frequencies.keys())
            missing_types = set(csv_frequencies.keys()) - set(discovered_frequencies.keys())
            
            if new_types:
                st.write("Newly discovered incident types:", ", ".join(new_types))
            if missing_types:
                st.write("Incident types not found in the current analysis:", ", ".join(missing_types))
            
            significant_diff = comparison_df[abs(comparison_df['Difference']) > 5].index.tolist()
            if significant_diff:
                st.write("Incident types with significant frequency differences:")
                for incident_type in significant_diff:
                    st.write(f"- {incident_type}: CSV: {csv_frequencies.get(incident_type, 0)}, "
                             f"Discovered: {discovered_frequencies.get(incident_type, 0)}")

        st.success("Frequency comparison completed!")

else:
    st.warning("Please select at least one incident type to analyze.")