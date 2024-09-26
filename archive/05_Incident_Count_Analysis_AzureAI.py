import os
import json
import glob
import re
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
)
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

# Load environment variables
load_dotenv()

# Azure Cognitive Services setup
text_analytics_key = os.getenv("AZURE_TEXT_ANALYTICS_KEY")
text_analytics_endpoint = os.getenv("AZURE_TEXT_ANALYTICS_ENDPOINT")
text_analytics_client = TextAnalyticsClient(text_analytics_endpoint, AzureKeyCredential(text_analytics_key))

# Azure Cognitive Search setup
search_service_name = os.getenv("AZURE_SEARCH_SERVICE_NAME")
search_admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
search_index_name = "incident-reports"

search_client = SearchClient(f"https://{search_service_name}.search.windows.net/",
                             search_index_name,
                             AzureKeyCredential(search_admin_key))

index_client = SearchIndexClient(f"https://{search_service_name}.search.windows.net/",
                                 AzureKeyCredential(search_admin_key))

# Azure Computer Vision setup
vision_key = os.getenv("AZURE_VISION_KEY")
vision_endpoint = os.getenv("AZURE_VISION_ENDPOINT")
vision_client = ComputerVisionClient(vision_endpoint, CognitiveServicesCredentials(vision_key))

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

# Create Azure Cognitive Search index
def create_search_index():
    index = SearchIndex(
        name=search_index_name,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="source", type=SearchFieldDataType.String),
        ],
    )
    index_client.create_or_update_index(index)

# Upload documents to Azure Cognitive Search
def upload_documents_to_search(documents: List[Dict[str, str]]):
    batch = []
    for i, doc in enumerate(documents):
        batch.append({
            "id": str(i),
            "content": doc['content'],
            "source": doc['source']
        })
        if len(batch) == 1000:
            search_client.upload_documents(batch)
            batch = []
    if batch:
        search_client.upload_documents(batch)

# Retrieve relevant documents using Azure Cognitive Search
def retrieve_relevant_docs(query: str, top_k: int = 5):
    results = search_client.search(query, top=top_k)
    return [{"content": doc["content"], "source": doc["source"]} for doc in results]

# Extract incident types using Azure Text Analytics
def extract_incident_types(documents, selected_types, num_docs):
    texts = [doc['content'] for doc in documents[:num_docs]]
    response = text_analytics_client.extract_key_phrases(texts)
    
    incident_types = []
    for doc in response:
        if not doc.is_error:
            for phrase in doc.key_phrases:
                if any(incident_type.lower() in phrase.lower() for incident_type in selected_types):
                    incident_types.append(phrase)
    
    return list(set(incident_types))

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
st.title("Incident Analysis Dashboard with Azure Technologies")

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

num_docs_to_analyze = st.slider("Number of documents to analyze", min_value=1, max_value=100, value=5, step=1)

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
    
    # Step 3: Create Azure Cognitive Search index and upload documents
    if 'processed_documents' in st.session_state and 'search_index_created' not in st.session_state:
        if st.button("Create Search Index and Upload Documents"):
            with st.spinner("Creating search index and uploading documents..."):
                create_search_index()
                upload_documents_to_search(st.session_state.processed_documents)
                st.session_state.search_index_created = True
            st.success("Search index created and documents uploaded successfully!")
    elif 'search_index_created' in st.session_state:
        st.success("Search index is already created and documents are uploaded.")
    
    # Button to extract incident types
    if st.button("Extract Incident Types"):
        if 'processed_documents' not in st.session_state:
            st.warning("Please process the documents first.")
        else:
            with st.spinner(f"Extracting incident types from {num_docs_to_analyze} documents..."):
                st.session_state.incident_types = extract_incident_types(st.session_state.processed_documents, selected_incident_types, num_docs_to_analyze)
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
            st.markdown('''
            **Interpretation**:
            
            - Rows represent the initial incident type, and columns represent the subsequent incident type.
            - Each number shows how many times an incident of the row type was followed by an incident of the column type.
            - The diagonal (where row and column are the same) shows incidents of the same type occurring consecutively.
            ''')

    # Question Answering section with Azure Cognitive Search
    st.header("Question Answering with Azure Cognitive Search")
    query = st.text_input("Enter your question about the incidents:")
    if st.button("Answer Question"):
        if query:
            if 'search_index_created' not in st.session_state:
                st.warning("Please complete all previous steps before asking a question.")
            else:
                with st.spinner("Answering question..."):
                    relevant_docs = retrieve_relevant_docs(query)
                    context = "\n".join([doc['content'] for doc in relevant_docs])
                    
                    # Use Azure Text Analytics for question answering
                    response = text_analytics_client.extract_key_phrases([context])
                    key_phrases = response[0].key_phrases if not response[0].is_error else []
                    
                    answer = f"Based on the relevant documents, the key phrases related to your question are: {', '.join(key_phrases[:5])}"
                    
                    st.subheader("Answer:")
                    st.write(answer)
                    
                    st.subheader("Relevant Document Chunks:")
                    for i, doc in enumerate(relevant_docs, 1):
                        st.write(f"{'Chunk' if st.session_state.use_chunking else 'Document'} {i} (Source: {doc['source']}):")
                        st.write(doc['content'])
                        st.write("---")
                st.success("Question answered using Azure Cognitive Search!")
        else:
            st.warning("Please enter a question.")

else:
    st.warning("Please select at least one incident type to analyze.")
