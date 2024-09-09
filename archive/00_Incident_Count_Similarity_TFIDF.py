import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import pandas as pd
from datetime import datetime
import json
import glob
import os
import re
import streamlit as st

# Load the documents from JSON files
@st.cache_data
def load_documents(directory):
    documents = []
    for filename in glob.glob(os.path.join(directory, 'incident_*.json')):
        with open(filename, 'r') as f:
            data = json.load(f)
            documents.append(data['incident_report'])
    return documents

# Load documents
documents = load_documents('data/incidents/')

# Create a TF-IDF vectorizer
@st.cache_resource
def create_tfidf_matrix(docs):
    vectorizer = TfidfVectorizer()
    return vectorizer, vectorizer.fit_transform(docs)

vectorizer, tfidf_matrix = create_tfidf_matrix(documents)

# Create a question-answering pipeline
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa_pipeline = load_qa_pipeline()

def retrieve_relevant_docs(query, top_k=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [documents[i] for i in top_indices]

def answer_question(question, context):
    return qa_pipeline(question=question, context=context)

def count_incident_types(incident_types):
    counts = {incident: sum(1 for doc in documents if incident in doc.lower()) for incident in incident_types}
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

def analyze_patterns(incident_types):
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
            if incident_type in text.lower():
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

# Sidebar for incident type selection
default_incident_types = ['slip', 'fire', 'safety violation', 'chemical spill', 'injury', 'near-miss', 
                          'electrical', 'ventilation', 'falling object', 'heat exhaustion']

selected_incident_types = st.sidebar.multiselect(
    "Select Incident Types to Analyze",
    options=default_incident_types,
    default=default_incident_types
)

# Main content
if selected_incident_types:
    st.header("Incident Type Counts")
    counts = count_incident_types(selected_incident_types)
    st.bar_chart(counts)

    st.header("Pattern Analysis")
    analysis_results = analyze_patterns(selected_incident_types)
    
    st.subheader("Incident Rates per Plant")
    st.bar_chart(analysis_results['incident_rates_per_plant'])
    
    st.subheader("Monthly Incident Counts")
    st.line_chart(analysis_results['monthly_incident_counts'])
    
    st.subheader("Incident Type Correlations")
    st.dataframe(analysis_results['incident_type_correlations'])

    st.header("Question Answering")
    query = st.text_input("Enter your question about the incidents:")
    if query:
        relevant_docs = retrieve_relevant_docs(query)
        for i, doc in enumerate(relevant_docs, 1):
            st.subheader(f"Relevant Document {i}")
            st.write(doc)
            answer = answer_question(query, doc)
            st.write("Answer:", answer['answer'])
            st.write("Confidence:", answer['score'])

else:
    st.warning("Please select at least one incident type to analyze.")