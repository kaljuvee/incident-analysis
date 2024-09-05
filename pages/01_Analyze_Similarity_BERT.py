import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, BertTokenizer, BertModel
import pandas as pd
from datetime import datetime
import json
import glob
import os
import re
import streamlit as st
import torch
from dotenv import load_dotenv
from openai import OpenAI
from gptcache import cache

# Load environment variables and set up OpenAI client
load_dotenv()
client = OpenAI()
cache.init()
cache.set_openai_key()

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

# Load pre-trained BERT model and tokenizer
@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

tokenizer, model = load_bert_model()

# Function to get BERT embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Create BERT embeddings for documents
@st.cache_resource
def create_bert_embeddings(docs):
    return np.array([get_embeddings(doc) for doc in docs])

document_embeddings = create_bert_embeddings(documents)

# Create a question-answering pipeline
@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa_pipeline = load_qa_pipeline()

def retrieve_relevant_docs(query, top_k=5):
    query_embedding = get_embeddings(query)
    similarities = cosine_similarity([query_embedding], document_embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [documents[i] for i in top_indices]

def answer_question(question, context):
    return qa_pipeline(question=question, context=context)

@st.cache_data
def extract_incident_types(documents):
    prompt = "Analyze the following incident reports and identify the main types of incidents mentioned. List only the incident types, separated by commas:\n\n"
    prompt += "\n\n".join(documents[:5])  # Use first 5 documents as a sample
    
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[{"role": "user", "content": prompt}]
    )
    incident_types = [t.strip() for t in response.choices[0].message.content.split(',')]
    return incident_types

def count_incident_types(documents, incident_types):
    counts = {incident: sum(1 for doc in documents if incident.lower() in doc.lower()) for incident in incident_types}
    return counts

def extract_date(text):
    # ... (keep the existing extract_date function)

def extract_plant(text):
    # ... (keep the existing extract_plant function)

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

# Extract incident types using OpenAI
incident_types = extract_incident_types(documents)

# Sidebar for incident type selection
selected_incident_types = st.sidebar.multiselect(
    "Select Incident Types to Analyze",
    options=incident_types,
    default=incident_types
)

# Main content
if selected_incident_types:
    st.header("Incident Type Counts")
    counts = count_incident_types(documents, selected_incident_types)
    st.bar_chart(counts)

    st.header("Pattern Analysis")
    analysis_results = analyze_patterns(documents, selected_incident_types)
    
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