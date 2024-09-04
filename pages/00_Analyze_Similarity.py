import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import pandas as pd
from datetime import datetime

# Load the documents (assuming they're stored in a text file, one per line)
with open('incident_documents.txt', 'r') as f:
    documents = f.readlines()

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Create a question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def retrieve_relevant_docs(query, top_k=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [documents[i] for i in top_indices]

def answer_question(question, context):
    return qa_pipeline(question=question, context=context)

def count_incident_types():
    incident_types = ['slip', 'fire', 'safety violation', 'chemical spill', 'injury', 'near-miss', 
                      'electrical', 'ventilation', 'falling object', 'heat exhaustion']
    counts = {incident: sum(1 for doc in documents if incident in doc.lower()) for incident in incident_types}
    return counts

def analyze_patterns():
    # This is a simplified analysis. In a real-world scenario, you'd need more sophisticated NLP and data analysis techniques.
    df = pd.DataFrame([doc.split(', ') for doc in documents], columns=['date', 'plant', 'incident'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Incident rates per plant
    incident_rates = df.groupby('plant')['incident'].count() / len(df)
    
    # Changes over time
    monthly_incidents = df.resample('M', on='date')['incident'].count()
    
    # Correlation between incident types (simplified)
    incident_types = df['incident'].unique()
    correlation_matrix = pd.DataFrame(index=incident_types, columns=incident_types)
    for i in incident_types:
        for j in incident_types:
            correlation_matrix.loc[i, j] = ((df['incident'] == i) & (df['incident'].shift() == j)).sum()
    
    return {
        'incident_rates_per_plant': incident_rates.to_dict(),
        'monthly_incident_counts': monthly_incidents.to_dict(),
        'incident_type_correlations': correlation_matrix.to_dict()
    }

# Example usage
query = "What are common causes of slip and fall incidents?"
relevant_docs = retrieve_relevant_docs(query)
for doc in relevant_docs:
    print(answer_question(query, doc))

print("\nIncident type counts:")
print(count_incident_types())

print("\nPattern analysis:")
print(analyze_patterns())