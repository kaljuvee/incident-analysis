import streamlit as st
import json
import glob
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the documents from JSON files
@st.cache_data
def load_documents(directory):
    documents = []
    for filename in glob.glob(os.path.join(directory, 'incident_*.json')):
        with open(filename, 'r') as f:
            data = json.load(f)
            documents.append({
                'id': os.path.basename(filename),
                'report': data['incident_report']
            })
    return documents

# Create a TF-IDF vectorizer
@st.cache_resource
def create_tfidf_matrix(docs):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc['report'] for doc in docs])
    return vectorizer, tfidf_matrix

def search_documents(query, vectorizer, tfidf_matrix, documents, top_k=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [documents[i] for i in top_indices]

def main():
    st.title("Browse and Search Incident Documents")

    # Load documents
    documents = load_documents('data/incidents/')
    vectorizer, tfidf_matrix = create_tfidf_matrix(documents)

    # Search functionality
    search_query = st.text_input("Search incidents:", "")
    if search_query:
        search_results = search_documents(search_query, vectorizer, tfidf_matrix, documents)
        st.subheader("Search Results")
        for doc in search_results:
            st.write(f"Document ID: {doc['id']}")
            st.write(doc['report'])
            st.write("---")
    
    # Browse functionality
    st.subheader("Browse All Documents")
    for i, doc in enumerate(documents):
        with st.expander(f"Document {i+1} - ID: {doc['id']}"):
            st.write(doc['report'])

if __name__ == "__main__":
    main()