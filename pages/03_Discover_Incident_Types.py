import streamlit as st
from datetime import datetime
import spacy
from typing import List, Dict, Tuple
import json
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans, DBSCAN
from collections import Counter
from utils.embeddings_util import select_embedding_index
from utils.db_util import get_datasets_with_counts
from openai import OpenAI

# Load the spaCy model
nlp = spacy.load("en_core_web_md")

# Load the OpenAI client
client = OpenAI()

# Define ground truth incident types (unchanged)
ORIG_INCIDENT_TYPES = [
    'Slip',
    'Fire',
    'Safety violation',
    'Chemical spill',
    'Injury',
    'Near miss',
    'Electrical',
    'Ventilation',
    'Falling object',
    'Heat exhaustion',
    'Scaffolding',  # New incident type
    'Bee stings'    # New incident type
]

def preprocess_text(text: str) -> str:
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def extract_entities(text: str) -> List[str]:
    doc = nlp(text)
    entities = [ent.text.lower() for ent in doc.ents]
    noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    return list(set(entities + noun_chunks))

def apply_lda(documents: List[Dict[str, str]], num_topics: int = 15) -> List[str]:
    preprocessed_docs = [preprocess_text(doc['description']) for doc in documents]
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english', ngram_range=(1, 2))
    doc_term_matrix = vectorizer.fit_transform(preprocessed_docs)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42, max_iter=20)
    lda.fit(doc_term_matrix)
    feature_names = vectorizer.get_feature_names_out()
    return [", ".join([feature_names[i] for i in topic.argsort()[:-15 - 1:-1]]) for topic in lda.components_]

def apply_nmf(documents: List[Dict[str, str]], num_topics: int = 15) -> List[str]:
    preprocessed_docs = [preprocess_text(doc['description']) for doc in documents]
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english', ngram_range=(1, 2))
    doc_term_matrix = vectorizer.fit_transform(preprocessed_docs)
    nmf = NMF(n_components=num_topics, random_state=42, max_iter=400)
    nmf.fit(doc_term_matrix)
    feature_names = vectorizer.get_feature_names_out()
    return [", ".join([feature_names[i] for i in topic.argsort()[:-15 - 1:-1]]) for topic in nmf.components_]

def apply_kmeans(embeddings: np.ndarray, n_clusters: int = 15) -> List[str]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    return [f"Cluster {i}" for i in range(n_clusters)]

def apply_dbscan(embeddings: np.ndarray, eps: float = 0.3, min_samples: int = 3) -> List[str]:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(embeddings)
    return [f"Cluster {i}" for i in set(cluster_labels) if i != -1]

def apply_entity_based(documents: List[Dict[str, str]], top_n: int = 30) -> List[str]:
    all_entities = []
    for doc in documents:
        all_entities.extend(extract_entities(doc['description']))
    entity_counts = Counter(all_entities)
    return [entity for entity, count in entity_counts.most_common(top_n)]

def gpt_based(documents: List[Dict[str, str]], prompt: str, temperature: float = 0.3, max_tokens: int = 200) -> List[str]:
    full_prompt = prompt + "\n\n" + "\n\n".join([doc['description'] for doc in documents])
    
    response = client.chat.completions.create(
        model=st.session_state.gpt_model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    content = response.choices[0].message.content
    incident_types = parse_gpt_response(content)
    return incident_types

def parse_gpt_response(content: str) -> List[str]:
    # Extract the incident types from the GPT response
    incident_types = []
    for line in content.split('\n'):
        if line.startswith("Incident Types:"):
            types = line.split(':')[1].strip()
            incident_types = [t.strip() for t in types.split(',')]
            break
    return incident_types

def discover_incident_types(documents: List[Dict[str, str]], embeddings: np.ndarray, gpt_prompt: str):
    results = {}
    
    # LDA
    results['LDA'] = apply_lda(documents)
    
    # NMF
    results['NMF'] = apply_nmf(documents)
    
    # K-Means
    results['KMeans'] = apply_kmeans(embeddings)
    
    # DBSCAN
    results['DBSCAN'] = apply_dbscan(embeddings)
    
    # Entity-based
    results['Entity-based'] = apply_entity_based(documents)
    
    # GPT-based
    results['GPT-based'] = gpt_based(documents, gpt_prompt)
    
    return results

def save_incident_info(discovery_results, index_id):
    info = {
        'index_id': index_id,
        'discovery_results': discovery_results,
        'timestamp': datetime.now().isoformat()
    }
    os.makedirs('embeddings', exist_ok=True)
    filename = f'embeddings/incident_info_{index_id}.json'
    with open(filename, 'w') as f:
        json.dump(info, f, indent=2)
    return filename

def load_incident_info(index_id):
    filename = f'embeddings/incident_info_{index_id}.json'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def calculate_match_percentage(discovered: List[str], ground_truth: List[str]) -> float:
    discovered_docs = [nlp(term.lower()) for term in discovered]
    ground_truth_docs = [nlp(term.lower()) for term in ground_truth]
    
    matches = 0
    for gt_doc in ground_truth_docs:
        similarities = [gt_doc.similarity(disc_doc) for disc_doc in discovered_docs]
        if similarities:
            max_similarity = max(similarities)
            if max_similarity > 0.7:  # Adjust this threshold as needed
                matches += max_similarity  # Use the similarity score instead of a binary match
    
    return (matches / len(ground_truth)) * 100

def compare_with_ground_truth(discovered_types: Dict[str, List[str]], ground_truth: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    comparison_data = []
    summary_data = []
    
    for method, types in discovered_types.items():
        match_percentage = calculate_match_percentage(types, ground_truth)
        
        matched_types = []
        for gt_type in ground_truth:
            gt_doc = nlp(gt_type.lower())
            best_match = max(types, key=lambda x: nlp(x.lower()).similarity(gt_doc))
            if nlp(best_match.lower()).similarity(gt_doc) > 0.7:
                matched_types.append(f"{gt_type} -> {best_match}")
        
        comparison_data.append({
            'Method': method,
            'Discovered Types': ', '.join(types),
            'Matched Ground Truth': ', '.join(matched_types),
            'Match Percentage': f"{match_percentage:.2f}%"
        })
        
        summary_data.append({
            'Method': method,
            'Match Percentage': match_percentage
        })
    
    # Add ground truth to the comparison
    comparison_data.append({
        'Method': 'Ground Truth',
        'Discovered Types': ', '.join(ground_truth),
        'Matched Ground Truth': ', '.join(ground_truth),
        'Match Percentage': '100.00%'
    })
    
    summary_data.append({
        'Method': 'Ground Truth',
        'Match Percentage': 100.00
    })
    
    comparison_df = pd.DataFrame(comparison_data)
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Match Percentage', ascending=False).reset_index(drop=True)
    
    return comparison_df, summary_df

# Streamlit app
st.title("Discover and Compare Incident Types")

# Load embeddings and index
selected_metadata, embeddings, index, processed_documents, embedding_model = select_embedding_index()

if selected_metadata is not None:
    st.success("Embeddings, index, and processed documents loaded successfully!")
    st.info(f"Number of documents to analyze: {len(processed_documents)}")
    st.info(f"Using embedding index: {selected_metadata['index_id']}")

    # Add GPT model selection and parameters
    st.sidebar.subheader("GPT Model Settings")
    gpt_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    st.session_state.gpt_model = st.sidebar.selectbox("Select GPT Model", gpt_models)

    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    max_tokens = st.sidebar.slider("Max Tokens", 50, 1000, 200, 10)

    # Add prompt input
    st.subheader("GPT Prompt")
    default_prompt = """Analyze the following incident reports and provide the following information:

List all unique incident types you can identify. Use concise, general categories that match and you may introduce new categories if necessary.

Format your response as follows:
Incident Types: [comma-separated list of incident types]

If you cannot identify any incident types, write "Unknown" for that category.
"""
    gpt_prompt = st.text_area("Customize the prompt for GPT-based incident type discovery:", 
                              value=default_prompt,
                              height=200)

    # Button to discover incident types
    if st.button("Discover Incident Types"):
        with st.spinner(f"Discovering incident types from {len(processed_documents)} documents..."):
            discovery_results = discover_incident_types(processed_documents, embeddings, gpt_prompt)
        
        st.success("Incident types discovered successfully!")
        
        # Side-by-side comparison and summary
        st.subheader("Comparison with Ground Truth")
        comparison_df, summary_df = compare_with_ground_truth(discovery_results, ORIG_INCIDENT_TYPES)
        
        st.write("Detailed Comparison:")
        st.dataframe(comparison_df)
        
        st.write("Summary of Match Percentages:")
        st.dataframe(summary_df)
        
        # Save the discovered incident information
        saved_file = save_incident_info(discovery_results, selected_metadata['index_id'])
        st.success(f"Incident information saved successfully in {saved_file}!")

else:
    st.error("Failed to load embeddings and index. Please run the Create Embeddings Index script first.")