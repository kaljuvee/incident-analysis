import streamlit as st
import pandas as pd
from typing import List, Dict
from difflib import SequenceMatcher
import json
import os
from utils.embeddings_util import select_embedding_index
from utils.analysis_util import normalize_string, extract_date, extract_plant, plot_incident_stats, categorize_incident

def load_incident_info(index_id):
    filename = f'embeddings/incident_info_{index_id}.json'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def compare_incident_types(original, discovered):
    if not discovered:
        return {
            'exact_matches': [],
            'close_matches': [],
            'unmatched_original': original,
            'unmatched_discovered': []
        }
    
    original_formatted = [normalize_string(o.replace('-', ' ')) for o in original]
    discovered_formatted = [normalize_string(d) for d in discovered]
    
    exact_matches = set(original_formatted) & set(discovered_formatted)
    
    close_matches = []
    unmatched_original = [o for o in original_formatted if o not in exact_matches]
    unmatched_discovered = [d for d in discovered_formatted if d not in exact_matches]
    
    for o in unmatched_original.copy():
        best_match = None
        best_ratio = 0
        for d in unmatched_discovered:
            ratio = SequenceMatcher(None, o, d).ratio()
            if ratio > best_ratio and ratio > 0.8:
                best_ratio = ratio
                best_match = d
        if best_match:
            close_matches.append((o, best_match))
            unmatched_discovered.remove(best_match)
            unmatched_original.remove(o)
    
    return {
        'exact_matches': list(exact_matches),
        'close_matches': close_matches,
        'unmatched_original': unmatched_original,
        'unmatched_discovered': unmatched_discovered
    }

def count_incident_types(documents, incident_types):
    counts = {incident: sum(1 for doc in documents if normalize_string(incident) in normalize_string(doc['description'])) for incident in incident_types}
    return counts

# Streamlit app
st.title("Evaluate Incident Analysis")

# Load embeddings and index
selected_metadata, embeddings, index, processed_documents, embedding_model = select_embedding_index()

if selected_metadata is not None:
    st.success("Embeddings, index, and processed documents loaded successfully!")

    # Load incident info
    incident_info = load_incident_info(selected_metadata['index_id'])
    if incident_info:
        st.success("Incident information loaded successfully!")
        st.write("Discovered on:", incident_info['timestamp'])
        st.write("Incident Types:", ", ".join(incident_info['incident_types']))
        st.write("Plants:", ", ".join(incident_info['plants']))
        st.write("Causes:", ", ".join(incident_info['causes']))
    else:
        st.warning("No incident information found. Please run the Discover Incident Types script first.")

    # ... (keep other parts of the script unchanged)

    # New button to perform additional analysis
# Inside the "Analyze" button function
if st.button("Analyze"):
    if incident_info:
        try:
            with st.spinner("Performing additional analysis..."):
                # Create DataFrame from processed_documents
                df = pd.DataFrame([
                    {
                        'Incident Type': categorize_incident(doc['description'], incident_info['incident_types']),
                        'Cause': next((cause for cause in incident_info['causes'] if normalize_string(cause) in normalize_string(doc['description'])), 'Unknown'),
                        'Plant': next((plant for plant in incident_info['plants'] if plant in doc['description']), extract_plant(doc['description'])),
                        'Date': extract_date(doc['description']),
                        'Description': doc['description']
                    }
                    for doc in processed_documents
                ])
                
                # Ensure 'Date' column is datetime
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Count incident types
                incident_counts = df['Incident Type'].value_counts().to_dict()
                
                # Ensure all incident types are represented in the counts
                for incident_type in incident_info['incident_types']:
                    if incident_type not in incident_counts:
                        incident_counts[incident_type] = 0
                
                # Perform analysis and display results
                plot_incident_stats(df, incident_counts, incident_info['plants'], incident_info['causes'])

            st.success("Additional analysis completed successfully!")
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.error("Please check your data and try again.")
    else:
        st.warning("No incident information available for analysis.")