import streamlit as st
import pandas as pd
from datetime import datetime
import re
from typing import List, Dict
from difflib import SequenceMatcher
import pickle
import json

ORIG_INCIDENT_TYPES = [
    'slip', 'fire', 'safety_violation', 'chemical_spill', 'injury', 'near_miss',
    'electrical', 'ventilation', 'falling object', 'heat exhaustion'
]

def compare_incident_types(original, discovered):
    if not discovered:
        return {
            'exact_matches': [],
            'close_matches': [],
            'unmatched_original': original,
            'unmatched_discovered': []
        }
    
    original_formatted = [o.replace('-', ' ') for o in original]
    
    exact_matches = set(original_formatted) & set(discovered)
    
    close_matches = []
    unmatched_original = [o for o in original_formatted if o not in exact_matches]
    unmatched_discovered = [d for d in discovered if d not in exact_matches]
    
    for o in unmatched_original:
        best_match = None
        best_ratio = 0
        for d in unmatched_discovered:
            ratio = SequenceMatcher(None, o, d).ratio()
            if ratio > best_ratio and ratio > 0.6:
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

def count_incident_types(documents, incident_types):
    counts = {incident: sum(1 for doc in documents if incident.lower() in doc['description'].lower()) for incident in incident_types}
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
            'date': extract_date(doc['description']),
            'plant': extract_plant(doc['description']),
            'incident': doc['description']
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

# Load saved data
def load_data():
    with open('processed_documents.pkl', 'rb') as f:
        return pickle.load(f)

# Load saved incident types
def load_incident_types():
    try:
        with open('incident_types.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# Streamlit app
st.title("Evaluate Incident Analysis")

# Load saved data
try:
    st.session_state.processed_documents = load_data()
    st.success("Processed documents loaded successfully!")
except FileNotFoundError:
    st.error("Saved data not found. Please run the Data Preparation and Ground Truth Analysis scripts first.")
    st.stop()

# Load saved incident types
saved_incident_types = load_incident_types()
if saved_incident_types:
    st.session_state.incident_types = saved_incident_types
    st.success("Saved incident types loaded successfully!")
else:
    st.warning("No saved incident types found. Please enter them manually or run the Ground Truth Analysis script first.")

# Input for incident types (only if not loaded from file)
if 'incident_types' not in st.session_state or not st.session_state.incident_types:
    incident_types_input = st.text_input("Enter incident types (comma-separated):")
    if incident_types_input:
        st.session_state.incident_types = [t.strip() for t in incident_types_input.split(',')]

# Button to compare incident types
if st.button("Compare Incident Types"):
    if 'incident_types' not in st.session_state or not st.session_state.incident_types:
        st.warning("Please enter incident types first.")
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
    if 'incident_types' not in st.session_state or not st.session_state.incident_types:
        st.warning("Please enter incident types first.")
    else:
        with st.spinner("Counting incidents..."):
            counts = count_incident_types(st.session_state.processed_documents, st.session_state.incident_types)
        st.success("Incidents counted successfully!")
        st.header("Incident Type Counts")
        st.bar_chart(counts)

# Button to perform pattern analysis
if st.button("Analyze Patterns"):
    if 'incident_types' not in st.session_state or not st.session_state.incident_types:
        st.warning("Please enter incident types first.")
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
