import streamlit as st
import pandas as pd
from typing import List, Dict
from difflib import SequenceMatcher
import json
from utils.embeddings_util import select_embedding_index
from utils.analysis_util import normalize_string, extract_date, extract_plant, plot_incident_stats

ORIG_INCIDENT_TYPES = [
    'Slip', 'Fire', 'Safety violation', 'Chemical spill', 'Injury',
    'Near miss', 'Electrical', 'Ventilation', 'Falling object', 'Heat exhaustion'
]

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
            if ratio > best_ratio and ratio > 0.8:  # Increased threshold for close matches
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
            if normalize_string(incident_type) in normalize_string(text):
                return incident_type
        return 'other'
    
    df['incident_type'] = df['incident'].apply(categorize_incident)
    
    correlation_matrix = pd.crosstab(df['incident_type'], df['incident_type'].shift())
    
    return {
        'incident_rates_per_plant': incident_rates.to_dict(),
        'monthly_incident_counts': monthly_incidents.to_dict(),
        'incident_type_correlations': correlation_matrix.to_dict()
    }

def load_incident_types():
    try:
        with open('incident_types.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# Streamlit app
st.title("Evaluate Incident Analysis")

# Sidebar for GPT model selection
st.sidebar.subheader("Model Settings")
gpt_models = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
st.session_state.gpt_model = st.sidebar.selectbox("Select GPT Model", gpt_models)
st.sidebar.info("Note: The selected GPT model is not used in the current analysis. It's included for consistency with other scripts and potential future use.")

# Load embeddings and index
selected_metadata, embeddings, index, processed_documents, embedding_model = select_embedding_index()

if selected_metadata is not None:
    st.success("Embeddings, index, and processed documents loaded successfully!")

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
                counts = count_incident_types(processed_documents, st.session_state.incident_types)
            st.success("Incidents counted successfully!")
            st.header("Incident Type Counts")
            st.bar_chart(counts)

    # Button to perform pattern analysis
    if st.button("Analyze Patterns"):
        if 'incident_types' not in st.session_state or not st.session_state.incident_types:
            st.warning("Please enter incident types first.")
        else:
            with st.spinner("Analyzing patterns..."):
                analysis_results = analyze_patterns(processed_documents, st.session_state.incident_types)
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

    # New button to perform additional analysis
    if st.button("Analyze"):
        if 'incident_types' not in st.session_state or not st.session_state.incident_types:
            st.warning("Please enter incident types first.")
        else:
            with st.spinner("Performing additional analysis..."):
                # Create DataFrame from processed_documents
                df = pd.DataFrame([
                    {
                        'Incident Type': doc.get('Incident Type', 'Unknown'),
                        'Cause': doc.get('Cause', 'Unknown'),
                        'Plant': extract_plant(doc['description']),
                        'Date': extract_date(doc['description']),
                        'Description': doc['description']
                    }
                    for doc in processed_documents
                ])
                
                # Perform analysis and display results
                plot_incident_stats(df)

                # Number of incidents over date/time
                st.subheader("Number of Incidents Over Time")
                incidents_over_time = df.groupby(df['Date'].dt.to_period('M')).size()
                st.line_chart(incidents_over_time)

            st.success("Additional analysis completed successfully!")

else:
    st.error("Failed to load embeddings and index. Please run the Create Embeddings Index script first.")
