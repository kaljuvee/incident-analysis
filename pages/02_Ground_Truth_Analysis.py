import streamlit as st
from datetime import datetime
import re
from openai import OpenAI
from typing import List, Dict
import json
from utils.embeddings_util import select_embedding_index

# Load environment variables and set up OpenAI client
client = OpenAI()

ORIG_INCIDENT_TYPES = [
    'slip', 'fire', 'safety_violation', 'chemical_spill', 'injury', 'near_miss',
    'electrical', 'ventilation', 'falling_object', 'heat_exhaustion'
]

def discover_incident_types(documents, num_docs, prompt):
    prompt += "\n\n".join([doc['description'] for doc in documents[:num_docs]])
    
    response = client.chat.completions.create(
        model=st.session_state.gpt_model,
        messages=[{"role": "user", "content": prompt}]
    )
    incident_types = [t.strip() for t in response.choices[0].message.content.split(',')]
    return incident_types

def save_incident_types(incident_types):
    with open('incident_types.json', 'w') as f:
        json.dump(incident_types, f)

# Initialize session state variables
if 'incident_types' not in st.session_state:
    st.session_state.incident_types = []
if 'saved_prompt' not in st.session_state:
    st.session_state.saved_prompt = None
if 'edited_prompt' not in st.session_state:
    st.session_state.edited_prompt = None

# Streamlit app
st.title("Ground Truth Analysis for Incidents")

# Load embeddings and index
selected_metadata, embeddings, index, processed_documents, embedding_model = select_embedding_index()

if selected_metadata is not None:
    st.success("Embeddings, index, and processed documents loaded successfully!")

    # Sidebar for GPT model selection
    gpt_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    st.session_state.gpt_model = st.sidebar.selectbox("Select GPT Model", gpt_models)

    # Slider for number of documents to analyze
    num_docs_to_analyze = st.slider("Number of documents to analyze", 
                                    min_value=1, 
                                    max_value=len(processed_documents), 
                                    value=min(5, len(processed_documents)), 
                                    step=1)

    # Default prompt
    default_prompt = """Analyze the following incident reports and list all unique incident types you can identify. 
    List only the incident types found, separated by commas. 
    Format multi-word types as word_anotherword (e.g., 'safety violation' should be 'safety_violation').
    Here are some example formats: slip, fire, safety_violation, chemical_spill, injury, near_miss, 
    electrical, ventilation, falling_object, heat_exhaustion.

    """

    # Load saved prompt or use default
    current_prompt = st.session_state.saved_prompt if st.session_state.saved_prompt else default_prompt

    st.subheader("Edit Prompt")
    st.session_state.edited_prompt = st.text_area("Customize the prompt for incident type discovery:", 
                                value=current_prompt,
                                height=200,
                                key="prompt_input")

    # Save prompt button
    if st.button("Save Prompt"):
        st.session_state.saved_prompt = st.session_state.edited_prompt
        st.success("Prompt saved successfully!")

    # Load saved prompt button
    if st.button("Load Saved Prompt"):
        if st.session_state.saved_prompt:
            st.session_state.edited_prompt = st.session_state.saved_prompt
            st.rerun()
        else:
            st.warning("No saved prompt found. Save a prompt first.")

    # Button to discover incident types
    if st.button("Discover Incident Types"):
        with st.spinner(f"Discovering incident types from {num_docs_to_analyze} documents..."):
            st.session_state.incident_types = discover_incident_types(processed_documents, num_docs_to_analyze, st.session_state.edited_prompt)
        st.success("Incident types discovered successfully!")
        st.write("Discovered incident types:", st.session_state.incident_types)
        
        # Save the discovered incident types
        save_incident_types(st.session_state.incident_types)
        st.success("Incident types saved successfully!")

    # Display current incident types
    if st.session_state.incident_types:
        st.subheader("Current Incident Types")
        st.write(", ".join(st.session_state.incident_types))
    else:
        st.info("No incident types discovered yet. Click 'Discover Incident Types' to start.")
else:
    st.error("Failed to load embeddings and index. Please run the Create Embeddings Index script first.")
