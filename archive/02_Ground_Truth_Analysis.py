import streamlit as st
from datetime import datetime
import re
from openai import OpenAI
from typing import List, Dict
import json
from utils.embeddings_util import select_embedding_index
from utils.db_util import get_datasets_with_counts

# Load environment variables and set up OpenAI client
client = OpenAI()

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
    'Heat exhaustion'
]

def discover_incident_types(documents, prompt):
    prompt += "\n\n".join([doc['description'] for doc in documents])
    
    response = client.chat.completions.create(
        model=st.session_state.gpt_model,
        messages=[{"role": "user", "content": prompt}],
        #temperature=0.7,  # Lower temperature for more focused outputs
        #max_tokens=500,  # Limit token count to encourage concise responses
        #stop=[",", "."]  # Stop generation at commas or periods to get clean list items
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

    # Display the number of documents to be analyzed
    st.info(f"Number of documents to analyze: {len(processed_documents)}")

    # Sidebar for GPT model selection
    gpt_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    st.session_state.gpt_model = st.sidebar.selectbox("Select GPT Model", gpt_models)

    # Default prompt
    default_prompt = """Analyze the following incident reports and list all unique incident types you can identify. Use concise, general categories that match the following list as closely as possible, but you may introduce new categories if necessary:
    List only the incident types found, separated by commas. If you cannot identify any incident types, assign a type that is most likely to be the incident type.
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
        with st.spinner(f"Discovering incident types from {len(processed_documents)} documents..."):
            st.session_state.incident_types = discover_incident_types(processed_documents, st.session_state.edited_prompt)
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
