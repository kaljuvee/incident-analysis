import streamlit as st
from datetime import datetime
import re
from openai import OpenAI
from typing import List, Dict
import json
import os
from utils.embeddings_util import select_embedding_index

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
    'Heat exhaustion',
    'Scaffolding',  # New incident type
    'Bee stings'    # New incident type
]

def discover_incident_info(documents, prompt, temperature, max_tokens):
    prompt += "\n\n".join([doc['description'] for doc in documents])
    
    response = client.chat.completions.create(
        model=st.session_state.gpt_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    content = response.choices[0].message.content
    incident_types = parse_response(content)
    return incident_types

def parse_response(content):
    incident_types = [t.strip() for t in content.split(':')[1].split(',')]
    return incident_types

def save_incident_info(incident_types, index_id):
    info = {
        'index_id': index_id,
        'incident_types': incident_types,
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

def save_prompt(prompt, version):
    filename = f"prompt_v{version}.md"
    with open(filename, 'w') as f:
        f.write(prompt)
    return filename

def load_prompt(version):
    filename = f"prompt_v{version}.md"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return f.read()
    return None

def get_latest_version():
    versions = [float(f.split('_v')[1].split('.md')[0]) for f in os.listdir() if f.startswith('prompt_v') and f.endswith('.md')]
    return max(versions) if versions else 1.0

# Initialize session state variables
if 'incident_types' not in st.session_state:
    st.session_state.incident_types = []

# Streamlit app
st.title("Discover Incident Types")

# Load embeddings and index
selected_metadata, embeddings, index, processed_documents, embedding_model = select_embedding_index()

if selected_metadata is not None:
    st.success("Embeddings, index, and processed documents loaded successfully!")

    # Display the number of documents to be analyzed
    st.info(f"Number of documents to analyze: {len(processed_documents)}")
    st.info(f"Using embedding index: {selected_metadata['index_id']}")

    # Sidebar for GPT model selection and parameters
    st.sidebar.subheader("Model Settings")
    gpt_models = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    st.session_state.gpt_model = st.sidebar.selectbox("Select GPT Model", gpt_models)
    
    # Expose temperature and max_tokens as sliders
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    max_tokens = st.sidebar.slider("Max Tokens", 50, 1000, 200, 10)

    # Default prompt
    default_prompt = """Analyze the following incident reports and provide the following information:

List all unique incident types you can identify. Use concise, general categories that match and you may introduce new categories if necessary.

Format your response as follows:
Incident Types: [comma-separated list of incident types]

If you cannot identify any incident types, write "Unknown" for that category.
"""

    # Prompt versioning
    st.subheader("Prompt Versioning")
    latest_version = get_latest_version()
    new_version = st.text_input("Enter version number (e.g., 1.0, 1.1)", value=f"{latest_version:.1f}")

    # Load saved prompt or use default
    load_version = st.selectbox("Load prompt version", 
                                [f"{v:.1f}" for v in sorted([float(f.split('_v')[1].split('.md')[0]) for f in os.listdir() if f.startswith('prompt_v') and f.endswith('.md')], reverse=True)],
                                index=0)
    
    if st.button("Load Selected Version"):
        loaded_prompt = load_prompt(load_version)
        if loaded_prompt:
            st.session_state.edited_prompt = loaded_prompt
            st.success(f"Prompt version {load_version} loaded successfully!")
        else:
            st.warning(f"No prompt found for version {load_version}")

    # Edit prompt
    st.subheader("Edit Prompt")
    st.session_state.edited_prompt = st.text_area("Customize the prompt for incident information discovery:", 
                                value=st.session_state.edited_prompt or default_prompt,
                                height=300,
                                key="prompt_input")

    # Save prompt button
    if st.button("Save Prompt"):
        saved_file = save_prompt(st.session_state.edited_prompt, new_version)
        st.success(f"Prompt saved successfully as {saved_file}!")

    # Button to discover incident information
    if st.button("Discover Incident Types"):
        with st.spinner(f"Discovering incident types from {len(processed_documents)} documents..."):
            st.session_state.incident_types = discover_incident_info(
                processed_documents, 
                st.session_state.edited_prompt,
                temperature,
                max_tokens
            )
        st.success("Incident types discovered successfully!")
        st.write("Discovered incident types:", st.session_state.incident_types)
        
        # Save the discovered incident information
        saved_file = save_incident_info(
            st.session_state.incident_types, 
            selected_metadata['index_id']
        )
        st.success(f"Incident types saved successfully in {saved_file}!")

    # Display current incident information
    if st.session_state.incident_types:
        st.subheader("Current Incident Types")
        st.write("Incident Types:", ", ".join(st.session_state.incident_types))
    else:
        # Try to load existing incident information
        existing_info = load_incident_info(selected_metadata['index_id'])
        if existing_info:
            st.subheader("Previously Discovered Incident Types")
            st.write("Incident Types:", ", ".join(existing_info.get('incident_types', ['None found'])))
            st.info(f"This information was discovered on {existing_info.get('timestamp', 'Unknown date')}")
        else:
            st.info("No incident types discovered yet. Click 'Discover Incident Types' to start.")
else:
    st.error("Failed to load embeddings and index. Please run the Create Embeddings Index script first.")
