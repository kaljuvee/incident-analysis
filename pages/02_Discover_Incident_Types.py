import streamlit as st
from datetime import datetime
import re
from openai import OpenAI
from typing import List, Dict
import json
import os
from utils.embeddings_util import select_embedding_index
from utils.db_util import get_datasets_with_counts

# Load environment variables and set up OpenAI client
client = OpenAI()

ORIG_INCIDENT_TYPES = [
    'Slip', 'Fire', 'Safety violation', 'Chemical spill', 'Injury',
    'Near miss', 'Electrical', 'Ventilation', 'Falling object', 'Heat exhaustion'
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
    incident_types, plants, causes = parse_response(content)
    return incident_types, plants, causes

def parse_response(content):
    sections = content.split('\n\n')
    incident_types = [t.strip() for t in sections[0].split(':')[1].split(',')]
    plants = [p.strip() for p in sections[1].split(':')[1].split(',')]
    causes = [c.strip() for c in sections[2].split(':')[1].split(',')]
    return incident_types, plants, causes

def save_incident_info(incident_types, plants, causes, index_id):
    info = {
        'index_id': index_id,
        'incident_types': incident_types,
        'plants': plants,
        'causes': causes,
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
if 'plants' not in st.session_state:
    st.session_state.plants = []
if 'causes' not in st.session_state:
    st.session_state.causes = []
if 'saved_prompt' not in st.session_state:
    st.session_state.saved_prompt = None
if 'edited_prompt' not in st.session_state:
    st.session_state.edited_prompt = None

# Streamlit app
st.title("Discover Incident Types, Plants, and Causes")

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

1. List all unique incident types you can identify. Use concise, general categories that match the following list as closely as possible:
Slip, Fire, Safety violation, Chemical spill, Injury, Near miss, Electrical, Ventilation, Falling object, Heat exhaustion
You may introduce new categories if necessary.

2. List all unique plants mentioned in the incidents.

3. List all unique causes of the incidents.

Format your response as follows:
Incident Types: [comma-separated list of incident types]

Plants: [comma-separated list of plants]

Causes: [comma-separated list of causes]

If you cannot identify any items in a category, write "Unknown" for that category.
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
    if st.button("Discover Incident Information"):
        with st.spinner(f"Discovering incident information from {len(processed_documents)} documents..."):
            st.session_state.incident_types, st.session_state.plants, st.session_state.causes = discover_incident_info(
                processed_documents, 
                st.session_state.edited_prompt,
                temperature,
                max_tokens
            )
        st.success("Incident information discovered successfully!")
        st.write("Discovered incident types:", st.session_state.incident_types)
        st.write("Discovered plants:", st.session_state.plants)
        st.write("Discovered causes:", st.session_state.causes)
        
        # Save the discovered incident information
        saved_file = save_incident_info(
            st.session_state.incident_types, 
            st.session_state.plants, 
            st.session_state.causes, 
            selected_metadata['index_id']
        )
        st.success(f"Incident information saved successfully in {saved_file}!")

    # Display current incident information
    if st.session_state.incident_types or st.session_state.plants or st.session_state.causes:
        st.subheader("Current Incident Information")
        st.write("Incident Types:", ", ".join(st.session_state.incident_types))
        st.write("Plants:", ", ".join(st.session_state.plants))
        st.write("Causes:", ", ".join(st.session_state.causes))
    else:
        # Try to load existing incident information
        existing_info = load_incident_info(selected_metadata['index_id'])
        if existing_info:
            st.subheader("Previously Discovered Incident Information")
            st.write("Incident Types:", ", ".join(existing_info['incident_types']))
            st.write("Plants:", ", ".join(existing_info['plants']))
            st.write("Causes:", ", ".join(existing_info['causes']))
            st.info(f"This information was discovered on {existing_info['timestamp']}")
        else:
            st.info("No incident information discovered yet. Click 'Discover Incident Information' to start.")
else:
    st.error("Failed to load embeddings and index. Please run the Create Embeddings Index script first.")
