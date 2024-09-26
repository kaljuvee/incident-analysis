import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from gptcache import cache
import faiss
from typing import List, Dict
import pickle
from utils.db_util import get_datasets_with_counts, read_documents_by_dataset
import json
import os

# Load environment variables and set up OpenAI client
load_dotenv()
client = OpenAI()
cache.init()
cache.set_openai_key()

# Define available models
EMBEDDING_MODELS = [
    "text-embedding-3-large",
    "text-embedding-ada-002",
    "text-embedding-3-small"
]

# Initialize session state variables
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def load_documents(dataset_id: str) -> List[Dict[str, str]]:
    documents = read_documents_by_dataset(dataset_id)
    parsed_documents = []
    for doc in documents:
        try:
            parsed_doc = json.loads(doc)
            parsed_documents.append({
                'description': parsed_doc.get('Description', ''),
                'source': parsed_doc.get('source', 'Unknown')
            })
        except json.JSONDecodeError:
            st.warning(f"Failed to parse document: {doc[:100]}...")  # Show first 100 characters of problematic document
    return parsed_documents

def chunk_documents(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    chunked_documents = []
    for doc in documents:
        chunks = chunk_text(doc['description'])
        for chunk in chunks:
            chunked_documents.append({
                'description': chunk,
                'source': doc['source']
            })
    return chunked_documents

@st.cache_resource
def create_embeddings(docs: List[Dict[str, str]], model: str):
    embeddings = []
    for doc in docs:
        response = client.embeddings.create(
            model=model,
            input=doc['description']
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)

def create_faiss_index(embeddings: np.ndarray):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_embeddings_and_index(embeddings, index, processed_documents, dataset_id):
    os.makedirs('embeddings', exist_ok=True)
    with open(f'embeddings/embeddings_{dataset_id}.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    faiss.write_index(index, f'embeddings/index_{dataset_id}.bin')
    with open(f'embeddings/processed_documents_{dataset_id}.pkl', 'wb') as f:
        pickle.dump(processed_documents, f)

# Streamlit app
st.title("Create Embeddings and Index for Incident Analysis")

# Sidebar for GPT model selection
gpt_models = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
st.session_state.gpt_model = st.sidebar.selectbox("Select GPT Model", gpt_models)

# Sidebar for vectorization method selection
st.session_state.embedding_model = st.sidebar.selectbox(
    "Select Embedding Model",
    options=EMBEDDING_MODELS,
    index=EMBEDDING_MODELS.index(st.session_state.get('embedding_model', EMBEDDING_MODELS[0]))
)

# Sidebar for document processing option
st.session_state.use_chunking = st.sidebar.radio("Document Processing", ["Full Documents", "Chunked Documents"]) == "Chunked Documents"

# Main content
# Step 1: Select dataset
datasets_with_counts = get_datasets_with_counts()

if not datasets_with_counts:
    st.warning("No datasets found in the database.")
else:
    dataset_options = {f"{data_set_id} ({count} documents)": data_set_id for data_set_id, count in datasets_with_counts}
    selected_dataset_option = st.selectbox(
        "Select a dataset to analyze",
        options=list(dataset_options.keys()),
        key="dataset_selection"
    )
    selected_dataset = dataset_options[selected_dataset_option]

    if st.button("Load Selected Dataset"):
        with st.spinner("Loading documents..."):
            documents = load_documents(selected_dataset)
            if st.session_state.use_chunking:
                st.session_state.processed_documents = chunk_documents(documents)
                st.success(f"Documents loaded and chunked successfully! Total chunks: {len(st.session_state.processed_documents)}")
            else:
                st.session_state.processed_documents = documents
                st.success(f"Documents loaded successfully! Total documents: {len(st.session_state.processed_documents)}")

# Step 2: Create embeddings
if st.session_state.processed_documents is not None and st.session_state.embeddings is None:
    if st.button("Create Embeddings"):
        with st.spinner(f"Creating embeddings using {st.session_state.embedding_model}..."):
            st.session_state.embeddings = create_embeddings(st.session_state.processed_documents, st.session_state.embedding_model)
        st.success("Embeddings created successfully!")
elif st.session_state.embeddings is not None:
    st.success("Embeddings are already created.")

# Step 3: Create FAISS index
if st.session_state.embeddings is not None and st.session_state.faiss_index is None:
    if st.button("Create Index"):
        with st.spinner("Creating an index..."):
            st.session_state.faiss_index = create_faiss_index(st.session_state.embeddings)
        st.success("Index created successfully!")
elif st.session_state.faiss_index is not None:
    st.success("Index is already created.")

# Save embeddings and index
if st.session_state.embeddings is not None and st.session_state.faiss_index is not None:
    if st.button("Save Embeddings and Index"):
        with st.spinner("Saving embeddings and index..."):
            save_embeddings_and_index(st.session_state.embeddings, st.session_state.faiss_index, st.session_state.processed_documents, selected_dataset)
        st.success(f"Embeddings and index saved successfully for dataset {selected_dataset}!")
