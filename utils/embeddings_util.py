import streamlit as st
import os
import json
import faiss
import pickle
from typing import List, Dict, Tuple, Any

def load_all_metadata() -> List[Dict[str, Any]]:
    all_metadata = []
    for file in os.listdir('embeddings'):
        if file.endswith("_metadata.json"):
            with open(os.path.join('embeddings', file), 'r') as f:
                metadata = json.load(f)
                all_metadata.append(metadata)
    return all_metadata

def load_embeddings_and_index(metadata: Dict[str, Any]) -> Tuple[Any, Any, Any, str]:
    try:
        dataset_id = metadata['dataset_id']
        index_id = metadata['index_id']
        with open(f'embeddings/{dataset_id}_{index_id}_embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        index = faiss.read_index(f'embeddings/{dataset_id}_{index_id}_index.bin')
        with open(f'embeddings/{dataset_id}_{index_id}_processed_documents.pkl', 'rb') as f:
            processed_documents = pickle.load(f)
        return embeddings, index, processed_documents, metadata['embedding_model']
    except FileNotFoundError:
        st.error(f"Embeddings or index not found for dataset {dataset_id} and index {index_id}.")
        return None, None, None, None

def select_embedding_index() -> Tuple[Dict[str, Any], Any, Any, Any, str]:
    all_metadata = load_all_metadata()
    
    if not all_metadata:
        st.warning("No embeddings found. Please run the Create Embeddings Index script first.")
        return None, None, None, None, None
    
    embedding_options = [
        f"Model: {m['embedding_model']} | Dataset: {m['dataset_id']} | ID: {m['index_id'][:8]}"
        for m in all_metadata
    ]
    selected_embedding_option = st.selectbox(
        "Select an embedding index",
        options=embedding_options,
        key="embedding_selection"
    )
    
    selected_metadata = all_metadata[embedding_options.index(selected_embedding_option)]
    
    embeddings, index, processed_documents, embedding_model = load_embeddings_and_index(selected_metadata)
    
    if embeddings is not None and index is not None and processed_documents is not None and embedding_model is not None:
        st.info(f"Using embedding model: {embedding_model}")
    else:
        st.warning(f"Failed to load embeddings for the selected set. Please try another or run the Create Embeddings Index script again.")
    
    return selected_metadata, embeddings, index, processed_documents, embedding_model
