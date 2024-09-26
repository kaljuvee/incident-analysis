import streamlit as st
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import pickle
import os
import json
from utils.db_util import get_datasets_with_counts
from datetime import datetime

load_dotenv()
client = OpenAI()

def load_all_metadata(dataset_id):
    metadata_files = [f for f in os.listdir('embeddings') if f.startswith(f"{dataset_id}_") and f.endswith("_metadata.json")]
    all_metadata = []
    for file in metadata_files:
        with open(os.path.join('embeddings', file), 'r') as f:
            metadata = json.load(f)
            all_metadata.append(metadata)
    return all_metadata

def load_embeddings_and_index(metadata):
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

def retrieve_relevant_docs(query: str, index, processed_documents, embedding_model, top_k: int = 5):
    try:
        query_embedding = client.embeddings.create(
            model=embedding_model,
            input=query
        ).data[0].embedding
        
        query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
        
        if query_embedding.shape[1] != index.d:
            st.error(f"Embedding dimension mismatch. Query: {query_embedding.shape[1]}, Index: {index.d}")
            return []

        distances, indices = index.search(query_embedding, top_k)
        return [processed_documents[i] for i in indices[0]]
    except Exception as e:
        st.error(f"An error occurred while retrieving relevant documents: {str(e)}")
        return []

st.title("Question Answering with RAG")

# Get available datasets
datasets_with_counts = get_datasets_with_counts()

if not datasets_with_counts:
    st.warning("No datasets found. Please run the Create Embeddings Index script first.")
else:
    dataset_options = {f"{data_set_id} ({count} documents)": data_set_id for data_set_id, count in datasets_with_counts}
    selected_dataset_option = st.selectbox(
        "Select a dataset to query",
        options=list(dataset_options.keys()),
        key="dataset_selection"
    )
    selected_dataset = dataset_options[selected_dataset_option]

    all_metadata = load_all_metadata(selected_dataset)
    
    if not all_metadata:
        st.warning(f"No embeddings found for dataset {selected_dataset}. Please run the Create Embeddings Index script for this dataset.")
    else:
        embedding_options = [
            f"Model: {m['embedding_model']} | dataset: {m.get('dataset_id', 'Unknown')} | ID: {m['index_id'][:8]}"
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
            gpt_model = st.selectbox("Select GPT Model", ["gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"])

            query = st.text_input("Enter your question about the incidents:")
            if st.button("Answer Question"):
                if query:
                    with st.spinner("Answering question..."):
                        relevant_docs = retrieve_relevant_docs(query, index, processed_documents, embedding_model)
                        if relevant_docs:
                            context = "\n".join([doc['description'] for doc in relevant_docs])
                            
                            try:
                                response = client.chat.completions.create(
                                    model=gpt_model,
                                    messages=[
                                        {"role": "system", "content": "You are a helpful assistant analyzing incident reports. Use the provided context to answer the question."},
                                        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
                                    ]
                                )
                                st.subheader("Answer:")
                                st.write(response.choices[0].message.content)
                                
                                st.subheader("Relevant Document Chunks:")
                                for i, doc in enumerate(relevant_docs, 1):
                                    st.write(f"Chunk {i} (Source: {doc['source']}):")
                                    st.write(doc['description'])
                                    st.write("---")
                            except Exception as e:
                                st.error(f"An error occurred while generating the answer: {str(e)}")
                        else:
                            st.warning("No relevant documents found for the query.")
                    st.success("Question answered using RAG!")
                else:
                    st.warning("Please enter a question.")
        else:
            st.warning(f"Failed to load embeddings for the selected set. Please try another or run the Create Embeddings Index script again.")
