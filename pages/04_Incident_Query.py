import streamlit as st
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import pickle
import os
from utils.db_util import get_datasets_with_counts

load_dotenv()
client = OpenAI()

def load_embeddings_and_index(dataset_id):
    try:
        with open(f'embeddings/embeddings_{dataset_id}.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        index = faiss.read_index(f'embeddings/index_{dataset_id}.bin')
        with open(f'embeddings/processed_documents_{dataset_id}.pkl', 'rb') as f:
            processed_documents = pickle.load(f)
        return embeddings, index, processed_documents
    except FileNotFoundError:
        st.error(f"Embeddings or index not found for dataset {dataset_id}. Please run the Create Embeddings Index script for this dataset first.")
        return None, None, None

def retrieve_relevant_docs(query: str, index, processed_documents, top_k: int = 5):
    query_embedding = client.embeddings.create(
        model="text-embedding-ada-002",  # Using a default model, change if necessary
        input=query
    ).data[0].embedding
    
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [processed_documents[i] for i in indices[0]]

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

    embeddings, index, processed_documents = load_embeddings_and_index(selected_dataset)

    if embeddings is not None and index is not None and processed_documents is not None:
        gpt_model = st.selectbox("Select GPT Model", ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"])

        query = st.text_input("Enter your question about the incidents:")
        if st.button("Answer Question"):
            if query:
                with st.spinner("Answering question..."):
                    relevant_docs = retrieve_relevant_docs(query, index, processed_documents)
                    context = "\n".join([doc['description'] for doc in relevant_docs])
                    
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
                st.success("Question answered using RAG!")
            else:
                st.warning("Please enter a question.")
    else:
        st.warning(f"Please run the Create Embeddings Index script for dataset {selected_dataset} before using this query interface.")
