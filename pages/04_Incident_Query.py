import streamlit as st
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import pickle

load_dotenv()
client = OpenAI()

def load_embeddings_and_index():
    with open('embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    index = faiss.read_index('faiss_index.bin')
    with open('processed_documents.pkl', 'rb') as f:
        processed_documents = pickle.load(f)
    return embeddings, index, processed_documents

def retrieve_relevant_docs(query: str, index, processed_documents, embedding_model: str, top_k: int = 5):
    query_embedding = client.embeddings.create(
        model=embedding_model,
        input=query
    ).data[0].embedding
    
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [processed_documents[i] for i in indices[0]]

def main():
    st.title("Question Answering with RAG")

    embeddings, index, processed_documents = load_embeddings_and_index()

    embedding_model = st.selectbox("Select Embedding Model", ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"])
    gpt_model = st.selectbox("Select GPT Model", ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"])

    query = st.text_input("Enter your question about the incidents:")
    if st.button("Answer Question"):
        if query:
            with st.spinner("Answering question..."):
                relevant_docs = retrieve_relevant_docs(query, index, processed_documents, embedding_model)
                context = "\n".join([doc['content'] for doc in relevant_docs])
                
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
                    st.write(doc['content'])
                    st.write("---")
            st.success("Question answered using RAG!")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
