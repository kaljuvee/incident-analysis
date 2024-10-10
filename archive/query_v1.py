import os
import logging
import sys
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # This import should work if you have the latest LangChain

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
MODEL_NAME = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-large"
DATA_DIR = "data/small"
INDEX_PATH = "faiss_index"

def load_and_process_documents():
    try:
        logging.info(f"Loading documents from {DATA_DIR}")
        loader = DirectoryLoader(DATA_DIR, glob="**/*.txt")
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} documents")

        if not documents:
            logging.error(f"No documents found in {DATA_DIR}")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        logging.info(f"Split into {len(texts)} text chunks")

        return texts
    except Exception as e:
        logging.error(f"Error in load_and_process_documents: {str(e)}")
        return None

def get_vector_store(texts):
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

        if os.path.exists(INDEX_PATH):
            logging.info("Loading existing vector store")
            vector_store = FAISS.load_local(INDEX_PATH, embeddings)
        else:
            logging.info("Creating new vector store")
            vector_store = FAISS.from_documents(texts, embeddings)
            vector_store.save_local(INDEX_PATH)

        return vector_store
    except Exception as e:
        logging.error(f"Error in get_vector_store: {str(e)}")
        return None

def count_incidents(query_incident, retriever):
    logging.info(f"Counting incidents for query: '{query_incident}'")
    
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    
    prompt_template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: Count how many times the incident type '{query_incident}' is mentioned in the relevant documents. 
    Provide the response as a single number. If no incidents of the specified type are found, respond with '0'.
    
    Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "query_incident"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = qa_chain({"query_incident": query_incident})
    count = int(result['result'].strip())
    logging.info(f"Incident count: {count}")
    return count

def main():
    logging.info("Starting the analysis process")

    texts = load_and_process_documents()
    if texts is None:
        logging.error("Failed to load and process documents. Exiting.")
        return

    vector_store = get_vector_store(texts)
    if vector_store is None:
        logging.error("Failed to create or load vector store. Exiting.")
        return

    query_incident = "fire"
    logging.info(f"Query incident set to: '{query_incident}'")

    try:
        total_count = count_incidents(query_incident, vector_store.as_retriever())
        logging.info(f"Analysis complete. Total count for '{query_incident}' incidents: {total_count}")
        print(f"Total count for '{query_incident}' incidents: {total_count}")
    except Exception as e:
        logging.error(f"Error during incident counting: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Unhandled exception in main: {str(e)}")
        sys.exit(1)
