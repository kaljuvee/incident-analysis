import os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
EMBEDDING_MODEL = "text-embedding-3-large"
DATA_DIR = "data/small"
PINECONE_INDEX_NAME = "incidents"

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def load_and_process_documents():
    try:
        logging.info(f"Loading documents from {DATA_DIR}")
        loader = DirectoryLoader(DATA_DIR, glob="**/*.txt")
        logging.info("DirectoryLoader created")
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} documents")

        if not documents:
            logging.error(f"No documents found in {DATA_DIR}")
            return None

        logging.info("Creating text splitter")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        logging.info("Splitting documents")
        texts = text_splitter.split_documents(documents)
        logging.info(f"Split into {len(texts)} text chunks")

        return texts
    except Exception as e:
        logging.error(f"Error in load_and_process_documents: {str(e)}")
        return None

def create_vector_store(texts):
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        logging.info("Creating new Pinecone vector store")
        
        # Create Pinecone vector store
        vector_store = LangchainPinecone.from_documents(
            documents=texts,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME
        )
        
        logging.info(f"Pinecone vector store created with index name: {PINECONE_INDEX_NAME}")
        return vector_store
    except Exception as e:
        logging.error(f"Error in create_vector_store: {str(e)}")
        return None

def main():
    try:
        logging.info("Starting the index creation process")

        texts = load_and_process_documents()
        logging.info("Document processing complete")

        if texts is None:
            logging.error("Failed to load and process documents. Exiting.")
            return

        logging.info(f"Number of text chunks: {len(texts)}")

        vector_store = create_vector_store(texts)
        logging.info("Vector store creation complete")

        if vector_store is None:
            logging.error("Failed to create vector store. Exiting.")
            return

        logging.info("Index creation complete.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()