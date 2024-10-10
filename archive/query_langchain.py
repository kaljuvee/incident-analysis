import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_pinecone import Pinecone as LangchainPinecone
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
MODEL_NAME = "gpt-4"
EMBEDDING_MODEL = "text-embedding-3-large"
PINECONE_INDEX_NAME = "incidents"

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def load_vector_store():
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = LangchainPinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)
        logging.info(f"Loaded Pinecone vector store: {PINECONE_INDEX_NAME}")
        return vector_store
    except Exception as e:
        logging.error(f"Error in load_vector_store: {str(e)}")
        return None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def count_incidents(query_incident, vector_store):
    logging.info(f"Counting incidents for query: '{query_incident}'")
    
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    
    # Use a pre-made prompt from the hub
    prompt = hub.pull("rlm/rag-prompt")
    
    # Use the vector store for retrieval
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Construct the LCEL chain
    qa_chain = (
        {
            "context": lambda x: format_docs(retriever.get_relevant_documents(x)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        logging.info("Invoking qa_chain")
        result = qa_chain.invoke(f"Count how many times the incident type '{query_incident}' is mentioned in the relevant documents. Your response should be ONLY a single number. If no incidents of the specified type are found, respond with '0'. Do not include any additional text or explanation.")
        logging.info(f"qa_chain result: {result}")
        
        count = result.strip()
        logging.info(f"Raw count result: {count}")
        
        try:
            count = int(count)
            logging.info(f"Converted count to integer: {count}")
        except ValueError:
            logging.warning(f"Unable to convert result to integer: {count}")
            count = 0
        
        logging.info(f"Final incident count: {count}")
        return count
    except Exception as e:
        logging.error(f"Error in count_incidents: {str(e)}")
        return 0

if __name__ == "__main__":
    logging.info("Starting the query process")
    
    # Load vector store
    vector_store = load_vector_store()
    if vector_store:
        query_incident = 'fire'
        incident_count = count_incidents(query_incident, vector_store)
        logging.info(f"Analysis complete. Total count for '{query_incident}' incidents: {incident_count}")
        print(f"Total count for '{query_incident}' incidents: {incident_count}")
    else:
        logging.error("Failed to load vector store. Cannot proceed with the analysis.")
