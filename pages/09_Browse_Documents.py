import streamlit as st
from utils import db_util
import math

st.title("Browse Incident Documents")

# Get all dataset IDs with their document counts
datasets = db_util.get_datasets_with_counts()

# Create a dropdown for dataset selection
dataset_options = [f"{dataset_id} ({count} documents)" for dataset_id, count in datasets]
selected_dataset = st.selectbox("Select a dataset:", dataset_options)

if selected_dataset:
    # Extract the dataset ID from the selected option
    selected_dataset_id = selected_dataset.split(" ")[0]
    
    # Load documents for the selected dataset
    documents = db_util.read_documents_by_dataset(selected_dataset_id)
    
    # Pagination
    docs_per_page = 100
    total_pages = math.ceil(len(documents) / docs_per_page)
    page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    start_idx = (page_number - 1) * docs_per_page
    end_idx = start_idx + docs_per_page

    # Browse functionality
    st.subheader(f"Browse Documents (Page {page_number} of {total_pages})")
    for i, doc in enumerate(documents[start_idx:end_idx], start=start_idx + 1):
        with st.expander(f"Document {i}"):
            st.write(doc)

    # Pagination controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Previous Page") and page_number > 1:
            st.rerun()
    with col3:
        if st.button("Next Page") and page_number < total_pages:
            st.rerun()
