import random
import pandas as pd
import streamlit as st
from faker import Faker
import plotly.express as px
import json
import datetime
from operator import itemgetter
from utils.db_util import store_documents, read_documents_by_dataset, create_tables, get_datasets_with_counts

# Initialize Faker
fake = Faker()

# Ensure database tables are created
create_tables()

incident_types = [
    'Slip',
    'Fire',
    'Safety violation',
    'Chemical spill',
    'Injury',
    'Near miss',
    'Electrical',
    'Ventilation',
    'Falling object',
    'Heat exhaustion',
    'Scaffolding',  # New incident type
    'Bee stings'    # New incident type
]

# Update the incident_descriptions with corresponding incident types
incident_descriptions = [
    ("Employee reported dizziness and nausea in the packaging area on {date}.", "Injury"),
    ("A fire alarm was triggered in the assembly line on {date}.", "Fire"),
    ("On {date}, a worker in the production area tripped over an unmarked hazard.", "Falling object"),
    ("A near-miss incident was reported on {date}.", "Near miss"),
    ("Symptoms of heat exhaustion were reported by a worker on {date}.", "Heat exhaustion"),
    ("Scaffolding material is observed removed from the structure of the north and south buildings between the north and south buildings on {date}.", "Scaffolding"),
    ("Bee hives are detected, one is located in a register and another is located on a post with the risk that personnel in the area may be stung since they are located on a pedestrian walkway on {date}.", "Bee stings")
]

# Function to plot incident statistics using Plotly
def plot_incident_stats(df):
    st.subheader("Incident Count Summary")
    
    # Calculate and display incident count by type
    incident_count_df = incident_count_by_type(df)
    incident_count_df.columns = ['Incident Type', 'Count']
    st.write(incident_count_df)

    # Plot for Incident Count by Type
    fig_count = px.bar(incident_count_df, x="Incident Type", y="Count", title="Incident Count by Type")
    st.plotly_chart(fig_count)

    # Number of incidents over date/time
    st.subheader("Number of Incidents Over Time")
    df['Date'] = pd.to_datetime(df['Date'])
    incidents_over_time = df.groupby(df['Date'].dt.to_period('M')).size().reset_index(name='Count')
    incidents_over_time['Date'] = incidents_over_time['Date'].dt.to_timestamp()
    
    fig_time = px.line(incidents_over_time, x='Date', y='Count', title="Incidents Over Time")
    st.plotly_chart(fig_time)

# Update the generate_synthetic_document function
def generate_synthetic_document(incident_id):
    date = fake.date_this_year()
    description_template, incident_type = random.choice(incident_descriptions)
    description = description_template.format(date=date)
    incident_report = {
        'Incident ID': incident_id,
        'Date': str(date),
        'Description': description,
        'Incident Type': incident_type,
        'created_at': datetime.datetime.now().isoformat()  # Add created_at field
    }
    return incident_report

# Generate documents and return DataFrame
def generate_documents(num_documents):
    data = []
    documents_to_store = []
    for i in range(1, num_documents + 1):
        document = generate_synthetic_document(i)
        data.append(document)
        documents_to_store.append(json.dumps(document))  # Convert to JSON string for storage
    
    # Store documents in the database and get the data_set_id
    data_set_id = store_documents(documents_to_store)
    
    return pd.DataFrame(data), data_set_id

# New function to calculate incident count by type
def incident_count_by_type(df):
    return df['Incident Type'].value_counts().reset_index()

# Streamlit UI starts here
st.title("Generate Synthetic Incident Data")

# Dropdown to select number of documents to generate
num_documents = st.selectbox(
    "Select number of documents to generate:",
    [100, 1000, 10000, 1000000],
    key="num_documents"
)

# Button to generate new data
if st.button("Generate New Data", key="generate_button"):
    st.write(f"Generating {num_documents} documents...")
    df, data_set_id = generate_documents(num_documents)
    st.success(f"Successfully generated and stored {num_documents} documents in the database with data_set_id: {data_set_id}")

# Get datasets with document counts
datasets_with_counts = get_datasets_with_counts()

if not datasets_with_counts:
    st.warning("No datasets found in the database.")
else:
    # Sort datasets by timestamp (newest first)
    sorted_datasets = sorted(datasets_with_counts, key=lambda x: x[2] if x[2] else datetime.datetime.min, reverse=True)
    
    # Create a dictionary for the selectbox options
    dataset_options = {f"{data_set_id} ({count} documents, {created_at.strftime('%Y-%m-%d %H:%M:%S') if created_at else 'N/A'})": data_set_id 
                       for data_set_id, count, created_at in sorted_datasets}
    
    selected_dataset_option = st.selectbox(
        "Select a dataset to view",
        options=list(dataset_options.keys()),
        key="existing_dataset"
    )
    
    # Extract the actual dataset ID from the selected option
    selected_dataset = dataset_options[selected_dataset_option]

    if st.button("View data", key="view_data_button"):
        documents = read_documents_by_dataset(selected_dataset)
        df = pd.DataFrame([json.loads(doc) for doc in documents])
        
        st.subheader(f"Data from Dataset: {selected_dataset}")
        
        # Convert created_at to datetime and sort
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.sort_values('created_at', ascending=False)
        
        # Display the dataframe
        st.dataframe(df.head(100))

        # Display statistics and plots
        plot_incident_stats(df)