import random
import pandas as pd
import streamlit as st
from faker import Faker
import plotly.express as px
import json
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
    'Heat exhaustion'
]
incident_causes = ['Human error', 'Equipment', 'Environment', 'Unknown']
plants = ['Plant A', 'Plant B', 'Plant C', 'Plant D']

# Sample descriptions for variation (without incident type and cause tags)
incident_descriptions = [
    "Employee reported dizziness and nausea in {plant}'s packaging area on {date}.",
    "A fire alarm was triggered in {plant}'s assembly line on {date}.",
    "On {date}, a worker in {plant}'s production area tripped over an unmarked hazard.",
    "A near-miss incident was reported on {date} at {plant}.",
    "Symptoms of heat exhaustion were reported by a worker in {plant} on {date}."
]

# Function to plot incident statistics using Plotly
def plot_incident_stats(df):
    st.subheader("Incident Cause by Location")

    # Incident Cause by Location (grouped by plant)
    incident_cause_by_location_df = incident_cause_by_location(df)
    st.write(incident_cause_by_location_df)

    # Plot for Incident Cause by Location
    incident_cause_by_location_melted = incident_cause_by_location_df.reset_index().melt(id_vars="Incident Type", var_name="Plant", value_name="Count")
    fig_location = px.bar(incident_cause_by_location_melted, x="Incident Type", y="Count", color="Plant", barmode="group", title="Incident Cause by Location")
    st.plotly_chart(fig_location)

    st.subheader("Incident Count by Cause")

    # Incident Count by Cause (grouped by cause)
    incidents_by_cause_df = incidents_by_cause(df)
    st.write(incidents_by_cause_df)

    # Plot for Incident Count by Cause
    incidents_by_cause_melted = incidents_by_cause_df.reset_index().melt(id_vars="Incident Type", var_name="Cause", value_name="Count")
    fig_cause = px.bar(incidents_by_cause_melted, x="Incident Type", y="Count", color="Cause", barmode="stack", title="Incident Count by Cause")
    st.plotly_chart(fig_cause)

    st.subheader("Incident Frequency by Type")
    
    # Calculate and display incident frequency by type
    incident_freq_df = incident_frequency_by_type(df)
    incident_freq_df.columns = ['Incident Type', 'Frequency']
    st.write(incident_freq_df)

    # Plot for Incident Frequency by Type
    fig_freq = px.bar(incident_freq_df, x="Incident Type", y="Frequency", title="Incident Frequency by Type")
    st.plotly_chart(fig_freq)

# Function to generate a synthetic document
def generate_synthetic_document(incident_id, plant):
    date = fake.date_this_year()
    description_template = random.choice(incident_descriptions)
    description = description_template.format(plant=plant, date=date)
    incident_report = {
        'Incident ID': incident_id,
        'Plant': plant,
        'Date': str(date),  # Convert to string for JSON serialization
        'Description': description
    }
    return incident_report

# Generate documents and return DataFrame
def generate_documents(num_documents):
    data = []
    documents_to_store = []
    for i in range(1, num_documents + 1):
        incident_type = random.choice(incident_types)
        cause = random.choice(incident_causes)
        plant = random.choice(plants)
        document = generate_synthetic_document(i, plant)
        document['Incident Type'] = incident_type
        document['Cause'] = cause
        data.append(document)
        documents_to_store.append(json.dumps(document))  # Convert to JSON string for storage
    
    # Store documents in the database and get the data_set_id
    data_set_id = store_documents(documents_to_store)
    
    return pd.DataFrame(data), data_set_id

# Incident count by cause summary
def incidents_by_cause(df):
    return df.groupby(['Incident Type', 'Cause']).size().unstack(fill_value=0)

# Incident cause by location summary
def incident_cause_by_location(df):
    return df.groupby(['Incident Type', 'Plant']).size().unstack(fill_value=0)

# New function to calculate incident frequency by type
def incident_frequency_by_type(df):
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
    # Create a dictionary for the selectbox options
    dataset_options = {f"{data_set_id} ({count} documents)": data_set_id for data_set_id, count in datasets_with_counts}
    
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
        st.dataframe(df.head(100))

        # Display statistics and plots
        plot_incident_stats(df)

        # Number of incidents over date/time
        st.subheader("Number of Incidents Over Time")
        df['Date'] = pd.to_datetime(df['Date'])
        incidents_over_time = df.groupby(df['Date'].dt.to_period('M')).size()
        st.line_chart(incidents_over_time)
