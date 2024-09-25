import os
import random
import pandas as pd
import streamlit as st
from faker import Faker
from zipfile import ZipFile
from io import BytesIO
from datetime import datetime
import plotly.express as px

# Initialize Faker
fake = Faker()

# Incident types and causes
incident_types = [
    'slip', 'fire', 'safety_violation', 'chemical_spill', 'injury', 'near_miss',
    'electrical', 'ventilation', 'falling_object', 'heat_exhaustion'
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

    # Save incident frequency data to CSV in the data folder
    os.makedirs('data', exist_ok=True)
    incident_freq_df.to_csv('data/incident_type_frequency.csv', index=False)
    st.success("Incident frequency data saved to 'data/incident_type_frequency.csv'")

# Function to generate a synthetic document
def generate_synthetic_document(incident_id, plant):
    date = fake.date_this_year()
    description_template = random.choice(incident_descriptions)
    description = description_template.format(plant=plant, date=date)
    incident_report = {
        'Incident ID': incident_id,
        'Plant': plant,
        'Date': date,
        'Description': description
    }
    return incident_report

# Generate documents and return DataFrame
def generate_documents(num_documents):
    data = []
    for i in range(1, num_documents + 1):
        incident_type = random.choice(incident_types)
        cause = random.choice(incident_causes)
        plant = random.choice(plants)
        document = generate_synthetic_document(i, plant)
        document['Incident Type'] = incident_type
        document['Cause'] = cause
        data.append(document)
    return pd.DataFrame(data)

# Incident count by cause summary
def incidents_by_cause(df):
    return df.groupby(['Incident Type', 'Cause']).size().unstack(fill_value=0)

# Incident cause by location summary
def incident_cause_by_location(df):
    return df.groupby(['Incident Type', 'Plant']).size().unstack(fill_value=0)

# New function to calculate incident frequency by type
def incident_frequency_by_type(df):
    return df['Incident Type'].value_counts().reset_index()

# Create a zip file from the dataframe and return as bytes
def create_zip_from_df(df):
    # Create a buffer for the zip file
    zip_buffer = BytesIO()
    
    with ZipFile(zip_buffer, "w") as zf:
        # Add each document to the zip file as a separate text file
        for index, row in df.iterrows():
            # Generate content for each incident report
            incident_content = f"Incident ID: {row['Incident ID']}\nPlant: {row['Plant']}\nDate: {row['Date']}\nDescription: {row['Description']}\nIncident Type: {row['Incident Type']}\nCause: {row['Cause']}\n"
            filename = f"incident_{row['Incident ID']}.txt"
            zf.writestr(filename, incident_content)
            
            # Save the file locally in the data directory
            os.makedirs('data', exist_ok=True)
            with open(os.path.join('data', filename), 'w') as f:
                f.write(incident_content)
    
    # Reset the buffer's position to the beginning
    zip_buffer.seek(0)
    return zip_buffer

# Streamlit UI starts here
st.title("Generate Synthetic Incident Data")

# Multiselect for incident types, causes, and plants
selected_incident_types = st.multiselect("Select Incident Types", incident_types, default=incident_types)
selected_causes = st.multiselect("Select Causes", incident_causes, default=incident_causes)
selected_plants = st.multiselect("Select Plants", plants, default=plants)

# Select number of documents
num_documents = st.selectbox(
    "Select number of documents to generate:",
    [100, 1000, 10000, 100000, 1000000]
)

# Debug statement to check if Streamlit is working
st.write("Ready to generate data...")

# Generate sample data
if st.button("Generate"):
    st.write(f"Generating {num_documents} documents...")
    
    # Generate documents based on selections
    df = generate_documents(num_documents)

    # Filter by selected types, causes, and plants
    df_filtered = df[
        (df['Incident Type'].isin(selected_incident_types)) &
        (df['Cause'].isin(selected_causes)) &
        (df['Plant'].isin(selected_plants))
    ]

    # Show a sample of up to 100 records
    st.subheader("Sample Data (Up to 100 Records)")
    st.dataframe(df_filtered.head(100))

    plot_incident_stats(df_filtered)
    
    # Generate a summary CSV
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("Download Data as CSV", csv, "incident_data.csv", "text/csv")
    
    # Create and offer ZIP download
    zip_buffer = create_zip_from_df(df_filtered)
    
    # Generate the ZIP file
    st.download_button(
        label="Download Data as ZIP",
        data=zip_buffer,
        file_name="incident_data.zip",
        mime="application/zip"
    )

    # Number of incidents over date/time
    st.subheader("Number of Incidents Over Time")
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])
    incidents_over_time = df_filtered.groupby(df_filtered['Date'].dt.to_period('M')).size()
    st.line_chart(incidents_over_time)

    # The file is now saved directly to the data folder
