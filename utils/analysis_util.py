import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
import re
from difflib import SequenceMatcher

def normalize_string(s):
    return s.lower().strip().strip("'\"")

def extract_date(text):
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', text)
    if date_match:
        return datetime.strptime(date_match.group(), '%Y-%m-%d')
    
    date_patterns = [
        (r'\d{2}/\d{2}/\d{4}', '%m/%d/%Y'),
        (r'\d{2}-\d{2}-\d{4}', '%m-%d-%Y'),
        (r'\w+ \d{1,2}, \d{4}', '%B %d, %Y')
    ]
    
    for pattern, date_format in date_patterns:
        date_match = re.search(pattern, text)
        if date_match:
            try:
                return datetime.strptime(date_match.group(), date_format)
            except ValueError:
                continue
    
    return None

def extract_plant(text):
    plant_match = re.search(r'Plant [A-Z]', text)
    return plant_match.group() if plant_match else 'Unknown'

def categorize_incident(text, incident_types):
    normalized_text = normalize_string(text)
    
    # Check for exact matches first
    for incident_type in incident_types:
        if normalize_string(incident_type) in normalized_text:
            return incident_type
    
    # If no exact match, look for partial matches
    for incident_type in incident_types:
        words = normalize_string(incident_type).split()
        if all(word in normalized_text for word in words):
            return incident_type
    
    # If still no match, look for any word match
    for incident_type in incident_types:
        words = normalize_string(incident_type).split()
        if any(word in normalized_text for word in words):
            return incident_type
    
    return 'Unknown'

def get_incidents_by_cause(df):
    if 'Incident Type' not in df.columns or 'Cause' not in df.columns:
        st.warning("Required columns 'Incident Type' or 'Cause' not found in the data.")
        return pd.DataFrame()
    return df.groupby(['Incident Type', 'Cause']).size().unstack(fill_value=0)

def incident_cause_by_location(df):
    if 'Incident Type' not in df.columns or 'Plant' not in df.columns:
        st.warning("Required columns 'Incident Type' or 'Plant' not found in the data.")
        return pd.DataFrame()
    return df.groupby(['Incident Type', 'Plant']).size().unstack(fill_value=0)

def incident_frequency_by_type(df):
    if 'Incident Type' not in df.columns:
        st.warning("Required column 'Incident Type' not found in the data.")
        return pd.DataFrame()
    return df['Incident Type'].value_counts().reset_index()

def plot_incidents_over_time(df):
    if 'Date' not in df.columns:
        st.warning("Required column 'Date' not found in the data.")
        return None
    incidents_over_time = df.groupby(df['Date'].dt.to_period('M')).size().reset_index()
    incidents_over_time['Date'] = incidents_over_time['Date'].dt.to_timestamp()
    
    fig = px.line(incidents_over_time, x='Date', y=0, 
                  labels={'Date': 'Date', '0': 'Number of Incidents'},
                  title='Number of Incidents Over Time')
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Number of Incidents',
        xaxis=dict(tickmode='auto', nticks=20),
        yaxis=dict(tickmode='auto', nticks=10)
    )
    
    return fig

def plot_incident_stats(df, incident_counts, plants, causes):
    st.subheader("Incident Type Counts")
    fig_counts = px.bar(x=list(incident_counts.keys()), y=list(incident_counts.values()), 
                        labels={'x': 'Incident Type', 'y': 'Count'},
                        title="Incident Type Counts")
    st.plotly_chart(fig_counts)
    
    incident_cause_by_location_df = incident_cause_by_location(df)
    if not incident_cause_by_location_df.empty:
        st.subheader("Incident Cause by Location")
        st.write(incident_cause_by_location_df)

        incident_cause_by_location_melted = incident_cause_by_location_df.reset_index().melt(id_vars="Incident Type", var_name="Plant", value_name="Count")
        fig_location = px.bar(incident_cause_by_location_melted, x="Incident Type", y="Count", color="Plant", barmode="group", title="Incident Cause by Location")
        st.plotly_chart(fig_location)

    incidents_by_cause_df = get_incidents_by_cause(df)
    if not incidents_by_cause_df.empty:
        st.subheader("Incident Count by Cause")
        st.write(incidents_by_cause_df)

        incidents_by_cause_melted = incidents_by_cause_df.reset_index().melt(id_vars="Incident Type", var_name="Cause", value_name="Count")
        fig_cause = px.bar(incidents_by_cause_melted, x="Incident Type", y="Count", color="Cause", barmode="stack", title="Incident Count by Cause")
        st.plotly_chart(fig_cause)

    incident_freq_df = incident_frequency_by_type(df)
    if not incident_freq_df.empty:
        st.subheader("Incident Frequency by Type")
        incident_freq_df.columns = ['Incident Type', 'Frequency']
        st.write(incident_freq_df)

        fig_freq = px.bar(incident_freq_df, x="Incident Type", y="Frequency", title="Incident Frequency by Type")
        st.plotly_chart(fig_freq)

    if 'Date' in df.columns:
        st.subheader("Number of Incidents Over Time")
        fig_time = plot_incidents_over_time(df)
        if fig_time:
            st.plotly_chart(fig_time, use_container_width=True)

    if 'Plant' in df.columns:
        st.subheader("Incidents by Plant")
        incidents_by_plant = df['Plant'].value_counts()
        fig_plant = px.bar(x=incidents_by_plant.index, y=incidents_by_plant.values, labels={'x': 'Plant', 'y': 'Number of Incidents'}, title="Incidents by Plant")
        st.plotly_chart(fig_plant)

    if 'Cause' in df.columns:
        st.subheader("Incidents by Cause")
        incidents_by_cause = df['Cause'].value_counts()
        fig_cause = px.bar(x=incidents_by_cause.index, y=incidents_by_cause.values, labels={'x': 'Cause', 'y': 'Number of Incidents'}, title="Incidents by Cause")
        st.plotly_chart(fig_cause)
