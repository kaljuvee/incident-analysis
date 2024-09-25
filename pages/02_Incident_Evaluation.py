import streamlit as st
import pandas as pd
from typing import List, Dict
import pickle

def load_incident_frequency():
    df = pd.read_csv('data/incident_type_frequency.csv')
    return dict(zip(df['Incident Type'], df['Frequency']))

def compare_frequencies(discovered_frequencies: Dict[str, int]):
    csv_frequencies = load_incident_frequency()

    st.subheader("Frequency Comparison")

    comparison_df = pd.DataFrame({
        'CSV Frequency': pd.Series(csv_frequencies),
        'Discovered Frequency': pd.Series(discovered_frequencies)
    }).fillna(0)

    comparison_df['Difference'] = comparison_df['Discovered Frequency'] - comparison_df['CSV Frequency']

    st.dataframe(comparison_df)

    st.bar_chart(comparison_df[['CSV Frequency', 'Discovered Frequency']])

    st.subheader("Analysis of Differences")
    new_types = set(discovered_frequencies.keys()) - set(csv_frequencies.keys())
    missing_types = set(csv_frequencies.keys()) - set(discovered_frequencies.keys())
    
    if new_types:
        st.write("Newly discovered incident types:", ", ".join(new_types))
    if missing_types:
        st.write("Incident types not found in the current analysis:", ", ".join(missing_types))
    
    significant_diff = comparison_df[abs(comparison_df['Difference']) > 5].index.tolist()
    if significant_diff:
        st.write("Incident types with significant frequency differences:")
        for incident_type in significant_diff:
            st.write(f"- {incident_type}: CSV: {csv_frequencies.get(incident_type, 0)}, "
                     f"Discovered: {discovered_frequencies.get(incident_type, 0)}")

def main():
    st.title("Incident Frequency Evaluation")

    if 'counts' not in st.session_state:
        st.warning("Please run the incident count analysis first.")
        return

    if st.button("Compare Frequencies"):
        with st.spinner("Comparing frequencies..."):
            compare_frequencies(st.session_state.counts)
        st.success("Frequency comparison completed!")

if __name__ == "__main__":
    main()