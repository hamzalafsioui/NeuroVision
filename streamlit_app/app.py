import os
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Diagnostic multimodal")
st.write("Starter interface | model features will be added later.")

if st.button("Test connection to the API"):
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        st.success("API is running")
        st.json(response.json())
    except requests.RequestException as error:
        st.error(f"Could not reach the API: {error}")