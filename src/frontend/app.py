
import os
from pathlib import Path

import requests
import streamlit as st


from dotenv import dotenv_values

# Load the project-root .env file used by the FastAPI backend.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8002"
)

env_values = dotenv_values(PROJECT_ROOT / ".env")
API_KEY = os.getenv("RAG_API_KEY") or dotenv_values(PROJECT_ROOT / ".env").get("RAG_API_KEY")
# Streamlit page
st.set_page_config(
    page_title="IT Incident Resolution Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("IT Incident Resolution Assistant")
st.write("Ask a question about an IT incident or troubleshooting issue.")

if not API_KEY:
    st.error("API key is missing. Check your .env file.")
    st.stop()

# Check FastAPI connection
try:
    health = requests.get(
        f"{API_URL}/health",
        timeout=5
    )
    health.raise_for_status()
    st.success("FastAPI backend is connected.")


except requests.exceptions.RequestException:
    st.error("Cannot connect to FastAPI. Please start the backend.")
    st.stop()

# User question
question = st.text_area(
    "Enter your IT support question:",
    placeholder="Example: How can I fix VPN connection issues?"
)

if st.button("Get Answer"):

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Searching documents and generating answer..."):

            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question.strip()},
                    headers={"X-API-Key": API_KEY},
                    timeout=180
                )

                response.raise_for_status()
                result = response.json()

                st.subheader("AI Answer")
                st.write(result.get("answer", "No answer returned."))

                st.subheader("Sources")
                sources = result.get("sources", [])

                if sources:
                    for source in sources:
                        st.write(f"- {source}")
                else:
                    st.info("No document sources returned.")

            except requests.exceptions.HTTPError as error:
                st.error(
                    f"API returned an error: "
                    f"{error.response.status_code}"
                )

            except requests.exceptions.Timeout:
                st.error("The request timed out.")

            except requests.exceptions.RequestException as error:
                st.error(f"Connection error: {error}")

            except ValueError:
                st.error("FastAPI returned an invalid JSON response.")
