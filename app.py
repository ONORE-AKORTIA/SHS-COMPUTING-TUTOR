import os
import pandas as pd
import streamlit as st
from groq import Groq

# Page configuration
st.set_page_config(
    page_title="SHS Computing AI Tutor", page_icon="💻", layout="centered"
)


# Initialize Groq client securely using Streamlit Secrets
def get_groq_client():
  api_key = st.secrets.get("GROQ_API_KEY")
  if not api_key:
    st.error("Groq API key not found in Streamlit Secrets!")
    return None
  return Groq(api_key=api_key)


# Available subjects and their consolidated datasets
def get_available_subjects():
  return {
      "Computing": ["waec_qa_dataset.csv"],
      "ICT": [
          "ICT_LM_FINAL_SECTION_1_LV_qa_dataset.csv",
          "ICT_LM_FINAL_SECTION_2-LV_qa_dataset.csv",
          "ICT_LM_FINAL_SECTION_3-LV_qa_dataset.csv",
          "ICT_LM_FINAL_SECTION_4-LV_qa_dataset.csv",
          "ICT_LM_FINAL_SECTION_5-LV_qa_dataset.csv",
          "LM ICT Sections 1-5_qa_dataset.csv",
      ],
      "Robotics": ["robotics_qa_dataset.csv"],
  }


@st.cache_data
def load_dataset(filenames):
  """Loads and combines dataset CSV files from the textbooks directory."""
  all_dfs = []
  for filename in filenames:
    file_path = os.path.join("textbooks", filename)
    if os.path.exists(file_path):
      try:
        df = pd.read_csv(file_path)
        all_dfs.append(df)
      except Exception as e:
        st.warning(f"Error loading {filename}: {e}")
  if all_dfs:
    return pd.concat(all_dfs, ignore_index=True)
  return pd.DataFrame(columns=["question_text", "answer_text"])


# Sidebar setup
st.sidebar.title("Navigation")
subjects = get_available_subjects()
selected_subject = st.sidebar.selectbox(
    "Select Subject", list(subjects.keys())
)

# Load data for selected subject
dataset_files = subjects[selected_subject]
df_dataset = load_dataset(dataset_files)

# Main UI Header
st.title("SHS Computing AI Tutor")
st.write(
    f"Your intelligent companion for **{selected_subject}**. Ask questions"
    " below!"
)

# Initialize chat history in session state
if "messages" not in st.session_state:
  st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# Handle user input
user_query = st.chat_input(f"Ask a question about {selected_subject}...")

if user_query:
  # Append user message
  st.session_state.messages.append({"role": "user", "content": user_query})
  with st.chat_message("user"):
    st.markdown(user_query)

  # Retrieve context from the dataset using simple keyword matching
  retrieved_context = (
      "No specific textbook context found. Answer based on general knowledge."
  )
  if not df_dataset.empty:
    # Search for rows containing keywords from the query
    matches = df_dataset[
        df_dataset["answer_text"].str.contains(
            user_query, case=False, na=False, regex=False
        )
    ]
    if not matches.empty:
      # OPTION 2 SAFEGUARD: Pull only the top 1 match and truncate text to prevent 413 token errors
      best_match = matches.iloc[0]["answer_text"]
      max_chars = 1200  # Strict character limit per prompt injection
      retrieved_context = (
          best_match[:max_chars] + "..."
          if len(best_match) > max_chars
          else best_match
      )

  # Generate AI response using Groq
  client = get_groq_client()
  if client:
    with st.chat_message("assistant"):
      with st.spinner("Thinking..."):
        try:
          completion = client.chat.completions.create(
              model="llama-3.1-8b-instant",
              messages=[
                  {
                      "role": "system",
                      "content": (
                          "You are an expert SHS AI Tutor. Use the provided"
                          " textbook context to explain concepts clearly and"
                          " concisely to students."
                      ),
                  },
                  {
                      "role": "user",
                      "content": (
                          f"Textbook Context:\n{retrieved_context}\n\nStudent"
                          f" Question: {user_query}"
                      ),
                  },
              ],
              max_tokens=500,  # Limits response size to conserve token bucket limits
              temperature=0.3,
          )
          ai_response = completion.choices[0].message.content
          st.markdown(ai_response)
          st.session_state.messages.append(
              {"role": "assistant", "content": ai_response}
          )
        except Exception as e:
          st.error(f"Error connecting to AI service: {e}")
