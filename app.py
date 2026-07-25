import base64
import os
import pandas as pd
from groq import Groq
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="SHS Computing AI Tutor", page_icon="💻", layout="centered"
)


# Function to load local image for HTML display
def img_to_base64(image_path):
  if os.path.exists(image_path):
    with open(image_path, "rb") as img_file:
      return base64.b64encode(img_file.read()).decode()
  return ""


# Convert your picture to base64
img_base64 = img_to_base64("ONORE_AKORTIA_1.jpg")

# Single inline header layout
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
        <span style="font-size: 2.5em;">💻</span>
        <div style="flex-grow: 1;">
            <h1 style="margin: 0; font-size: 1.8em; line-height: 1.2;">SHS Computing AI Tutor</h1>
            <p style="margin: 0; color: #666; font-size: 0.95em;">Your personal WAEC & NaCCA curriculum study assistant.</p>
        </div>
        <img src="data:image/jpeg;base64,{img_base64}" width="75" style="border-radius: 8px; object-fit: cover;">
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize Groq client using Streamlit Secrets
try:
  api_key = st.secrets["GROQ_API_KEY"]
  client = Groq(api_key=api_key)
except Exception as e:
  st.error(
      "Groq API key not found. Please configure it in your Streamlit Secrets."
  )
  st.stop()


# Load dataset cached for performance
@st.cache_data
def load_data():
  df = pd.read_csv("waec_qa_dataset.csv")
  return df


df = load_data()


# Function to query the Groq LLM
def ask_ai_tutor(query, dataset, student_name, student_school):
  # Build context from dataset rows
  context_text = "\n".join(
      [
          f"Q: {row['question_text']}\nA: {row['answer_text']}"
          for _, row in dataset.head(15).iterrows()
      ]
  )

  # Streamlined prompt incorporating student name and school
  prompt = f"""
    You are Sir O.K., an expert, friendly AI tutor for Ghanaian Senior High School (SHS) Computing students (created by Mr. Onore Akortia from OLA SHS, Ho).
    You are currently speaking with a student named {student_name} from {student_school}.

    CRITICAL INSTRUCTIONS:
    - Do NOT re-introduce yourself, mention who created you, or state your aims in your response. Jump straight into answering the student's question directly.
    - Keep your tone concise, encouraging, clear, and strictly educational aligned with WAEC/NaCCA standards.
    - Use the provided context to answer accurately. If it's not in the context, use standard curriculum knowledge.
    - After answering, concisely ask the student if they would like further explanations based on specific sub-themes related to the topic.

    Context:
    {context_text}

    Student's Question: {query}
    """

  try:
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": prompt,
        }],
        model="llama-3.1-8b-instant",
    )
    return chat_completion.choices[0].message.content
  except Exception as e:
    return (
        f"Sorry {student_name}, I encountered an error connecting to the AI"
        f" service: {e}"
    )


# --- Session State Initialization ---
if "user_name" not in st.session_state:
  st.session_state.user_name = None

if "user_school" not in st.session_state:
  st.session_state.user_school = None

if "messages" not in st.session_state:
  st.session_state.messages = []

# Step 1: Capture Student Name and School if not already provided
if not st.session_state.user_name or not st.session_state.user_school:
  st.subheader("Welcome! Let's get started.")
  with st.form("student_info_form"):
    name_input = st.text_input("Please enter your full name:")
    school_input = st.text_input("Please enter your school name:")
    submit_button = st.form_submit_button("Start Learning")

    if submit_button:
      if name_input.strip() and school_input.strip():
        st.session_state.user_name = name_input.strip()
        st.session_state.user_school = school_input.strip()
        
        # Initial greeting happens ONCE
        initial_welcome = (
            f"Hello {st.session_state.user_name} from {st.session_state.user_school}! "
            "I am Sir O.K., your SHS Computing tutor. What computing topic would you like to explore today?"
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": initial_welcome}
        )
        st.rerun()
      else:
        st.warning("Please fill in both your name and school to continue.")
else:
  # Display active student info banner in the sidebar
  st.sidebar.markdown(f"**Student:** {st.session_state.user_name}")
  st.sidebar.markdown(f"**School:** {st.session_state.user_school}")
  if st.sidebar.button("Log out / Change Details"):
    st.session_state.user_name = None
    st.session_state.user_school = None
    st.session_state.messages = []
    st.rerun()

  # Display chat history cleanly
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  # Handle user input from chat box
  if user_query := st.chat_input("Ask a computing question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
      st.markdown(user_query)

    with st.chat_message("assistant"):
      with st.spinner("Thinking..."):
        answer = ask_ai_tutor(
            user_query, df, st.session_state.user_name, st.session_state.user_school
        )
        st.markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
