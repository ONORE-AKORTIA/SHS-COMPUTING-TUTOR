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

# Single inline header layout (Displays Laptop, Title, and Picture perfectly together)
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
  # Ensure your waec_qa_dataset.csv file is in the same directory as app.py
  df = pd.read_csv("waec_qa_dataset.csv")
  return df


df = load_data()


# Function to query the Groq LLM
def ask_ai_tutor(query, dataset, student_name):
  # Build context from dataset rows
  context_text = "\n".join(
      [
          f"Q: {row['question_text']}\nA: {row['answer_text']}"
          for _, row in dataset.head(15).iterrows()
      ]
  )

  prompt = f"""
    You are an expert, friendly AI tutor for Ghanaian Senior High School (SHS) Computing students.
    You are currently speaking with a student named {student_name}.
    Use the following official curriculum context to answer the student's question accurately. 
    If the answer isn't directly in the context, use your knowledge aligned with WAEC standards.
    Keep your tone encouraging, clear, and educational.
    Your name is Sir O.K. You are created by Mr. ONORE AKORTIA, a teacher from OLA SHS, HO.
    After each response, ask the learner if he or she would like to have further explanations based on suggested themes deduced from the response.

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

if "messages" not in st.session_state:
  st.session_state.messages = []

# Step 1: Capture Student Name if not already provided
if not st.session_state.user_name:
  st.subheader("Welcome! Let's get started.")
  with st.form("name_form"):
    name_input = st.text_input("Please enter your name:")
    submit_button = st.form_submit_button("Start Learning")

    if submit_button:
      if name_input.strip():
        st.session_state.user_name = name_input.strip()
        # Initial greeting message stored once in session history
        initial_welcome = (
            f"Hello {st.session_state.user_name}! I am Sir O.K., your SHS"
            " Computing tutor created by Mr. Onore Akortia from OLA SHS, Ho."
            " I'm here to help you master your computing topics and ace your"
            " exams. What would you like to study today?"
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": initial_welcome}
        )
        st.rerun()
      else:
        st.warning("Please enter a valid name to continue.")
else:
  # Display active student info banner
  st.sidebar.markdown(f"**Logged in student:** {st.session_state.user_name}")
  if st.sidebar.button("Log out / Change Name"):
    st.session_state.user_name = None
    st.session_state.messages = []
    st.rerun()

  # Display chat history (prevents re-introductory loops)
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  # Handle user input from chat box
  if user_query := st.chat_input("Ask a computing question..."):
    # Append user query to history and display it
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
      st.markdown(user_query)

    # Generate and display assistant response
    with st.chat_message("assistant"):
      with st.spinner("Thinking..."):
        answer = ask_ai_tutor(user_query, df, st.session_state.user_name)
        st.markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
