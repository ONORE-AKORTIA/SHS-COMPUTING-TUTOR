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


# Function to transcribe audio using Groq's Whisper model
def transcribe_audio(audio_file):
  try:
    with open("temp_audio.wav", "wb") as f:
      f.write(audio_file.getbuffer())

    with open("temp_audio.wav", "rb") as file:
      transcription = client.audio.transcriptions.create(
          file=("temp_audio.wav", file.read()),
          model="whisper-large-v3-turbo",
          prompt=(
              "Ghanaian Senior High School computing context, WAEC terms, IT,"
              " programming, algorithms."
          ),
          language="en",
      )
    if os.path.exists("temp_audio.wav"):
      os.remove("temp_audio.wav")

    return transcription.text
  except Exception as e:
    return None


# Function to query the Groq LLM
def ask_ai_tutor(query, dataset, student_name, student_school, previous_themes=None):
  context_text = "\n".join(
      [
          f"Q: {row['question_text']}\nA: {row['answer_text']}"
          for _, row in dataset.head(15).iterrows()
      ]
  )

  theme_instruction = ""
  if previous_themes:
    theme_instruction = f"""
    The user is selecting or referring to one of these previously suggested themes: {previous_themes}.
    If the user's input is a number (like 1, 2, 3) or matches one of these themes, provide a detailed explanation of that specific theme aligned with the WAEC/NaCCA curriculum.
    """

  prompt = f"""
    You are Sir O.K., an expert, friendly AI tutor for Ghanaian Senior High School (SHS) Computing students (created by Mr. Onore Akortia from OLA SHS, Ho).
    You are currently speaking with a student named {student_name} from {student_school}.

    CRITICAL INSTRUCTIONS:
    - Do NOT re-introduce yourself, mention who created you, or state your aims in your response. Jump straight into answering the student's question directly.
    - Keep your tone concise, encouraging, clear, and strictly educational aligned with WAEC/NaCCA standards.
    - Use the provided context to answer accurately. If it's not in the context, use standard curriculum knowledge.
    - After your answer, provide a numbered list of exactly 3 suggested follow-up sub-themes related to the topic for further exploration.
    
    {theme_instruction}

    Context:
    {context_text}

    Student's Input: {query}
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

if "last_themes" not in st.session_state:
  st.session_state.last_themes = None

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

        initial_welcome = (
            f"Hello {st.session_state.user_name} from"
            f" {st.session_state.user_school}! I am Sir O.K., your SHS"
            " Computing tutor. What computing topic would you like to explore"
            " today?"
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
    st.session_state.last_themes = None
    st.rerun()

  # Display chat history cleanly
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  # Input selection tabs for Voice vs Text
  input_method = st.radio(
      "Choose input method:", ["⌨️ Type Question", "🎤 Speak Question"], horizontal=True
  )

  user_query = None

  if input_method == "⌨️ Type Question":
    user_query = st.chat_input("Ask a computing question or type a theme number (e.g. 1, 2)...")
  else:
    st.markdown("### Record your question:")
    audio_value = st.audio_input("Click to record your voice question")
    if audio_value:
      with st.spinner("Transcribing your voice..."):
        transcribed_text = transcribe_audio(audio_value)
        if transcribed_text:
          st.success(f"Transcribed: \"{transcribed_text}\"")
          user_query = transcribed_text
        else:
          st.error(
              "Could not transcribe audio. Please try speaking again or type"
              " your question."
          )

  # Process the query if received
  if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
      st.markdown(user_query)

    with st.chat_message("assistant"):
      with st.spinner("Thinking..."):
        answer = ask_ai_tutor(
            user_query,
            df,
            st.session_state.user_name,
            st.session_state.user_school,
            previous_themes=st.session_state.last_themes,
        )
        st.markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        
        # Simple extraction or holding of the context for the next turn
        st.session_state.last_themes = answer
        st.rerun()
