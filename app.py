import base64
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


# Helper function to convert local image to base64
def get_image_base64(path):
  if os.path.exists(path):
    with open(path, "rb") as f:
      data = f.read()
    return base64.b64encode(data).decode()
  return None


# Function to transcribe audio using Groq's Whisper model
def transcribe_audio(audio_file):
  client = get_groq_client()
  if not client:
    return None
  try:
    with open("temp_audio.wav", "wb") as f:
      f.write(audio_file.getbuffer())

    with open("temp_audio.wav", "rb") as file:
      transcription = client.audio.transcriptions.create(
          file=("temp_audio.wav", file.read()),
          model="whisper-large-v3-turbo",
          prompt=(
              "Ghanaian Senior High School educational context, WAEC terms,"
              " Computing, ICT, Robotics."
          ),
          language="en",
      )
    if os.path.exists("temp_audio.wav"):
      os.remove("temp_audio.wav")

    return transcription.text
  except Exception as e:
    return None


# Sidebar setup for Subject Selection, Student Details, and Input Method
st.sidebar.title("Navigation & Profile")
subjects = get_available_subjects()
selected_subject = st.sidebar.selectbox(
    "Select Subject", list(subjects.keys())
)

st.sidebar.markdown("---")
st.sidebar.subheader("Student Details")
student_name = st.sidebar.text_input("Full Name", value="")
student_school = st.sidebar.text_input("School Name", value="")

st.sidebar.markdown("---")
input_method = st.sidebar.radio(
    "Choose input method:", ["⌨️ Type Question", "🎤 Speak Question"]
)

# Load data for selected subject
dataset_files = subjects[selected_subject]
df_dataset = load_dataset(dataset_files)

# Single inline header layout with computer icon and user picture side-by-side on the left of the title
laptop_img_base64 = get_image_base64("laptop.png")
user_img_base64 = get_image_base64("ONORE_AKORTIA_1.jpg")

st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
        <span style="font-size: 2.5em;">💻</span>
        {f"<img src='data:image/jpeg;base64,{user_img_base64}' width='60' style='border-radius: 8px; object-fit: cover;'>" if user_img_base64 else ""}
        <div style="flex-grow: 1;">
            <h1 style="margin: 0; font-size: 1.8em; line-height: 1.2;">SHS Computing AI Tutor</h1>
            <p style="margin: 0; color: #666; font-size: 0.95em;">Your intelligent companion for <b>{selected_subject}</b>.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize chat history and greeting state
if "messages" not in st.session_state:
  st.session_state.messages = []
if "greeted" not in st.session_state:
  st.session_state.greeted = False

# Trigger initial personalized greeting once
if not st.session_state.greeted and student_name:
  initial_greeting = (
      f"Hello {student_name} from {student_school}! I am your AI tutor ready"
      f" to help you master {selected_subject}. How can I assist you today?"
  )
  st.session_state.messages.append(
      {"role": "assistant", "content": initial_greeting}
  )
  st.session_state.greeted = True

# Display chat history
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# Handle user input based on sidebar choice (Text or Voice)
user_query = None

if input_method == "⌨️ Type Question":
  user_query = st.chat_input(f"Ask a question about {selected_subject}...")
else:
  st.markdown("### Record your question:")
  audio_value = st.audio_input("Click to record your voice question")
  if audio_value:
    with st.spinner("Transcribing your voice..."):
      transcribed_text = transcribe_audio(audio_value)
      if transcribed_text:
        st.success(f'Transcribed: "{transcribed_text}"')
        user_query = transcribed_text
      else:
        st.error(
            "Could not transcribe audio. Please try speaking again or type"
            " your question."
        )

if user_query:
  # Append user message
  st.session_state.messages.append({"role": "user", "content": user_query})
  with st.chat_message("user"):
    st.markdown(user_query)

  # Retrieve context from the dataset using standard keyword matching
  retrieved_context = (
      "No specific textbook context found. Answer based on general knowledge."
  )
  if not df_dataset.empty:
    matches = df_dataset[
        df_dataset["answer_text"].str.contains(
            user_query, case=False, na=False, regex=False
        )
    ]
    if not matches.empty:
      retrieved_context = matches.iloc[0]["answer_text"]

  # Generate precise AI response using Groq with controlled response token size
  client = get_groq_client()
  if client:
    with st.chat_message("assistant"):
      with st.spinner("Thinking..."):
        try:
          persona_prompt = (
              f"You are an expert SHS AI Tutor helping {student_name} from"
              f" {student_school}. Provide precise, concise, and direct answers"
              " based on the provided textbook context."
          )
          completion = client.chat.completions.create(
              model="llama-3.1-8b-instant",
              messages=[
                  {"role": "system", "content": persona_prompt},
                  {
                      "role": "user",
                      "content": (
                          f"Textbook Context:\n{retrieved_context}\n\nStudent"
                          f" Question: {user_query}"
                      ),
                  },
              ],
              max_tokens=300,  # Reduced token limit for precise responses
              temperature=0.3,
          )
          ai_response = completion.choices[0].message.content
          st.markdown(ai_response)
          st.session_state.messages.append(
              {"role": "assistant", "content": ai_response}
          )
        except Exception as e:
          st.error(f"Error connecting to AI service: {e}")
