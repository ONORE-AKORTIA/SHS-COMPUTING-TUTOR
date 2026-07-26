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
              " Computing, ICT, Robotics, multiple choice letters A B C D."
          ),
          language="en",
      )
    if os.path.exists("temp_audio.wav"):
      os.remove("temp_audio.wav")

    return transcription.text
  except Exception as e:
    return None


# Sidebar setup for Subject Selection, Student Details, and Mode
st.sidebar.title("Navigation & Profile")
subjects = get_available_subjects()

if "prev_subject" not in st.session_state:
  st.session_state.prev_subject = "Computing"

selected_subject = st.sidebar.selectbox(
    "Select Subject", list(subjects.keys())
)

# Handle subject switch notification
if selected_subject != st.session_state.prev_subject:
  switch_msg = (
      f"Subject area has been switched from"
      f" **{st.session_state.prev_subject}** to **{selected_subject}**."
  )
  if "messages" in st.session_state:
    st.session_state.messages.append({"role": "assistant", "content": switch_msg})
  st.session_state.prev_subject = selected_subject

st.sidebar.markdown("---")
st.sidebar.subheader("Student Details")
# Encouraging full name including first, middle (if any), and last name in a single field
student_full_name = st.sidebar.text_input(
    "Full Name (First, Middle, Last)", value=""
)
student_school = st.sidebar.text_input("School Name", value="")

st.sidebar.markdown("---")
learning_mode = st.sidebar.radio(
    "Select Mode", ["💬 Study & Chat", "📝 WAEC Exam Practice"]
)

st.sidebar.markdown("---")
input_method = st.sidebar.radio(
    "Choose input method:", ["⌨️ Type Question", "🎤 Speak Question"]
)

# Load data for selected subject
dataset_files = subjects[selected_subject]
df_dataset = load_dataset(dataset_files)

# Layout: Laptop icon -> Title -> Your picture (ONORE_AKORTIA_1.jpg)
user_img_base64 = get_image_base64("ONORE_AKORTIA_1.jpg")

st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
        <span style="font-size: 2.5em;">💻</span>
        <div style="flex-grow: 1;">
            <h1 style="margin: 0; font-size: 1.8em; line-height: 1.2;">SHS Computing AI Tutor</h1>
            <p style="margin: 0; color: #666; font-size: 0.95em;">Your intelligent companion for <b>{selected_subject}</b>.</p>
        </div>
        {f"<img src='data:image/jpeg;base64,{user_img_base64}' width='75' style='border-radius: 8px; object-fit: cover;'>" if user_img_base64 else ""}
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize global session stores for progress tracking and multi-session persistence simulation
if "user_sessions" not in st.session_state:
  st.session_state.user_sessions = {}
if "messages" not in st.session_state:
  st.session_state.messages = []
if "greeted" not in st.session_state:
  st.session_state.greeted = False
if "exam_question_count" not in st.session_state:
  st.session_state.exam_question_count = 0
if "max_exam_questions" not in st.session_state:
  st.session_state.max_exam_questions = 5  # Standard session length

# Check for returning vs first-time user profile based on full name and school
user_key = f"{student_full_name.strip().lower()}_{student_school.strip().lower()}"

if student_full_name and student_school:
  if user_key not in st.session_state.user_sessions and st.session_state.greeted:
    # Initialize profile storage
    st.session_state.user_sessions[user_key] = {
        "messages": st.session_state.messages,
        "is_returning": True,
    }
  elif (
      user_key in st.session_state.user_sessions
      and not st.session_state.greeted
  ):
    # Load previous history for continuing user
    st.session_state.messages = st.session_state.user_sessions[user_key][
        "messages"
    ]

# Trigger initial personalized greeting only when ALL THREE fields (Subject, Name, School) are provided
if not st.session_state.greeted and student_full_name and student_school:
  user_status_label = (
      "Welcome back"
      if user_key in st.session_state.user_sessions
      else "Welcome (First-time user)"
  )
  initial_greeting = (
      f"Hello {student_full_name} from {student_school}! ({user_status_label})."
      f" I am your AI tutor ready to help you master {selected_subject} for"
      " WAEC. How can I assist you today?"
  )
  st.session_state.messages.append(
      {"role": "assistant", "content": initial_greeting}
  )
  st.session_state.greeted = True
  if user_key not in st.session_state.user_sessions:
    st.session_state.user_sessions[user_key] = {
        "messages": st.session_state.messages,
        "is_returning": False,
    }

# Display progress bar if in WAEC Exam Practice mode
if learning_mode == "📝 WAEC Exam Practice" and student_full_name:
  answered = st.session_state.exam_question_count
  total = st.session_state.max_exam_questions
  progress_val = min(float(answered) / float(total), 1.0)
  st.progress(
      progress_val,
      text=(
          f"Exam Session Progress: Question {answered} of {total} answered"
      ),
  )

# Display chat history
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# Handle user input based on sidebar choice (Text or Voice)
user_query = None

if input_method == "⌨️ Type Question":
  prompt_label = (
      f"Answer/Ask a question about {selected_subject}..."
      if learning_mode == "💬 Study & Chat"
      else (
          "Type your answer (e.g. option letter A/B/C/D for objective, or"
          " detailed text for essay)..."
      )
  )
  user_query = st.chat_input(prompt_label)
else:
  st.markdown("### Record your input:")
  audio_value = st.audio_input("Click to record your voice")
  if audio_value:
    with st.spinner("Transcribing your voice..."):
      transcribed_text = transcribe_audio(audio_value)
      if transcribed_text:
        st.success(f'Transcribed: "{transcribed_text}"')
        user_query = transcribed_text
      else:
        st.error("Could not transcribe audio. Please try again.")

if user_query:
  st.session_state.messages.append({"role": "user", "content": user_query})
  with st.chat_message("user"):
    st.markdown(user_query)

  if learning_mode == "📝 WAEC Exam Practice":
    st.session_state.exam_question_count += 1

  # Retrieve context from dataset
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

  client = get_groq_client()
  if client:
    with st.chat_message("assistant"):
      with st.spinner("Thinking..."):
        try:
          if learning_mode == "📝 WAEC Exam Practice":
            persona_prompt = (
                f"You are an expert WAEC Examiner in {selected_subject} testing"
                f" {student_full_name} from {student_school} based on textbook"
                " materials. \n\nRules for Examiner Mode:\n1. If the question"
                " has multiple-choice options (A, B, C, D), accept the single"
                " letter response as complete representation.\n2. If the"
                " question is an essay-type instruction (containing Differentiate,"
                " Describe, Explain, Analyse, Examine, etc.), expect a complete"
                " descriptive text response.\n3. Evaluate the student's"
                " response, assign a percentage score (0-100), and provide a"
                " strict remark based on this scale:\n   - 90 and above:"
                " Distinction\n   - 80 and above: Excellent\n   - 70 and above:"
                " Very Good\n   - 60 and above: Good\n   - 50 and above: Pass\n  "
                " - 40 and above: Nice work, keep trying\n   - 30 and above: Can"
                " do better\n   - Below 30: You have to sit up\n4. If the"
                " student's answer is wrong, you MUST explicitly display the"
                " correct answer to aid learning.\n5. If the user is requesting"
                " a new question, generate a fresh WAEC-style question (either"
                " objective with A, B, C, D options or essay-type)."
            )
          else:
            persona_prompt = (
                f"You are an expert SHS AI Tutor helping {student_full_name}"
                f" from {student_school} in {selected_subject}. Provide"
                " precise, concise, and direct answers based on the provided"
                " textbook context."
            )

          completion = client.chat.completions.create(
              model="llama-3.1-8b-instant",
              messages=[
                  {"role": "system", "content": persona_prompt},
                  {
                      "role": "user",
                      "content": (
                          f"Textbook Context:\n{retrieved_context}\n\nStudent"
                          f" Input: {user_query}"
                      ),
                  },
              ],
              max_tokens=450,
              temperature=0.3,
          )
          ai_response = completion.choices[0].message.content
          st.markdown(ai_response)
          st.session_state.messages.append(
              {"role": "assistant", "content": ai_response}
          )

          # Update persistent user session state history
          if user_key in st.session_state.user_sessions:
            st.session_state.user_sessions[user_key]["messages"] = (
                st.session_state.messages
            )

        except Exception as e:
          st.error(f"Error connecting to AI service: {e}")
