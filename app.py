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

# Initialize global session stores for progress tracking and exam session metrics
if "user_sessions" not in st.session_state:
  st.session_state.user_sessions = {}
if "messages" not in st.session_state:
  st.session_state.messages = []
if "greeted" not in st.session_state:
  st.session_state.greeted = False

# Exam session state variables
if "exam_active" not in st.session_state:
  st.session_state.exam_active = False
if "total_questions" not in st.session_state:
  st.session_state.total_questions = 5
if "current_question_num" not in st.session_state:
  st.session_state.current_question_num = 0
if "correct_count" not in st.session_state:
  st.session_state.correct_count = 0
if "wrong_count" not in st.session_state:
  st.session_state.wrong_count = 0
if "question_type" not in st.session_state:
  st.session_state.question_type = "objective"  # objective or essay

user_key = f"{student_full_name.strip().lower()}_{student_school.strip().lower()}"

if student_full_name and student_school:
  if user_key not in st.session_state.user_sessions and st.session_state.greeted:
    st.session_state.user_sessions[user_key] = {
        "messages": st.session_state.messages,
        "is_returning": True,
    }
  elif (
      user_key in st.session_state.user_sessions
      and not st.session_state.greeted
  ):
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

# Display progress dashboard if in WAEC Exam Practice mode and exam is active
if learning_mode == "📝 WAEC Exam Practice" and st.session_state.exam_active:
  answered = (
      st.session_state.current_question_num - 1
      if st.session_state.current_question_num > 0
      else 0
  )
  total = st.session_state.total_questions
  left = max(total - answered, 0)
  correct = st.session_state.correct_count
  wrong = st.session_state.wrong_count

  progress_val = min(float(answered) / float(total), 1.0)
  st.progress(
      progress_val,
      text=(
          f"Progress Dashboard | Answered: {answered}/{total} | Left:"
          f" {left} | Correct: {correct} | Wrong: {wrong}"
      ),
  )

# Display chat history
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# Handle user input based on sidebar choice (Text or Voice)
user_query = None

if input_method == "⌨️ Type Question":
  if learning_mode == "📝 WAEC Exam Practice" and not st.session_state.exam_active:
    prompt_label = (
        "Type the total number of questions you want for this session (e.g.,"
        " 5):"
    )
  elif (
      learning_mode == "📝 WAEC Exam Practice" and st.session_state.exam_active
  ):
    prompt_label = (
        "Type your answer (Option letter A/B/C/D for objective, or complete"
        " response for essay)..."
    )
  else:
    prompt_label = f"Ask a question about {selected_subject}..."
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
            if not st.session_state.exam_active:
              # Try to parse total questions from user input
              try:
                parsed_total = int(user_query.strip())
                if parsed_total > 0:
                  st.session_state.total_questions = parsed_total
              except ValueError:
                pass  # Keep default if not integer

              st.session_state.exam_active = True
              st.session_state.current_question_num = 1
              st.session_state.correct_count = 0
              st.session_state.wrong_count = 0

              start_prompt = (
                  f"You are an expert WAEC Examiner in {selected_subject} testing"
                  f" {student_full_name} from {student_school}. The student"
                  f" requested an exam session of"
                  f" {st.session_state.total_questions} questions. Please"
                  " present Question 1. Specify clearly whether it is an"
                  " objective question (with options A, B, C, D) or an essay-type"
                  " question."
              )
              completion = client.chat.completions.create(
                  model="llama-3.1-8b-instant",
                  messages=[{"role": "user", "content": start_prompt}],
                  max_tokens=400,
                  temperature=0.3,
              )
              ai_response = completion.choices[0].message.content
              st.markdown(ai_response)
              st.session_state.messages.append(
                  {"role": "assistant", "content": ai_response}
              )
            else:
              # Active exam evaluation step
              st.session_state.current_question_num += 1
              current_q = st.session_state.current_question_num
              total_q = st.session_state.total_questions

              if current_q <= total_q:
                exam_eval_prompt = (
                    f"You are an expert WAEC Examiner in {selected_subject} testing"
                    f" {student_full_name} from {student_school}. \n\nEvaluate"
                    f" the student's latest answer: '{user_query}' for the"
                    " current question.\n\nRules:\n1. For Multiple Choice"
                    " (objective) questions: Do NOT give percentage scores. Give"
                    " a friendly remark of EXCELLENT, GREAT JOB, AMAZING,"
                    " WONDERFUL, CONGRATULATIONS if correct, or TRY AGAIN if"
                    " wrong. If wrong, display the correct option and answer to"
                    " aid learning.\n2. For Essay-type questions: Evaluate the"
                    " response thoroughly based on textbook standards.\n3."
                    " Present the next question (Question {current_q} of"
                    f" {total_q})."
                )
              else:
                # Exam session exhausted, provide final performance rating and remark
                exam_eval_prompt = (
                    f"You are an expert WAEC Examiner in {selected_subject} testing"
                    f" {student_full_name} from {student_school}. The student"
                    f" has completed all {total_q} questions in this session."
                    f" Total Correct: {st.session_state.correct_count}, Total"
                    f" Wrong: {st.session_state.wrong_count}.\n\nProvide an"
                    " overall performance rating and a final summary remark to"
                    " help the learner prepare for WAEC. Reset exam session"
                    " state after this."
                )
                st.session_state.exam_active = False

              completion = client.chat.completions.create(
                  model="llama-3.1-8b-instant",
                  messages=[
                      {"role": "system", "content": exam_eval_prompt},
                      {
                          "role": "user",
                          "content": (
                              f"Textbook Context:\n{retrieved_context}\n\nStudent"
                              f" Response: {user_query}"
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
                max_tokens=400,
                temperature=0.3,
            )
            ai_response = completion.choices[0].message.content
            st.markdown(ai_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": ai_response}
            )

          if user_key in st.session_state.user_sessions:
            st.session_state.user_sessions[user_key]["messages"] = (
                st.session_state.messages
            )

        except Exception as e:
          st.error(f"Error connecting to AI service: {e}")
