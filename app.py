import base64
import os
import random
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
              " Computing, ICT, Robotics, multiple choice labels A B C D."
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

# Handle subject switch notification with clear indication
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

# Sub-dropdown for WAEC Exam Practice question types
exam_question_type = "MCQ"
if learning_mode == "📝 WAEC Exam Practice":
  st.sidebar.markdown("---")
  exam_question_type = st.sidebar.selectbox(
      "Select Question Type", ["MCQ", "Short Answer", "Essay"]
  )

st.sidebar.markdown("---")
input_method = st.sidebar.radio(
    "Choose input method:", ["⌨️ Type Question", "🎤 Speak Question"]
)

# Load data for selected subject
dataset_files = subjects[selected_subject]
df_dataset = load_dataset(dataset_files)

# Layout: Laptop icon -> Title -> User picture (ONORE_AKORTIA_1.jpg) in a single uniform column
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

# Initialize session state stores
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
if "asked_questions" not in st.session_state:
  st.session_state.asked_questions = []

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

# Trigger initial personalized greeting ONLY when ALL THREE fields (Subject, Full Name, School) are provided
if not st.session_state.greeted and student_full_name and student_school:
  user_status_label = (
      "Welcome back"
      if user_key in st.session_state.user_sessions
      else "Welcome (First-time user)"
  )
  initial_greeting = (
      f"Hello {student_full_name} from {student_school}! ({user_status_label})."
      f" I am Sir O.K, your AI tutor ready to help you master {selected_subject}"
      " for WAEC. How can I assist you today?"
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
          f"Progress Dashboard ({exam_question_type}) | Answered:"
          f" {answered}/{total} | Left: {left} | Correct: {correct} | Wrong:"
          f" {wrong}"
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
        f"Type your answer for {exam_question_type} (e.g., option letter A/B/C/D"
        " for MCQ, short answer phrase, or essay write-up)..."
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

  filtered_context = "No specific textbook content found."
  if not df_dataset.empty:
    query_terms = [
        term.strip() for term in user_query.split() if len(term) > 3
    ]
    if query_terms:
      pattern = "|".join(query_terms)
      matched_rows = df_dataset[
          df_dataset["question_text"].str.contains(
              pattern, case=False, na=False, regex=True
          )
          | df_dataset["answer_text"].str.contains(
              pattern, case=False, na=False, regex=True
          )
      ]
      if not matched_rows.empty:
        filtered_context = "\n".join(
            matched_rows["answer_text"].head(5).tolist()
        )
      else:
        filtered_context = "\n".join(
            df_dataset["answer_text"].head(5).tolist()
        )
    else:
      filtered_context = "\n".join(df_dataset["answer_text"].head(5).tolist())

  client = get_groq_client()
  if client:
    with st.chat_message("assistant"):
      with st.spinner("Thinking..."):
        try:
          if learning_mode == "📝 WAEC Exam Practice":
            if not st.session_state.exam_active:
              try:
                parsed_total = int(user_query.strip())
                if parsed_total > 0:
                  st.session_state.total_questions = parsed_total
              except ValueError:
                pass

              st.session_state.exam_active = True
              st.session_state.current_question_num = 1
              st.session_state.correct_count = 0
              st.session_state.wrong_count = 0
              st.session_state.asked_questions = []

              start_prompt = (
                  f"You are Sir O.K, an expert WAEC Examiner in"
                  f" {selected_subject} testing {student_full_name} from"
                  f" {student_school}. The student requested an exam session"
                  f" of {st.session_state.total_questions} questions focusing"
                  f" specifically on topic or request: '{user_query}' and"
                  f" question type: {exam_question_type}. Please generate"
                  f" Question 1 strictly relevant to the requested topic.\n\n"
                  f"CRITICAL FORMATTING RULES:\n"
                  f"1. The question number and the actual question text MUST"
                  f" be on separate lines (e.g., 'Question 1\n\nWhat is...').\n"
                  f"2. Every question must include a complete, explicit question"
                  f" statement—never output only options or plausible"
                  f" answers.\n"
                  f"3. For MCQ, display options clearly (in a Markdown table"
                  f" or clean vertical layout).\n"
                  f"4. ABSOLUTELY DO NOT output the correct answer or solution"
                  f" key at this stage. Only output the question and options.\n"
                  f"5. NEVER output introductory meta-text explaining your"
                  f" evaluation process."
              )
              completion = client.chat.completions.create(
                  model="llama-3.1-8b-instant",
                  messages=[
                      {
                          "role": "system",
                          "content": (
                              f"Relevant Textbook Content:\n{filtered_context}"
                          ),
                      },
                      {"role": "user", "content": start_prompt},
                  ],
                  max_tokens=400,
                  temperature=0.3,
              )
              ai_response = completion.choices[0].message.content
              st.session_state.asked_questions.append(ai_response)
              st.markdown(ai_response)
              st.session_state.messages.append(
                  {"role": "assistant", "content": ai_response}
              )
            else:
              # Evaluate previous answer first, then present next question or final score
              current_q = st.session_state.current_question_num
              total_q = st.session_state.total_questions

              # Simple heuristic check or evaluation prompt to determine correct/wrong for score tracking
              # We can instruct model to output a distinct marker or parse correctness, or evaluate via LLM.
              # Let's ask the model to evaluate and include a hidden tag or clear text we can parse, or update counters.
              # To make it robust, we evaluate via prompt and update counters based on response keywords or dedicated evaluation.

              # Let's run evaluation and next step generation in one call
              is_last_question = current_q >= total_q

              eval_and_next_prompt = (
                  f"You are Sir O.K, an expert WAEC Examiner in"
                  f" {selected_subject} testing {student_full_name} from"
                  f" {student_school}.\n\n"
                  f"The student's latest answer for Question {current_q} of"
                  f" {total_q} is: '{user_query}'.\n\n"
                  f"PREVIOUSLY ASKED QUESTIONS (DO NOT REPEAT THESE TOPICS OR"
                  f" QUESTIONS):\n"
                  f"{chr(10).join(st.session_state.asked_questions)}\n\n"
                  f"Strict Rules:\n"
                  f"1. Evaluate if the student's answer is correct or wrong."
                  f" Based on correctness, update internal scores.\n"
                  f"2. Start your response with a randomized, enthusiastic"
                  f" remark addressing the student by name. VARY the remark"
                  f" every time (e.g. if correct: 'FANTASTIC WORK, {student_full_name}!'"
                  f" or 'BRILLIANT EFFORT, {student_full_name}!'; if wrong: 'GOOD"
                  f" TRY, {student_full_name}, LET'S REVIEW THIS CONCEPT' or"
                  f" 'NOT QUITE, {student_full_name}, BUT KEEP PUSHING'). NEVER"
                  f" repeat the exact same phrase always.\n"
                  f"3. If wrong, explicitly state and display the correct"
                  f" answer.\n"
                  f"4. DO NOT display the options of the previous question"
                  f" again.\n"
              )

              if not is_last_question:
                next_q_num = current_q + 1
                eval_and_next_prompt += (
                    f"5. Since this is NOT the final question, present"
                    f" Question {next_q_num} of {total_q} on the topic"
                    f" requested by the user, ensuring it is entirely NEW and"
                    f" unrepeated.\n"
                    f"6. Format rules for the new question: Question number"
                    f" and question text MUST be on separate lines. Every"
                    f" question must include a full question statement. For"
                    f" MCQ, present options cleanly without revealing the"
                    f" correct answer."
                )
              else:
                eval_and_next_prompt += (
                    f"5. Since this was the FINAL question (Question {current_q}"
                    f" of {total_q}), DO NOT present any new questions. Instead,"
                    f" display the OVERALL FINAL SCORE and performance summary"
                    f" for {student_full_name} immediately after the"
                    f" evaluation remark."
                )

              completion = client.chat.completions.create(
                  model="llama-3.1-8b-instant",
                  messages=[
                      {
                          "role": "system",
                          "content": (
                              f"Relevant Textbook Content:\n{filtered_context}"
                          ),
                      },
                      {"role": "system", "content": eval_and_next_prompt},
                      {
                          "role": "user",
                          "content": f"Evaluate student response: {user_query}",
                      },
                  ],
                  max_tokens=500,
                  temperature=0.4,
              )
              ai_response = completion.choices[0].message.content

              # Simple heuristic to update counters based on LLM output or user input matching
              # Let's inspect response or use a simple check. To be extremely precise with dashboard updates:
              response_lower = ai_response.lower()
              if (
                  "correct" in response_lower
                  or "excellent" in response_lower
                  or "fantastic" in response_lower
                  or "brilliant" in response_lower
                  or "right" in response_lower
              ) and "not correct" not in response_lower:
                st.session_state.correct_count += 1
              else:
                st.session_state.wrong_count += 1

              st.session_state.current_question_num += 1
              if is_last_question:
                st.session_state.exam_active = False

              st.session_state.asked_questions.append(ai_response)
              st.markdown(ai_response)
              st.session_state.messages.append(
                  {"role": "assistant", "content": ai_response}
              )
          else:
            persona_prompt = (
                f"You are Sir O.K, an expert SHS AI Tutor helping"
                f" {student_full_name} from {student_school} in"
                f" {selected_subject}. Provide precise, concise, and direct"
                " answers based on the provided textbook context."
            )

            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Relevant Textbook Content:\n{filtered_context}"
                        ),
                    },
                    {"role": "system", "content": persona_prompt},
                    {"role": "user", "content": f"Student Input: {user_query}"},
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
