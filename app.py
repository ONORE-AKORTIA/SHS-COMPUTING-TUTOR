import base64
import os
import random
import pandas as pd
import streamlit as st
from groq import Groq

# Centralized model configuration
ACTIVE_MODEL = "openai/gpt-oss-20b"

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

# Initialize session state stores & Cognitive Student Model variables
if "user_sessions" not in st.session_state:
    st.session_state.user_sessions = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "greeted" not in st.session_state:
    st.session_state.greeted = False

# Exam session & Knowledge Tracing state variables
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
if "current_correct_option" not in st.session_state:
    st.session_state.current_correct_option = None
if "current_topic" not in st.session_state:
    st.session_state.current_topic = "General Concept"
if "topic_performance" not in st.session_state:
    st.session_state.topic_performance = {}  # {topic: {"correct": x, "total": y}}

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

# Trigger initial personalized greeting ONLY when ALL THREE fields are provided
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
            f"Progress Dashboard ({exam_question_type}) | Current Topic:"
            f" {st.session_state.current_topic} | Answered: {answered}/{total} |"
            f" Correct: {correct} | Wrong: {wrong}"
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
                            st.session_state.current_correct_option = None
                            st.session_state.topic_performance = {}

                            start_prompt = (
                                f"You are Sir O.K, an expert WAEC Examiner in"
                                f" {selected_subject} testing {student_full_name} from"
                                f" {student_school}. The student requested an exam session"
                                f" of {st.session_state.total_questions} questions focusing"
                                f" specifically on topic or request: '{user_query}' and"
                                f" question type: {exam_question_type}. Please generate"
                                f" Question 1 strictly relevant toграмм the requested topic.\n\n"
                                f"CRITICAL FORMATTING & METADATA RULES:\n"
                                f"1. The question number (e.g., 'Question 1') and the actual"
                                f" question text MUST be on separate lines using double"
                                f" newlines.\n"
                                f"2. Every question must include a complete, explicit question"
                                f" statement.\n"
                                f"3. For MCQ, options MUST be presented in a nicely formatted"
                                f" Markdown table with columns 'Option' and 'Description', labeled"
                                f" A, B, C, and D.\n"
                                f"4. ABSOLUTELY DO NOT output the correct answer key in the main text.\n"
                                f"5. Include the specific syllabus topic name at the very end in format [TOPIC: Name of Topic].\n"
                                f"6. Include the correct option tag at the very end in format [CORRECT: X] (e.g. [CORRECT: B])."
                            )
                            completion = client.chat.completions.create(
                                model=ACTIVE_MODEL,
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

                            # Parse Topic metadata
                            topic_name = "General Concept"
                            if "[topic:" in ai_response.lower():
                                try:
                                    parts = ai_response.lower().split("[topic:")
                                    topic_name = (
                                        parts[1].split("]")[0].strip().title()
                                    )
                                except Exception:
                                    pass
                            st.session_state.current_topic = topic_name

                            # Parse Correct Option metadata
                            correct_letter = "A"
                            if "[correct:" in ai_response.lower():
                                try:
                                    parts = ai_response.lower().split("[correct:")
                                    correct_letter = (
                                        parts[1].split("]")[0].strip()[0].upper()
                                    )
                                except Exception:
                                    pass
                            st.session_state.current_correct_option = correct_letter

                            # Clean display response by stripping hidden tags
                            display_response = ai_response
                            for tag in ["[correct:", "[topic:"]:
                                if tag in display_response.lower():
                                    display_response = display_response.split(
                                        tag.upper()
                                    )[0].split(tag)[0]
                            display_response = display_response.strip()

                            st.session_state.asked_questions.append(display_response)
                            st.markdown(display_response)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": display_response}
                            )
                        else:
                            current_q = st.session_state.current_question_num
                            total_q = st.session_state.total_questions
                            is_last_question = current_q >= total_q

                            user_ans_clean = user_query.strip().upper()
                            expected_letter = (
                                st.session_state.current_correct_option or "A"
                            )
                            active_topic = st.session_state.current_topic

                            is_correct = False
                            if (
                                user_ans_clean == expected_letter
                                or user_ans_clean.startswith(expected_letter)
                            ):
                                is_correct = True

                            # Update Cognitive Model Knowledge Tracing Tracker
                            if (
                                active_topic
                                not in st.session_state.topic_performance
                            ):
                                st.session_state.topic_performance[active_topic] = {
                                    "correct": 0,
                                    "total": 0,
                                }
                            st.session_state.topic_performance[active_topic][
                                "total"
                            ] += 1
                            if is_correct:
                                st.session_state.topic_performance[active_topic][
                                    "correct"
                                ] += 1

                            if is_correct:
                                st.session_state.correct_count += 1
                                eval_remark = random.choice([
                                    f"EXCELLENT WORK, {student_full_name}! YOU NAILED IT!",
                                    f"BRILLIANT EFFORT, {student_full_name}! THAT IS SPOT ON!",
                                    (
                                        f"FANTASTIC, {student_full_name}! YOU'RE MAKING GREAT"
                                        " PROGRESS!"
                                    ),
                                ])
                            else:
                                st.session_state.wrong_count += 1
                                eval_remark = random.choice([
                                    (
                                        f"GOOD TRY, {student_full_name}, BUT NOT QUITE RIGHT."
                                        f" THE CORRECT ANSWER WAS OPTION {expected_letter}."
                                    ),
                                    (
                                        f"NOT TO WORRY, {student_full_name}, LET'S LEARN FROM"
                                        f" THIS. THE CORRECT ANSWER IS OPTION"
                                        f" {expected_letter}."
                                    ),
                                    (
                                        f"KEEP PUSHING, {student_full_name}! THE CORRECT OPTION"
                                        f" FOR THE PREVIOUS QUESTION WAS {expected_letter}."
                                    ),
                                ])

                            # Build cognitive diagnostic breakdown if last question
                            cognitive_summary_text = ""
                            if is_last_question:
                                cognitive_summary_text = (
                                    "\n\n### 🧠 Cognitive Knowledge Tracing Report\n"
                                )
                                for top, stats in st.session_state.topic_performance.items():
                                    pct = (
                                        (stats["correct"] / stats["total"]) * 100
                                        if stats["total"] > 0
                                        else 0
                                    )
                                    status_emoji = (
                                        "🟢 Mastered"
                                        if pct >= 70
                                        else "🔴 Needs Revision"
                                    )
                                    cognitive_summary_text += f"- **{top}**: {stats['correct']}/{stats['total']} correct ({pct:.0f}%) — {status_emoji}\n"

                            eval_and_next_prompt = (
                                f"You are Sir O.K, an expert WAEC Examiner in"
                                f" {selected_subject} testing {student_full_name} from"
                                f" {student_school}.\n\n"
                                f"Evaluation Result for Question {current_q} (Topic: {active_topic}):"
                                f" Student answered '{user_query}'. Evaluation is"
                                f" {'CORRECT' if is_correct else 'INCORRECT'} (Correct option"
                                f" was {expected_letter}).\n\n"
                                f"PREVIOUSLY ASKED QUESTIONS (DO NOT REPEAT):\n"
                                f"{chr(10).join(st.session_state.asked_questions)}\n\n"
                                f"Instructions:\n"
                                f"1. Start your response with the evaluation remark:"
                                f" '{eval_remark}'\n"
                                f"2. DO NOT display the options or table of the previous question.\n"
                            )

                            if not is_last_question:
                                next_q_num = current_q + 1
                                eval_and_next_prompt += (
                                    f"3. Present Question {next_q_num} of {total_q} on the"
                                    f" topic requested by the user, ensuring it is entirely NEW"
                                    f" and unrepeated.\n"
                                    f"4. Format rules: Question number and text on separate lines. Provide MCQ table options A, B, C, D.\n"
                                    f"5. Include topic tag [TOPIC: Topic Name] and correct option tag [CORRECT: X] at the very end."
                                )
                            else:
                                eval_and_next_prompt += (
                                    f"3. Since this was the FINAL question (Question {current_q}"
                                    f" of {total_q}), display the OVERALL FINAL SCORE and comprehensive"
                                    f" performance summary for {student_full_name} immediately"
                                    f" following the evaluation remark, including this exact cognitive knowledge tracing report:\n"
                                    f"{cognitive_summary_text}"
                                )

                            completion = client.chat.completions.create(
                                model=ACTIVE_MODEL,
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
                                        "content": f"Proceed with evaluation and next step.",
                                    },
                                ],
                                max_tokens=600,
                                temperature=0.4,
                            )
                            ai_response = completion.choices[0].message.content

                            # Parse next question metadata if not last question
                            if not is_last_question:
                                if "[topic:" in ai_response.lower():
                                    try:
                                        parts = ai_response.lower().split("[topic:")
                                        st.session_state.current_topic = (
                                            parts[1].split("]")[0].strip().title()
                                        )
                                    except Exception:
                                        pass

                                if "[correct:" in ai_response.lower():
                                    try:
                                        parts = ai_response.lower().split("[correct:")
                                        st.session_state.current_correct_option = (
                                            parts[1].split("]")[0].strip()[0].upper()
                                        )
                                    except Exception:
                                        pass

                            display_response = ai_response
                            for tag in ["[correct:", "[topic:"]:
                                if tag in display_response.lower():
                                    display_response = display_response.split(
                                        tag.upper()
                                    )[0].split(tag)[0]
                            display_response = display_response.strip()

                            st.session_state.current_question_num += 1
                            if is_last_question:
                                st.session_state.exam_active = False

                            st.session_state.asked_questions.append(display_response)
                            st.markdown(display_response)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": display_response}
                            )
                    else:
                        persona_prompt = (
                            f"You are Sir O.K, an expert SHS AI Tutor helping"
                            f" {student_full_name} from {student_school} in"
                            f" {selected_subject}. Provide precise, concise, and direct"
                            " answers based on the provided textbook context."
                        )

                        completion = client.chat.completions.create(
                            model=ACTIVE_MODEL,
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        f"Relevant Textbook Content:\n{filtered_context}"
                                    ),
                                },
                                {"role": "system", "content": persona_prompt},
                                {"role": "user", "content": user_query},
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
