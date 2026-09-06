import base64
import os
import random
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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
    "Select Mode", ["💬 Study & Chat", "📝 WAEC Exam Practice", "🎨 Whiteboard Concept Studio"]
)

# Sub-dropdown for WAEC Exam Practice question types
exam_question_type = "MCQ"
if learning_mode == "📝 WAEC Exam Practice":
    st.sidebar.markdown("---")
    exam_question_type = st.sidebar.selectbox(
        "Select Question Type", ["MCQ", "Short Answer", "Essay"]
    )
    
    if "prev_learning_mode" not in st.session_state:
        st.session_state.prev_learning_mode = learning_mode
    if st.session_state.prev_learning_mode != learning_mode:
        st.session_state.exam_active = False
        st.session_state.exam_state_stage = "awaiting_config"
        st.session_state.prev_learning_mode = learning_mode

st.sidebar.markdown("---")
input_method = st.sidebar.radio(
    "Choose input method:", ["⌨️ Type Question", "🎤 Speak Question"]
)

if learning_mode == "📝 WAEC Exam Practice":
    if st.sidebar.button("🔄 Restart Exam Session"):
        st.session_state.exam_active = False
        st.session_state.exam_state_stage = "awaiting_config"
        st.session_state.current_question_num = 1
        st.session_state.correct_count = 0
        st.session_state.wrong_count = 0
        st.session_state.asked_questions = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"Exam session reset. Please type your desired topic and number of questions (e.g., 'Databases, 2 questions')."
        })
        st.rerun()

# Load data for selected subject
dataset_files = subjects[selected_subject]
df_dataset = load_dataset(dataset_files)

# Layout: Laptop icon -> Title -> User picture (ONORE_AKORTIA_1.jpg)
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
if "exam_state_stage" not in st.session_state:
    st.session_state.exam_state_stage = "awaiting_config"
if "total_questions" not in st.session_state:
    st.session_state.total_questions = 2
if "target_topic" not in st.session_state:
    st.session_state.target_topic = "General Syllabus"
if "current_question_num" not in st.session_state:
    st.session_state.current_question_num = 1
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
    st.session_state.topic_performance = {}
if "last_revision_guide" not in st.session_state:
    st.session_state.last_revision_guide = None

user_key = f"{student_full_name.strip().lower()}_{student_school.strip().lower()}"

if student_full_name and student_school:
    if user_key not in st.session_state.user_sessions and st.session_state.greeted:
        st.session_state.user_sessions[user_key] = {
            "messages": st.session_state.messages,
            "is_returning": True,
        }
    elif user_key in st.session_state.user_sessions and not st.session_state.greeted:
        st.session_state.messages = st.session_state.user_sessions[user_key]["messages"]

if not st.session_state.greeted and student_full_name and student_school:
    user_status_label = "Welcome back" if user_key in st.session_state.user_sessions else "Welcome (First-time user)"
    initial_greeting = (
        f"Hello {student_full_name} from {student_school}! ({user_status_label})."
        f" I am Sir O.K, your AI tutor ready to help you master {selected_subject}"
        " for WAEC. How can I assist you today?"
    )
    if learning_mode == "📝 WAEC Exam Practice":
        initial_greeting += f"\n\n👉 **Exam Practice Ready:** Please type your desired topic and number of questions below (e.g., *'Networking, 2 questions'*)."
    elif learning_mode == "🎨 Whiteboard Concept Studio":
        initial_greeting += f"\n\n🎨 **Whiteboard Studio Ready:** Watch animated network topologies and computing concepts come alive with audio explanations!"

    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
    st.session_state.greeted = True
    if user_key not in st.session_state.user_sessions:
        st.session_state.user_sessions[user_key] = {
            "messages": st.session_state.messages,
            "is_returning": False,
        }

if learning_mode == "📝 WAEC Exam Practice" and not st.session_state.exam_active and st.session_state.exam_state_stage == "awaiting_config":
    if not any("Exam Practice Ready" in m["content"] or "topic and number of questions" in m["content"] for m in st.session_state.messages[-2:]):
        config_prompt = f"📝 **WAEC Exam Practice Mode Activated ({exam_question_type})**. Please type your desired topic and number of questions below:"
        st.session_state.messages.append({"role": "assistant", "content": config_prompt})

if (
    learning_mode == "📝 WAEC Exam Practice"
    and st.session_state.exam_active
    and st.session_state.exam_state_stage == "in_progress"
):
    answered = st.session_state.current_question_num - 1
    total = st.session_state.total_questions
    correct = st.session_state.correct_count
    wrong = st.session_state.wrong_count

    progress_val = min(float(answered) / float(total), 1.0)
    st.progress(
        progress_val,
        text=(
            f"Progress Dashboard ({exam_question_type}) | Topic Focus:"
            f" {st.session_state.target_topic} | Question:"
            f" {st.session_state.current_question_num}/{total} | Correct:"
            f" {correct} | Wrong: {wrong}"
        ),
    )

# ==========================================================
# 🎨 WHITEBOARD CONCEPT STUDIO (WITH RESPONSIVE MEDIA PLAYER & CONTROLS)
# ==========================================================
if learning_mode == "🎨 Whiteboard Concept Studio":
    st.markdown("### 🎨 Sir O.K Animated Whiteboard Studio")
    
    wb_concept = st.selectbox(
        "Select Concept to Animate & Explain:",
        ["Star Network Topology", "Bus Network Topology", "Ring Network Topology", "SQL Database JOINs", "CPU Fetch-Decode-Execute Cycle"]
    )
    
    # Generate audio explanation script via Groq
    client = get_groq_client()
    explanation_text = f"Welcome to Sir O.K's Whiteboard Studio. Today we are exploring {wb_concept}. In this architecture, data is transmitted efficiently across nodes, ensuring reliability and performance for WAEC examinations."
    if client:
        try:
            comp = client.chat.completions.create(
                model=ACTIVE_MODEL,
                messages=[{"role": "user", "content": f"Provide a brief, 2-sentence audio narration script explaining {wb_concept} for WAEC students."}],
                max_tokens=100,
            )
            explanation_text = comp.choices[0].message.content
        except Exception:
            pass

    # Fully Responsive HTML/JS Animated Video Player Component with Browser Text-to-Speech Audio & Visible Controls
    player_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            background-color: #121212;
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 5px;
            box-sizing: border-box;
        }}
        .player-container {{
            background: #1e1e1e;
            border: 3px solid #00ffcc;
            border-radius: 12px;
            padding: 12px;
            box-shadow: 0 4px 20px rgba(0,255,204,0.2);
            width: 100%;
            max-width: 100%;
            box-sizing: border-box;
        }}
        .player-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            border-bottom: 1px solid #333;
            padding-bottom: 6px;
            flex-wrap: wrap;
            gap: 6px;
        }}
        .brand {{
            color: #00ffcc;
            font-weight: bold;
            font-size: 0.95em;
        }}
        .topic-badge {{
            background: #282828;
            border: 1px solid #00ffcc;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 0.85em;
            color: #00ffcc;
            font-weight: bold;
        }}
        .screen {{
            position: relative;
            width: 100%;
            padding-bottom: 50%;
            background: #0a0a0a;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #333;
        }}
        .screen svg {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }}
        .controls-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 10px;
            background: #252525;
            padding: 8px 12px;
            border-radius: 8px;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .btn-group {{
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }}
        button {{
            background: #333;
            color: #fff;
            border: 1px solid #555;
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.85em;
        }}
        button:hover {{
            background: #00ffcc;
            color: #000;
            border-color: #00ffcc;
        }}
        #playBtn {{
            background: #00ffcc;
            color: #000;
            border-color: #00ffcc;
        }}
        .status {{
            font-size: 0.8em;
            color: #aaa;
        }}
        .narration {{
            margin-top: 10px;
            background: #181818;
            border-left: 4px solid #00ffcc;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.9em;
            line-height: 1.4;
            color: #ddd;
        }}
    </style>
    </head>
    <body>
    <div class="player-container">
        <div class="player-header">
            <span class="brand">📺 SIR O.K. WHITEBOARD STUDIO</span>
            <span class="topic-badge" id="topicTitle">{wb_concept}</span>
        </div>
        
        <div class="screen">
            <svg id="wbSvg" viewBox="0 0 600 300">
                <circle cx="300" cy="150" r="32" fill="#1a1a1a" stroke="#00ffcc" stroke-width="3" />
                <text x="300" y="146" fill="#00ffcc" font-size="10" font-weight="bold" text-anchor="middle">CENTRAL</text>
                <text x="300" y="160" fill="#00ffcc" font-size="10" font-weight="bold" text-anchor="middle">SWITCH</text>
                
                <line x1="300" y1="150" x2="110" y2="70" stroke="#444" stroke-width="2" stroke-dasharray="4"/>
                <line x1="300" y1="150" x2="490" y2="70" stroke="#444" stroke-width="2" stroke-dasharray="4"/>
                <line x1="300" y1="150" x2="110" y2="230" stroke="#444" stroke-width="2" stroke-dasharray="4"/>
                <line x1="300" y1="150" x2="490" y2="230" stroke="#444" stroke-width="2" stroke-dasharray="4"/>

                <circle cx="0" cy="0" r="6" fill="#ff0055">
                    <animateMotion id="m1" path="M 300,150 L 110,70" dur="2s" repeatCount="indefinite" />
                </circle>
                <circle cx="0" cy="0" r="6" fill="#00ffcc">
                    <animateMotion id="m2" path="M 300,150 L 490,70" dur="1.5s" repeatCount="indefinite" />
                </circle>
                <circle cx="0" cy="0" r="6" fill="#ffbb00">
                    <animateMotion id="m3" path="M 110,230 L 300,150" dur="2.2s" repeatCount="indefinite" />
                </circle>
                <circle cx="0" cy="0" r="6" fill="#00ffcc">
                    <animateMotion id="m4" path="M 490,230 L 300,150" dur="1.8s" repeatCount="indefinite" />
                </circle>

                <g transform="translate(110, 70)">
                    <rect x="-26" y="-18" width="52" height="36" rx="6" fill="#2a2a2a" stroke="#fff" stroke-width="2"/>
                    <text x="0" y="4" fill="#fff" font-size="10" font-weight="bold" text-anchor="middle">PC 1</text>
                </g>
                <g transform="translate(490, 70)">
                    <rect x="-26" y="-18" width="52" height="36" rx="6" fill="#2a2a2a" stroke="#fff" stroke-width="2"/>
                    <text x="0" y="4" fill="#fff" font-size="10" font-weight="bold" text-anchor="middle">PC 2</text>
                </g>
                <g transform="translate(110, 230)">
                    <rect x="-26" y="-18" width="52" height="36" rx="6" fill="#2a2a2a" stroke="#fff" stroke-width="2"/>
                    <text x="0" y="4" fill="#fff" font-size="10" font-weight="bold" text-anchor="middle">PC 3</text>
                </g>
                <g transform="translate(490, 230)">
                    <rect x="-26" y="-18" width="52" height="36" rx="6" fill="#2a2a2a" stroke="#fff" stroke-width="2"/>
                    <text x="0" y="4" fill="#fff" font-size="10" font-weight="bold" text-anchor="middle">PC 4</text>
                </g>
            </svg>
        </div>

        <div class="controls-bar">
            <div class="btn-group">
                <button id="playBtn" onclick="togglePlay()">⏸️ Pause</button>
                <button onclick="restartPlayer()">🔄 Restart</button>
            </div>
            <div class="status" id="statusText">Status: Playing</div>
        </div>

        <div class="narration">
            <b>🎙️ Sir O.K Audio Narration:</b> <span id="narrationText">{explanation_text}</span>
        </div>
    </div>

    <script>
        let isPlaying = true;
        const svg = document.getElementById('wbSvg');
        const narrationString = `{explanation_text}`;

        function speakNarration() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(narrationString);
                utterance.rate = 1.0;
                utterance.pitch = 1.0;
                window.speechSynthesis.speak(utterance);
            }}
        }}

        window.addEventListener('DOMContentLoaded', () => {{
            speakNarration();
        }});

        function togglePlay() {{
            isPlaying = !isPlaying;
            if (isPlaying) {{
                svg.unpauseAnimations();
                document.getElementById('playBtn').innerText = '⏸️ Pause';
                document.getElementById('statusText').innerText = 'Status: Playing';
                speakNarration();
            }} else {{
                svg.pauseAnimations();
                document.getElementById('playBtn').innerText = '▶️ Play';
                document.getElementById('statusText').innerText = 'Status: Paused';
                if ('speechSynthesis' in window) {{
                    window.speechSynthesis.cancel();
                }}
            }}
        }}

        function restartPlayer() {{
            svg.setCurrentTime(0);
            isPlaying = true;
            svg.unpauseAnimations();
            document.getElementById('playBtn').innerText = '⏸️ Pause';
            document.getElementById('statusText').innerText = 'Status: Playing';
            speakNarration();
        }}
    </script>
    </body>
    </html>
    """

    components.html(player_html, height=480, scrolling=False)

else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if st.session_state.last_revision_guide:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Offline Study Tools")
    st.sidebar.download_button(
        label="Download Revision Guide (.txt)",
        data=st.session_state.last_revision_guide,
        file_name=f"SirOK_{selected_subject}_Revision_Guide.txt",
        mime="text/plain",
    )

user_query = None

if learning_mode != "🎨 Whiteboard Concept Studio":
    if input_method == "⌨️ Type Question":
        if (
            learning_mode == "📝 WAEC Exam Practice"
            and st.session_state.exam_state_stage == "awaiting_config"
        ):
            prompt_label = (
                "Type your desired topic and number of questions (e.g., 'Databases,"
                " 2 questions'):"
            )
        elif (
            learning_mode == "📝 WAEC Exam Practice"
            and st.session_state.exam_state_stage == "in_progress"
        ):
            prompt_label = (
                f"Type your answer for Question {st.session_state.current_question_num}"
                f" of {st.session_state.total_questions} (A, B, C, or D)..."
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
            with st.spinner("Preparing response..."):
                try:
                    if learning_mode == "📝 WAEC Exam Practice":
                        if st.session_state.exam_state_stage == "awaiting_config":
                            import re

                            digits_found = re.findall(r"\d+", user_query)
                            parsed_total = (
                                int(digits_found[0]) if digits_found else 2
                            )
                            if parsed_total <= 0:
                                parsed_total = 2

                            st.session_state.total_questions = parsed_total

                            cleaned_topic = user_query
                            for d in digits_found:
                                cleaned_topic = cleaned_topic.replace(d, "")
                            for word in [
                                "questions",
                                "question",
                                "qrs",
                                "qs",
                                "tests",
                                "test",
                                "exams",
                                "exam",
                                "practice",
                                "for",
                                "on",
                                "about",
                                "need",
                                "want",
                                "give",
                                "me",
                                "mcq",
                                "mcqs",
                            ]:
                                cleaned_topic = re.sub(
                                    rf"\b{word}\b", "", cleaned_topic, flags=re.IGNORECASE
                                )
                            cleaned_topic = cleaned_topic.strip(" ,.-")
                            if not cleaned_topic:
                                cleaned_topic = "General Syllabus"

                            st.session_state.target_topic = cleaned_topic

                            st.session_state.exam_active = True
                            st.session_state.exam_state_stage = "in_progress"
                            st.session_state.current_question_num = 1
                            st.session_state.correct_count = 0
                            st.session_state.wrong_count = 0
                            st.session_state.asked_questions = []
                            st.session_state.current_correct_option = None
                            st.session_state.topic_performance = {}
                            st.session_state.last_revision_guide = None

                            start_prompt = (
                                f"You are Sir O.K, an expert WAEC Examiner in"
                                f" {selected_subject} testing {student_full_name} from"
                                f" {student_school}. The student requested an exam practice session"
                                f" of exactly **{st.session_state.total_questions} questions** focusing on: **{st.session_state.target_topic}** using question type: {exam_question_type}.\n\n"
                                f"INSTRUCTIONS:\n"
                                f"1. Begin with a brief enthusiastic confirmation and Question 1 right now.\n"
                                f"2. For MCQ, options MUST be presented in a nicely formatted Markdown table with columns 'Option' and 'Description', labeled A, B, C, and D.\n"
                                f"3. ABSOLUTELY DO NOT output the correct answer key in the main text.\n"
                                f"4. Include topic tag [TOPIC: Name of Topic] and correct option tag [CORRECT: X] at the very end."
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
                                max_tokens=500,
                                temperature=0.3,
                            )
                            ai_response = completion.choices[0].message.content

                            topic_name = st.session_state.target_topic
                            if "[topic:" in ai_response.lower():
                                try:
                                    parts = ai_response.lower().split("[topic:")
                                    topic_name = (
                                        parts[1].split("]")[0].strip().title()
                                    )
                                except Exception:
                                    pass
                            st.session_state.current_topic = topic_name

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
                            st.rerun()
                        else:
                            current_q = st.session_state.current_question_num
                            total_q = st.session_state.total_questions

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
                                eval_remark = f"EXCELLENT WORK, {student_full_name}! YOU NAILED IT!"
                            else:
                                st.session_state.wrong_count += 1
                                eval_remark = f"GOOD TRY, {student_full_name}. THE CORRECT ANSWER WAS OPTION {expected_letter}."

                            is_last_question = current_q >= total_q

                            cognitive_summary_text = ""
                            weak_topics = []
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
                                    if pct < 70:
                                        weak_topics.append(top)

                            prev_questions_text = ""
                            if st.session_state.asked_questions:
                                prev_questions_text = "\n".join(
                                    [
                                        f"Previous Question {i+1}:\n{q}"
                                        for i, q in enumerate(
                                            st.session_state.asked_questions
                                        )
                                    ]
                                )

                            eval_and_next_prompt = (
                                f"You are Sir O.K, an expert WAEC Examiner in"
                                f" {selected_subject} guiding {student_full_name}.\n\n"
                                f"Evaluation Result for Question {current_q} of {total_q} (Topic: {active_topic}):\n"
                                f"- Student Answer: '{user_query}'\n"
                                f"- Result: {'CORRECT' if is_correct else 'INCORRECT'} (Correct option was {expected_letter}).\n\n"
                            )

                            if not is_last_question:
                                next_q_num = current_q + 1
                                eval_and_next_prompt += (
                                    f"Present **Question {next_q_num} of {total_q}** on {st.session_state.target_topic}.\n"
                                    f"Include topic tag [TOPIC: Topic Name] and correct option tag [CORRECT: X] at the very end."
                                )
                            else:
                                evaluation_summary_score = (
                                    f"\n\n### 📊 Final Examination Summary\n"
                                    f"- **Total Questions:** {total_q}\n"
                                    f"- **Correct Answers:** {st.session_state.correct_count}\n"
                                    f"- **Wrong Answers:** {st.session_state.wrong_count}\n"
                                    f"- **Overall Score:** {(st.session_state.correct_count / total_q) * 100:.1f}%\n"
                                )
                                weak_topics_str = (
                                    ", ".join(weak_topics)
                                    if weak_topics
                                    else st.session_state.target_topic
                                )
                                eval_and_next_prompt += (
                                    f"Since this was the FINAL question, display score summary:\n{evaluation_summary_score}\n"
                                    f"Followed by cognitive report:\n{cognitive_summary_text}\n"
                                    f"Write an exhaustive, textbook-quality study guide targeting **{weak_topics_str}**."
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
                                        "content": "Proceed immediately.",
                                    },
                                ],
                                max_tokens=1500,
                                temperature=0.3,
                            )
                            ai_response = completion.choices[0].message.content

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
                            if not is_last_question:
                                for tag in ["[correct:", "[topic:"]:
                                    if tag in display_response.lower():
                                        display_response = display_response.split(
                                            tag.upper()
                                        )[0].split(tag)[0]
                            else:
                                st.session_state.last_revision_guide = (
                                    f"SIR O.K AI TUTOR - OFFICIAL REVISION GUIDE\n"
                                    f"Student: {student_full_name} | School: {student_school}\n"
                                    f"Subject: {selected_subject}\n"
                                    f"--------------------------------------------------\n\n"
                                    + display_response
                                )

                            display_response = display_response.strip()

                            if not is_last_question:
                                st.session_state.current_question_num += 1
                            else:
                                st.session_state.exam_active = False
                                st.session_state.exam_state_stage = (
                                    "awaiting_config"
                                )

                            st.session_state.asked_questions.append(display_response)
                            st.markdown(display_response)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": display_response}
                            )
                    else:
                        persona_prompt = (
                            f"You are Sir O.K, an expert SHS AI Tutor helping"
                            f" {student_full_name} from {student_school} in"
                            f" {selected_subject}."
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
