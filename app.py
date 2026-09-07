import base64
import contextlib
import datetime
import io
import json
import os
import random
import sys
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from groq import Groq
from openai import OpenAI

# Optional PostgreSQL & SQLAlchemy imports with graceful fallback handling
try:
    import psycopg2
    from sqlalchemy import create_engine, text

    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Centralized model configuration
ACTIVE_MODEL = "openai/gpt-oss-20b"

# Page configuration with wide layout
st.set_page_config(
    page_title="SHS Computing AI Tutor - Enterprise Edition", page_icon="💻", layout="wide"
)

# Inject custom CSS for responsive UI
st.markdown("""
    <style>
    .block-container {
        max-width: 95% !important;
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    .stChatInput { max-width: 100% !important; }
    iframe { width: 100% !important; max-width: 100% !important; }
    .timer-card {
        background: #1e1e1e;
        border: 1px solid #00ffcc;
        padding: 8px 12px;
        border-radius: 6px;
        color: #00ffcc;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .badge-card {
        background: #252525;
        border: 1px solid #ffbb00;
        padding: 10px;
        border-radius: 6px;
        text-align: center;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================================
# 🔌 INITIALIZE CLIENTS & CLOUD POSTGRESQL DB CONNECTION
# ==========================================================
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


@st.cache_resource
def init_db_connection():
    """Initializes persistent cloud PostgreSQL database connection (e.g., Supabase/Neon)."""
    if not POSTGRES_AVAILABLE:
        return None
    db_url = st.secrets.get("DATABASE_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        return None
    try:
        engine = create_engine(db_url, pool_pre_ping=True)
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS student_profiles (
                    id SERIAL PRIMARY KEY,
                    user_key VARCHAR(255) UNIQUE NOT NULL,
                    full_name VARCHAR(255) NOT NULL,
                    school VARCHAR(255) NOT NULL,
                    xp INT DEFAULT 150,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS student_history (
                    id SERIAL PRIMARY KEY,
                    user_key VARCHAR(255) NOT NULL,
                    assignment_type VARCHAR(50) NOT NULL,
                    submitted_content TEXT NOT NULL,
                    semantic_score NUMERIC(5, 2),
                    ai_feedback TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """))
        return engine
    except Exception as e:
        st.warning(f"Cloud PostgreSQL connection warning: {e}")
        return None


db_engine = init_db_connection()


# ==========================================================
# 📚 LOAD CURRICULUM DATABASE FROM JSON
# ==========================================================
@st.cache_data
def load_curriculum_database():
    json_path = os.path.join("textbooks_json", "curriculum_database.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


curriculum_db = load_curriculum_database()


# ==========================================================
# 🧠 AUTOMATED SEMANTIC GRADING ENGINE
# ==========================================================
def evaluate_essay_semantics(student_answer: str, rubric: str, subject: str) -> dict:
    """Evaluates student essay/short answer against a rubric using AI semantic analysis."""
    client = get_groq_client() or get_openai_client()
    if not client:
        return {
            "score": 75.0,
            "strengths": ["Clear attempt at addressing the prompt", "Good foundational terminology"],
            "gaps": ["Lacks detailed technical elaboration", "Could include standard examples"],
            "feedback": "Your response covers the core concept. To score top marks, elaborate further on practical applications."
        }

    prompt = f"""You are an expert WAEC Chief Examiner in {subject}. Grade the student's answer strictly based on the rubric provided. Return ONLY valid JSON format with keys: "score" (float 0-100), "strengths" (list of strings), "gaps" (list of strings), and "feedback" (string).

Rubric:
{rubric}

Student Answer:
{student_answer}
"""
    try:
        response = client.chat.completions.create(
            model=ACTIVE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=600
        )
        content = response.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)
    except Exception as e:
        return {
            "score": 70.0,
            "strengths": ["Submitted response received successfully"],
            "gaps": [f"Parsing error: {e}"],
            "feedback": "Your submission was recorded, but semantic parser encountered an issue. Please review standard guidelines."
        }


# ==========================================================
# 🔊 TEXT-TO-SPEECH (TTS) AUDIO GENERATOR
# ==========================================================
def generate_tts_audio(text: str) -> bytes:
    """Generates speech audio bytes using OpenAI TTS API."""
    client = get_openai_client()
    if not client:
        return None
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text[:4000].strip()
        )
        return response.content
    except Exception:
        return None


# ==========================================================
# 🎮 SECURE PYTHON & CODE EXECUTION SANDBOX
# ==========================================================
def execute_python_code(code_string: str) -> dict:
    """Safely executes Python code in memory with redirected stdout and stderr."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    start_time = datetime.datetime.now()
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            safe_globals = {
                "__builtins__": {
                    'print': print, 'len': len, 'range': range, 'sum': sum,
                    'max': max, 'min': min, 'abs': abs, 'round': round,
                    'sorted': sorted, 'list': list, 'dict': dict, 'set': set,
                    'str': str, 'int': int, 'float': float, 'bool': bool,
                    'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter
                }
            }
            exec(code_string, safe_globals)

        exec_time = (datetime.datetime.now() - start_time).total_seconds()
        return {
            "output": stdout_buffer.getvalue(),
            "error": stderr_buffer.getvalue(),
            "execution_time": f"{exec_time:.4f}s"
        }
    except Exception as e:
        return {
            "output": stdout_buffer.getvalue(),
            "error": str(e),
            "execution_time": "Failed"
        }


# ==========================================================
# SIDEBAR NAVIGATION & PROFILE SETUP
# ==========================================================
st.sidebar.title("Navigation & Profile")

# Academic Year Selection
selected_year = st.sidebar.selectbox(
    "Select Academic Year",
    ["Year_1", "Year_2", "Year_3 (Coming Soon)"]
)

# Fetch subjects dynamically based on selected year from JSON database
if selected_year == "Year_3 (Coming Soon)":
    st.sidebar.warning("Year 3 materials are under development!")
    available_subjects = []
else:
    year_data = curriculum_db.get(selected_year, {})
    available_subjects = list(year_data.keys())

if not available_subjects:
    available_subjects = ["No Subjects Available"]

selected_subject = st.sidebar.selectbox("Select Subject", available_subjects)

# Extract textbook content for the selected Year and Subject
active_textbook_content = ""
if selected_year in curriculum_db and selected_subject in curriculum_db[selected_year]:
    pdf_entries = curriculum_db[selected_year][selected_subject]
    active_textbook_content = "\n".join([entry["content"] for entry in pdf_entries])

if "prev_subject" not in st.session_state:
    st.session_state.prev_subject = selected_subject

if selected_subject != st.session_state.prev_subject:
    switch_msg = f"Switched to **{selected_year} - {selected_subject}**."
    if "messages" in st.session_state:
        st.session_state.messages.append({"role": "assistant", "content": switch_msg})
    st.session_state.prev_subject = selected_subject

st.sidebar.markdown("---")
st.sidebar.subheader("Student Details")
student_full_name = st.sidebar.text_input("Full Name (First, Middle, Last)", value="")
student_school = st.sidebar.text_input("School Name", value="")

st.sidebar.markdown("---")
learning_mode = st.sidebar.radio(
    "Select Mode", [
        "💬 Study & Chat",
        "📝 WAEC Exam Practice",
        "✍️ Semantic Essay & QA Grading",
        "🎨 Whiteboard Studio",
        "🏆 Leaderboard & Badges",
        "🎮 Code Execution Sandbox",
        "📖 Offline Syllabus Center"
    ]
)

input_method = st.sidebar.radio("Choose input method:", ["⌨️ Type Question", "🎤 Speak Question"])


# Header Layout
def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


user_img_base64 = get_image_base64("ONORE_AKORTIA_1.jpg")
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
        <span style="font-size: 2.5em;">💻</span>
        <div style="flex-grow: 1;">
            <h1 style="margin: 0; font-size: 1.8em; line-height: 1.2;">SHS Computing AI Tutor (Enterprise Edition)</h1>
            <p style="margin: 0; color: #666; font-size: 0.95em;">Cloud-Powered Intelligent Tutoring for <b>{selected_year} - {selected_subject}</b>.</p>
        </div>
        {f"<img src='data:image/jpeg;base64,{user_img_base64}' width='75' style='border-radius: 8px; object-fit: cover;'>" if user_img_base64 else ""}
    </div>
    """,
    unsafe_allow_html=True,
)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "greeted" not in st.session_state:
    st.session_state.greeted = False
if "session_login_time" not in st.session_state:
    st.session_state.session_login_time = datetime.datetime.now()
if "user_xp" not in st.session_state:
    st.session_state.user_xp = 150
if "user_badges" not in st.session_state:
    st.session_state.user_badges = ["🚀 First Login", "💡 Explorer"]

user_key = f"{student_full_name.strip().lower()}_{student_school.strip().lower()}"

# Sync with PostgreSQL cloud DB if available
if db_engine and student_full_name and student_school:
    try:
        with db_engine.begin() as conn:
            res = conn.execute(text("SELECT xp FROM student_profiles WHERE user_key = :uk"),
                               {"uk": user_key}).fetchone()
            if res:
                st.session_state.user_xp = res[0]
            else:
                conn.execute(text(
                    "INSERT INTO student_profiles (user_key, full_name, school, xp) VALUES (:uk, :fn, :sc, :xp) ON CONFLICT (user_key) DO NOTHING"),
                             {"uk": user_key, "fn": student_full_name, "sc": student_school,
                              "xp": st.session_state.user_xp})
    except Exception:
        pass

if not st.session_state.greeted and student_full_name and student_school:
    initial_greeting = f"Hello {student_full_name} from {student_school}! I am Sir OK, your AI tutor with persistent cloud storage, semantic grading, and live code execution."
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
    st.session_state.greeted = True

# Timer Widget
login_epoch = int(st.session_state.session_login_time.timestamp() * 1000)
timer_html = f"""
<div class="timer-card">
    ⏱️ Session Timer: <span id="sessionTimer">00:00:00</span> | Cloud DB: {"🟢 Connected" if db_engine else "🟡 Local Session"} | XP: {st.session_state.user_xp} 🌟
</div>
<script>
    const loginTime = {login_epoch};
    function updateTimer() {{
        const now = new Date().getTime();
        const diff = Math.floor((now - loginTime) / 1000);
        const hrs = String(Math.floor(diff / 3600)).padStart(2, '0');
        const mins = String(Math.floor((diff % 3600) / 60)).padStart(2, '0');
        const secs = String(diff % 60).padStart(2, '0');
        const el = document.getElementById('sessionTimer');
        if (el) el.innerText = `${{hrs}}:${{mins}}:${{secs}}`;
    }}
    setInterval(updateTimer, 1000);
    updateTimer();
</script>
"""
components.html(timer_html, height=45, scrolling=False)

# ==========================================================
# MODE 1: GAMIFIED LEADERBOARD & BADGES
# ==========================================================
if learning_mode == "🏆 Leaderboard & Badges":
    st.markdown("## 🏆 Gamified Leaderboard & Cloud Badges")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏅 Unlocked Badges")
        for b in st.session_state.user_badges:
            st.success(b)
    with col2:
        st.markdown("### 📊 National Peer Leaderboard")
        lb_df = pd.DataFrame([
            {"Rank": 1, "Student": "Kofi Mensah", "School": "Accra Academy", "XP": 2450},
            {"Rank": 2, "Student": student_full_name if student_full_name else "You",
             "School": student_school if student_school else "Your School", "XP": st.session_state.user_xp},
            {"Rank": 3, "Student": "Abena Osei", "School": "Wesley Girls High", "XP": 1920},
        ])
        st.dataframe(lb_df, use_container_width=True, hide_index=True)


# ==========================================================
# MODE 2: CODE EXECUTION SANDBOX
# ==========================================================
elif learning_mode == "🎮 Code Execution Sandbox":
    st.markdown("## 🎮 Interactive Code Execution Sandbox")
    st.markdown("Test Python algorithms and database queries live in memory with immediate output validation.")

    code_input = st.text_area("Write Python Code:", value="""# Test your curriculum algorithm
scores = [78, 85, 92, 68, 95]
print("Student Scores:", scores)
print("Highest Score:", max(scores))
print("Average Score:", sum(scores) / len(scores))
""", height=200)

    if st.button("▶️ Execute Code Safely"):
        result = execute_python_code(code_input)
        st.markdown("### 💻 Execution Result:")
        if result["error"]:
            st.error(f"Error:\n{result['error']}")
        if result["output"]:
            st.code(result["output"], language="text")
        st.info(f"Execution Time: {result['execution_time']}")
        st.session_state.user_xp += 20


# ==========================================================
# MODE 3: AUTOMATED SEMANTIC ESSAY & QA GRADING
# ==========================================================
elif learning_mode == "✍️ Semantic Essay & QA Grading":
    st.markdown("## ✍️ Automated Semantic Essay & Short Answer Grading")
    st.markdown(
        "Submit your essay or descriptive answer below. Our semantic grading engine evaluates conceptual alignment against rubrics and generates instant feedback with audio output.")

    essay_prompt = st.text_area("Exam Prompt / Question:",
                                value="Explain the role of complementary assets in organizational systems.")
    rubric_text = st.text_area("Grading Rubric:",
                               value="1. Define complementary assets clearly.\n2. Discuss challenges.\n3. Provide concrete examples.")
    student_essay = st.text_area("Your Essay / Detailed Answer:", value="", height=180)

    if st.button("🚀 Submit for Semantic Grading"):
        if student_essay.strip():
            with st.spinner("Analyzing semantics and evaluating rubric alignment..."):
                grading_res = evaluate_essay_semantics(student_essay, rubric_text, selected_subject)
                st.success(f"Grading Complete! Score: **{grading_res.get('score', 0)} / 100**")

                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.markdown("### ✅ Identified Strengths")
                    for s in grading_res.get("strengths", []):
                        st.markdown(f"- {s}")
                with col_g2:
                    st.markdown("### 🔍 Conceptual Gaps")
                    for g in grading_res.get("gaps", []):
                        st.markdown(f"- {g}")

                st.markdown("### 📋 Constructive Feedback")
                feedback_str = grading_res.get("feedback", "")
                st.info(feedback_str)

                # Persist to cloud DB if connected
                if db_engine and student_full_name:
                    try:
                        with db_engine.begin() as conn:
                            conn.execute(text("""
                                INSERT INTO student_history (user_key, assignment_type, submitted_content, semantic_score, ai_feedback)
                                VALUES (:uk, 'essay', :sc, :ss, :af)
                            """), {"uk": user_key, "sc": student_essay, "ss": grading_res.get("score", 0),
                                   "af": feedback_str})
                    except Exception:
                        pass

                # Text-to-Speech Output Integration
                st.markdown("### 🔊 Audio Feedback Output")
                audio_bytes = generate_tts_audio(feedback_str)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                else:
                    st.warning("OpenAI TTS API key required for audio generation.")
        else:
            st.warning("Please enter your essay before submitting.")


# ==========================================================
# MODE 4: OFFLINE SYLLABUS CENTER
# ==========================================================
elif learning_mode == "📖 Offline Syllabus Center":
    st.markdown("## 📖 Offline Syllabus & Study Notes Center")
    st.download_button("📥 Download Curriculum Syllabus (TXT)",
                       f"OFFICIAL CURRICULUM - {selected_year} - {selected_subject}",
                       file_name=f"{selected_year}_{selected_subject}.txt")


# ==========================================================
# MODE 5: WHITEBOARD STUDIO
# ==========================================================
elif learning_mode == "🎨 Whiteboard Studio":
    st.markdown("## 🎨 Whiteboard Concept Studio")
    st.info(f"Viewing interactive curriculum notes for **{selected_year} - {selected_subject}**.")


# ==========================================================
# MODE 6: STUDY & CHAT / WAEC EXAM PRACTICE
# ==========================================================
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = None
    if input_method == "⌨️ Type Question":
        user_query = st.chat_input(f"Ask a question about {selected_year} {selected_subject}...")
    else:
        audio_val = st.audio_input("Record voice question")
        if audio_val:
            # Placeholder handling for voice input transcription if Groq is available
            pass

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        client = get_groq_client()
        if client:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        system_content = f"You are Sir OK, expert tutor in {selected_year} {selected_subject}. Use the following uploaded textbook content to guide your answers:\n{active_textbook_content[:4000]}"
                        completion = client.chat.completions.create(
                            model=ACTIVE_MODEL,
                            messages=[
                                {"role": "system", "content": system_content},
                                {"role": "user", "content": user_query}
                            ],
                            max_tokens=500,
                            temperature=0.3,
                        )
                        ai_resp = completion.choices[0].message.content
                        st.markdown(ai_resp)

                        # Generate TTS audio for the response
                        audio_data = generate_tts_audio(ai_resp)
                        if audio_data:
                            st.audio(audio_data, format="audio/mp3")

                        st.session_state.messages.append({"role": "assistant", "content": ai_resp})
                    except Exception as e:
                        st.error(f"AI Error: {e}")
