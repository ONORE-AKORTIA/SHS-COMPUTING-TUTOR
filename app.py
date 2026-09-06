import base64
import os
import random
import json
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
        initial_greeting += f"\n\n🎨 **Whiteboard Studio Ready:** Explore comprehensive animated network topologies and computing concepts with synchronized audio and live transcripts!"

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
# 🎨 WHITEBOARD CONCEPT STUDIO (HIERARCHICAL SELECT, 120s PACED SVG, INTEGRATED AUDIO, LIVE TRANSCRIPT)
# ==========================================================
if learning_mode == "🎨 Whiteboard Concept Studio":
    st.markdown("### 🎨 Sir O.K Animated Whiteboard Studio")
    
    # Categorized hierarchy for better organization and subtopic selection
    concept_categories = {
        "🌐 Network Topologies": [
            "Star Network Topology",
            "Bus Network Topology",
            "Ring Network Topology",
            "Mesh Network Topology"
        ],
        "🗄️ Database Systems": [
            "SQL Database JOINs",
            "Database Normalization (1NF, 2NF, 3NF)",
            "Entity-Relationship (ER) Diagrams"
        ],
        "💻 Computer Architecture": [
            "CPU Fetch-Decode-Execute Cycle",
            "Memory Hierarchy (Cache, RAM, Storage)",
            "Logic Gates & Boolean Algebra"
        ]
    }

    if "wb_category" not in st.session_state:
        st.session_state.wb_category = list(concept_categories.keys())[0]
    if "wb_concept" not in st.session_state:
        st.session_state.wb_concept = concept_categories[st.session_state.wb_category][0]

    # Hierarchical Dropdown Selection (Category -> Subtopic)
    col_cat, col_sub = st.columns(2)
    with col_cat:
        selected_category = st.selectbox(
            "Select Topic Category:",
            list(concept_categories.keys()),
            key="wb_category_selectbox"
        )
    
    # If category changed, reset subtopic to first in category
    if selected_category != st.session_state.wb_category:
        st.session_state.wb_category = selected_category
        st.session_state.wb_concept = concept_categories[selected_category][0]

    with col_sub:
        available_subtopics = concept_categories[st.session_state.wb_category]
        current_sub_index = available_subtopics.index(st.session_state.wb_concept) if st.session_state.wb_concept in available_subtopics else 0
        selected_subtopic = st.selectbox(
            "Select Specific Concept (Subtopic):",
            available_subtopics,
            index=current_sub_index,
            key="wb_subtopic_selectbox"
        )
        st.session_state.wb_concept = selected_subtopic

    wb_concept = st.session_state.wb_concept

    # Flatten list for forward/backward navigation
    flat_concept_list = []
    for cat_items in concept_categories.values():
        flat_concept_list.extend(cat_items)
    current_flat_idx = flat_concept_list.index(wb_concept) if wb_concept in flat_concept_list else 0

    # Navigation controls (Previous / Next)
    nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
    with nav_col1:
        if st.button("⏮️ Previous Concept"):
            new_idx = (current_flat_idx - 1) % len(flat_concept_list)
            target_concept = flat_concept_list[new_idx]
            for cat, items in concept_categories.items():
                if target_concept in items:
                    st.session_state.wb_category = cat
                    st.session_state.wb_concept = target_concept
                    break
            st.rerun()
    with nav_col3:
        if st.button("Next Concept ⏭️"):
            new_idx = (current_flat_idx + 1) % len(flat_concept_list)
            target_concept = flat_concept_list[new_idx]
            for cat, items in concept_categories.items():
                if target_concept in items:
                    st.session_state.wb_category = cat
                    st.session_state.wb_concept = target_concept
                    break
            st.rerun()

    # Generate comprehensive audio explanation script via Groq (synthesized for thorough explanation, paced up to 120s max)
    client = get_groq_client()
    default_explanation = (
        f"Welcome to Sir O.K's Whiteboard Concept Studio. Today we are exploring {wb_concept} in detail. "
        f"This foundational concept is critical for mastering your WAEC examinations. "
        f"As illustrated in the whiteboard diagram, data flow and architectural interactions operate synchronously to ensure maximum efficiency, robust error handling, and optimal system performance."
    )
    explanation_text = default_explanation
    if client:
        try:
            comp = client.chat.completions.create(
                model=ACTIVE_MODEL,
                messages=[{
                    "role": "user", 
                    "content": f"Provide a comprehensive, highly detailed 4 to 5 sentence audio narration script explaining {wb_concept} thoroughly for WAEC computing students."
                }],
                max_tokens=250,
            )
            raw_text = comp.choices[0].message.content
            if raw_text and len(raw_text.strip()) > 10:
                explanation_text = raw_text.replace('\n', ' ')
        except Exception:
            pass

    # Build Concept-Specific SVG Graphics (with smooth, longer 15s-20s pacing scaled for comprehensive comprehension)
    svg_content = ""
    if wb_concept == "Star Network Topology":
        svg_content = """
            <circle cx="300" cy="130" r="30" fill="#1a1a1a" stroke="#00ffcc" stroke-width="3" />
            <text x="300" y="126" fill="#00ffcc" font-size="8" font-weight="bold" text-anchor="middle">CENTRAL</text>
            <text x="300" y="138" fill="#00ffcc" font-size="8" font-weight="bold" text-anchor="middle">SWITCH</text>
            
            <line x1="300" y1="130" x2="100" y2="55" stroke="#555" stroke-width="2" stroke-dasharray="4"/>
            <line x1="300" y1="130" x2="500" y2="55" stroke="#555" stroke-width="2" stroke-dasharray="4"/>
            <line x1="300" y1="130" x2="100" y2="205" stroke="#555" stroke-width="2" stroke-dasharray="4"/>
            <line x1="300" y1="130" x2="500" y2="205" stroke="#555" stroke-width="2" stroke-dasharray="4"/>

            <circle r="6" fill="#ff0055"><animateMotion path="M 300,130 L 100,55" dur="12s" repeatCount="indefinite"/></circle>
            <circle r="6" fill="#00ffcc"><animateMotion path="M 300,130 L 500,55" dur="10s" repeatCount="indefinite"/></circle>
            <circle r="6" fill="#ffbb00"><animateMotion path="M 100,205 L 300,130" dur="14s" repeatCount="indefinite"/></circle>
            <circle r="6" fill="#00ffcc"><animateMotion path="M 500,205 L 300,130" dur="11s" repeatCount="indefinite"/></circle>

            <g transform="translate(100, 55)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">PC 1</text></g>
            <g transform="translate(500, 55)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">PC 2</text></g>
            <g transform="translate(100, 205)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">PC 3</text></g>
            <g transform="translate(500, 205)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">PC 4</text></g>
        """
    elif wb_concept == "Bus Network Topology":
        svg_content = """
            <line x1="50" y1="130" x2="550" y2="130" stroke="#00ffcc" stroke-width="6" stroke-linecap="round"/>
            <text x="300" y="115" fill="#00ffcc" font-size="10" font-weight="bold" text-anchor="middle">MAIN BACKBONE CABLE</text>
            
            <line x1="120" y1="130" x2="120" y2="60" stroke="#aaa" stroke-width="2"/>
            <line x1="280" y1="130" x2="280" y2="200" stroke="#aaa" stroke-width="2"/>
            <line x1="420" y1="130" x2="420" y2="60" stroke="#aaa" stroke-width="2"/>

            <circle r="6" fill="#ff0055"><animateMotion path="M 60,130 L 540,130" dur="12s" repeatCount="indefinite"/></circle>

            <g transform="translate(120, 45)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">Node A</text></g>
            <g transform="translate(280, 215)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">Node B</text></g>
            <g transform="translate(420, 45)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">Node C</text></g>
        """
    elif wb_concept == "Ring Network Topology":
        svg_content = """
            <circle cx="300" cy="130" r="75" fill="none" stroke="#00ffcc" stroke-width="3" stroke-dasharray="6,4"/>
            <text x="300" y="125" fill="#00ffcc" font-size="9" font-weight="bold" text-anchor="middle">CLOSED LOOP</text>
            
            <circle r="6" fill="#ffbb00"><animateMotion path="M 300,55 A 75,75 0 1,1 299.9,55" dur="14s" repeatCount="indefinite"/></circle>

            <g transform="translate(300, 50)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">Node 1</text></g>
            <g transform="translate(385, 130)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">Node 2</text></g>
            <g transform="translate(300, 210)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">Node 3</text></g>
            <g transform="translate(215, 130)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">Node 4</text></g>
        """
    elif wb_concept == "Mesh Network Topology":
        svg_content = """
            <line x1="150" y1="70" x2="450" y2="70" stroke="#555" stroke-width="2" stroke-dasharray="3"/>
            <line x1="150" y1="70" x2="300" y2="190" stroke="#555" stroke-width="2" stroke-dasharray="3"/>
            <line x1="450" y1="70" x2="300" y2="190" stroke="#555" stroke-width="2" stroke-dasharray="3"/>
            <line x1="150" y1="70" x2="100" y2="190" stroke="#00ffcc" stroke-width="2"/>
            <line x1="450" y1="70" x2="500" y2="190" stroke="#00ffcc" stroke-width="2"/>
            <line x1="100" y1="190" x2="300" y2="190" stroke="#00ffcc" stroke-width="2"/>
            <line x1="300" y1="190" x2="500" y2="190" stroke="#00ffcc" stroke-width="2"/>

            <circle r="6" fill="#ff0055"><animateMotion path="M 150,70 L 450,70 L 300,190 Z" dur="15s" repeatCount="indefinite"/></circle>

            <g transform="translate(150, 70)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">Node A</text></g>
            <g transform="translate(450, 70)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">Node B</text></g>
            <g transform="translate(100, 190)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">Node C</text></g>
            <g transform="translate(300, 190)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">Node D</text></g>
            <g transform="translate(500, 190)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">Node E</text></g>
        """
    elif wb_concept == "SQL Database JOINs":
        svg_content = """
            <rect x="120" y="70" width="120" height="110" rx="6" fill="#1e1e1e" stroke="#00ffcc" stroke-width="2"/>
            <text x="180" y="95" fill="#00ffcc" font-size="10" font-weight="bold" text-anchor="middle">TABLE A</text>
            <line x1="120" y1="105" x2="240" y2="105" stroke="#00ffcc" stroke-width="1"/>
            <text x="180" y="130" fill="#ddd" font-size="8" text-anchor="middle">ID | Name</text>
            <text x="180" y="150" fill="#ddd" font-size="8" text-anchor="middle">1  | Alice</text>

            <rect x="360" y="70" width="120" height="110" rx="6" fill="#1e1e1e" stroke="#ffbb00" stroke-width="2"/>
            <text x="420" y="95" fill="#ffbb00" font-size="10" font-weight="bold" text-anchor="middle">TABLE B</text>
            <line x1="360" y1="105" x2="480" y2="105" stroke="#ffbb00" stroke-width="1"/>
            <text x="420" y="130" fill="#ddd" font-size="8" text-anchor="middle">ID | Course</text>
            <text x="420" y="150" fill="#ddd" font-size="8" text-anchor="middle">1  | ICT</text>

            <path d="M 245,125 Q 300,90 355,125" fill="none" stroke="#ff0055" stroke-width="3" stroke-dasharray="4"/>
            <text x="300" y="85" fill="#ff0055" font-size="9" font-weight="bold" text-anchor="middle">INNER JOIN</text>
        """
    elif wb_concept == "Database Normalization (1NF, 2NF, 3NF)":
        svg_content = """
            <rect x="80" y="90" width="120" height="80" rx="6" fill="#222" stroke="#ff0055" stroke-width="2"/>
            <text x="140" y="125" fill="#ff0055" font-size="9" font-weight="bold" text-anchor="middle">Unnormalized</text>
            <text x="140" y="140" fill="#ff0055" font-size="9" font-weight="bold" text-anchor="middle">Data (UNF)</text>

            <rect x="240" y="90" width="120" height="80" rx="6" fill="#222" stroke="#ffbb00" stroke-width="2"/>
            <text x="300" y="125" fill="#ffbb00" font-size="9" font-weight="bold" text-anchor="middle">1NF &amp; 2NF</text>
            <text x="300" y="140" fill="#ffbb00" font-size="9" font-weight="bold" text-anchor="middle">Atomicity</text>

            <rect x="400" y="90" width="120" height="80" rx="6" fill="#222" stroke="#00ffcc" stroke-width="2"/>
            <text x="460" y="125" fill="#00ffcc" font-size="9" font-weight="bold" text-anchor="middle">3NF (Third</text>
            <text x="460" y="140" fill="#00ffcc" font-size="9" font-weight="bold" text-anchor="middle">Normal Form)</text>

            <line x1="205" y1="130" x2="235" y2="130" stroke="#fff" stroke-width="2"/>
            <line x1="365" y1="130" x2="395" y2="130" stroke="#fff" stroke-width="2"/>
            <circle r="5" fill="#00ffcc"><animateMotion path="M 80,130 L 240,130 L 400,130" dur="10s" repeatCount="indefinite"/></circle>
        """
    elif wb_concept == "Entity-Relationship (ER) Diagrams":
        svg_content = """
            <rect x="80" y="100" width="100" height="60" rx="6" fill="#222" stroke="#00ffcc" stroke-width="2"/>
            <text x="130" y="135" fill="#00ffcc" font-size="10" font-weight="bold" text-anchor="middle">STUDENT</text>

            <ellipse cx="300" cy="130" rx="55" ry="35" fill="#222" stroke="#ffbb00" stroke-width="2"/>
            <text x="300" y="135" fill="#ffbb00" font-size="9" font-weight="bold" text-anchor="middle">ENROLLS</text>

            <rect x="420" y="100" width="100" height="60" rx="6" fill="#222" stroke="#ff0055" stroke-width="2"/>
            <text x="470" y="135" fill="#ff0055" font-size="10" font-weight="bold" text-anchor="middle">COURSE</text>

            <line x1="185" y1="130" x2="240" y2="130" stroke="#fff" stroke-width="2"/>
            <line x1="355" y1="130" x2="415" y2="130" stroke="#fff" stroke-width="2"/>
            <circle r="5" fill="#ffbb00"><animateMotion path="M 130,130 L 300,130 L 470,130" dur="10s" repeatCount="indefinite"/></circle>
        """
    elif wb_concept == "Memory Hierarchy (Cache, RAM, Storage)":
        svg_content = """
            <polygon points="300,45 420,135 180,135" fill="#222" stroke="#00ffcc" stroke-width="2"/>
            <text x="300" y="80" fill="#00ffcc" font-size="9" font-weight="bold" text-anchor="middle">CPU Registers &amp; Cache</text>
            
            <polygon points="180,140 420,140 470,215 130,215" fill="#1e1e1e" stroke="#ffbb00" stroke-width="2"/>
            <text x="300" y="180" fill="#ffbb00" font-size="9" font-weight="bold" text-anchor="middle">Main Memory (RAM)</text>

            <circle r="6" fill="#ff0055"><animateMotion path="M 300,60 L 300,180" dur="8s" repeatCount="indefinite"/></circle>
        """
    elif wb_concept == "Logic Gates & Boolean Algebra":
        svg_content = """
            <rect x="220" y="90" width="100" height="80" rx="6" fill="#222" stroke="#00ffcc" stroke-width="2"/>
            <text x="270" y="135" fill="#00ffcc" font-size="11" font-weight="bold" text-anchor="middle">AND GATE</text>

            <line x1="120" y1="110" x2="215" y2="110" stroke="#fff" stroke-width="2"/>
            <text x="140" y="105" fill="#aaa" font-size="9">Input A</text>
            <line x1="120" y1="150" x2="215" y2="150" stroke="#fff" stroke-width="2"/>
            <text x="140" y="165" fill="#aaa" font-size="9">Input B</text>

            <line x1="325" y1="130" x2="420" y2="130" stroke="#ffbb00" stroke-width="3"/>
            <text x="350" y="120" fill="#ffbb00" font-size="9">Output Q</text>

            <circle r="5" fill="#00ffcc"><animateMotion path="M 120,110 L 270,130 L 420,130" dur="9s" repeatCount="indefinite"/></circle>
        """
    else:  # CPU Fetch-Decode-Execute Cycle
        svg_content = """
            <rect x="60" y="90" width="100" height="80" rx="6" fill="#222" stroke="#00ffcc" stroke-width="2"/>
            <text x="110" y="135" fill="#00ffcc" font-size="10" font-weight="bold" text-anchor="middle">MEMORY</text>

            <rect x="240" y="90" width="120" height="80" rx="6" fill="#222" stroke="#ffbb00" stroke-width="2"/>
            <text x="300" y="125" fill="#ffbb00" font-size="9" font-weight="bold" text-anchor="middle">CONTROL</text>
            <text x="300" y="140" fill="#ffbb00" font-size="9" font-weight="bold" text-anchor="middle">UNIT (CU)</text>

            <rect x="440" y="90" width="100" height="80" rx="6" fill="#222" stroke="#ff0055" stroke-width="2"/>
            <text x="490" y="135" fill="#ff0055" font-size="10" font-weight="bold" text-anchor="middle">ALU</text>

            <line x1="165" y1="120" x2="235" y2="120" stroke="#fff" stroke-width="2"/>
            <line x1="365" y1="120" x2="435" y2="120" stroke="#fff" stroke-width="2"/>
            <circle r="5" fill="#00ffcc"><animateMotion path="M 165,120 L 235,120 L 365,120 L 435,120" dur="12s" repeatCount="indefinite"/></circle>
        """

    # Safely JSON-encode the explanation text and topic to prevent JS string breaking or audio failure
    safe_explanation_json = json.dumps(explanation_text)
    safe_title_json = json.dumps(wb_concept)

    # Fully Responsive HTML/JS Component with Integrated Play/Pause, Navigation, and Live Responsive Subtitle/Transcript Display Box
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
            padding: 4px;
            box-sizing: border-box;
        }}
        .player-container {{
            background: #1e1e1e;
            border: 2px solid #00ffcc;
            border-radius: 10px;
            padding: 12px;
            box-shadow: 0 4px 15px rgba(0,255,204,0.15);
            width: 100%;
            box-sizing: border-box;
        }}
        .player-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            border-bottom: 1px solid #333;
            padding-bottom: 6px;
        }}
        .brand {{
            color: #00ffcc;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .topic-badge {{
            background: #282828;
            border: 1px solid #00ffcc;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            color: #00ffcc;
            font-weight: bold;
        }}
        .screen {{
            position: relative;
            width: 100%;
            padding-bottom: 42%;
            background: #0a0a0a;
            border-radius: 6px;
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
            border-radius: 6px;
            flex-wrap: wrap;
            gap: 6px;
        }}
        .btn-group {{
            display: flex;
            gap: 8px;
        }}
        button {{
            background: #333;
            color: #fff;
            border: 1px solid #555;
            padding: 6px 14px;
            border-radius: 4px;
            font-weight: bold;
            cursor: pointer;
            font-size: 0.8em;
            transition: all 0.2s;
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
            font-size: 0.75em;
            color: #aaa;
        }}
        .transcript-box {{
            margin-top: 10px;
            background: #181818;
            border-left: 3px solid #00ffcc;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.85em;
            line-height: 1.4;
            color: #ddd;
            min-height: 50px;
            box-sizing: border-box;
        }}
        .transcript-title {{
            font-weight: bold;
            color: #00ffcc;
            margin-bottom: 4px;
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        #liveCaptionText {{
            color: #ffffff;
        }}
    </style>
    </head>
    <body>
    <div class="player-container">
        <div class="player-header">
            <span class="brand">📺 SIR O.K. STUDIO</span>
            <span class="topic-badge" id="displayConceptTitle"></span>
        </div>
        
        <div class="screen">
            <svg id="wbSvg" viewBox="0 0 600 260">
                {svg_content}
            </svg>
        </div>

        <div class="controls-bar">
            <div class="btn-group">
                <button id="playBtn" onclick="togglePlayPause()">⏸️ Pause Audio &amp; Animation</button>
                <button onclick="restartExperience()">🔄 Restart</button>
            </div>
            <div class="status" id="statusText">Playing (Audio Active)</div>
        </div>

        <div class="transcript-box">
            <div class="transcript-title">🎙️ Live Audio Transcript &amp; Narration</div>
            <div id="liveCaptionText"></div>
        </div>
    </div>

    <script>
        const conceptTitle = {safe_title_json};
        const narrationText = {safe_explanation_json};
        
        document.getElementById('displayConceptTitle').innerText = conceptTitle;
        document.getElementById('liveCaptionText').innerText = narrationText;

        let isPlaying = true;
        const svg = document.getElementById('wbSvg');
        let utterance = null;

        function initSpeech() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                utterance = new SpeechSynthesisUtterance(narrationText);
                utterance.rate = 0.90; // Balanced pacing for up to 120s comprehensive delivery
                utterance.pitch = 1.0;

                utterance.onboundary = function(event) {{
                    if (event.name === 'word') {{
                        // Highlight or keep transcript active
                    }}
                }};

                utterance.onend = function() {{
                    if (isPlaying) {{
                        document.getElementById('statusText').innerText = 'Completed / Paused';
                    }}
                }};

                window.speechSynthesis.speak(utterance);
            }}
        }}

        function stopSpeech() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
            }}
        }}

        // Autoplay on load
        window.addEventListener('load', () => {{
            setTimeout(initSpeech, 300);
        }});

        function togglePlayPause() {{
            isPlaying = !isPlaying;
            const playBtn = document.getElementById('playBtn');
            const statusText = document.getElementById('statusText');

            if (isPlaying) {{
                try {{ svg.unpauseAnimations(); }} catch(e) {{}}
                playBtn.innerText = '⏸️ Pause Audio &amp; Animation';
                statusText.innerText = 'Playing (Audio Active)';
                initSpeech();
            }} else {{
                try {{ svg.pauseAnimations(); }} catch(e) {{}}
                playBtn.innerText = '▶️ Play Audio &amp; Animation';
                statusText.innerText = 'Paused';
                stopSpeech();
            }}
        }}

        function restartExperience() {{
            try {{ svg.setCurrentTime(0); }} catch(e) {{}}
            isPlaying = true;
            try {{ svg.unpauseAnimations(); }} catch(e) {{}}
            document.getElementById('playBtn').innerText = '⏸️ Pause Audio &amp; Animation';
            document.getElementById('statusText').innerText = 'Playing (Audio Active)';
            initSpeech();
        }}
    </script>
    </body>
    </html>
    """

    components.html(player_html, height=460, scrolling=False)

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
