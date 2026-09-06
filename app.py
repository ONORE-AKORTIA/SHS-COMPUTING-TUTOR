import base64
import os
import random
import json
import datetime
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from groq import Groq

# Centralized model configuration
ACTIVE_MODEL = "openai/gpt-oss-20b"

# Page configuration with wide layout to utilize 80%+ of available space
st.set_page_config(
    page_title="SHS Computing AI Tutor", page_icon="💻", layout="wide"
)

# Inject custom CSS to maximize width, responsiveness across all modes, and eliminate scroll traps
st.markdown("""
    <style>
    .block-container {
        max-width: 95% !important;
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    .stChatInput {
        max-width: 100% !important;
    }
    iframe {
        width: 100% !important;
        max-width: 100% !important;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize Groq client securely using Streamlit Secrets
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("Groq API key not found in Streamlit Secrets!")
        return None
    return Groq(api_key=api_key)


# Available subjects, consolidated datasets, and comprehensive curriculum hierarchy for Whiteboard Studio
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


def get_curriculum_hierarchy():
    return {
        "Computing": {
            "Network Topologies": [
                "Star Network Topology",
                "Bus Network Topology",
                "Ring Network Topology",
                "Mesh Network Topology",
                "Tree Network Topology",
                "Hybrid Network Topology"
            ],
            "Database Systems": [
                "Entity-Relationship (ER) Diagrams & Entities",
                "SQL Database JOINs (INNER, LEFT, RIGHT)",
                "Database Normalization (1NF, 2NF, 3NF)",
                "Primary Keys, Foreign Keys & Constraints",
                "Relational Database Management Systems (RDBMS)"
            ],
            "Programming & Algorithms": [
                "Flowcharts and Pseudocode Logic",
                "Control Structures (Loops and Conditionals)",
                "Arrays, Lists and Data Structures",
                "Object-Oriented Programming Principles"
            ],
            "Cybersecurity & Ethics": [
                "Data Privacy and Confidentiality",
                "Encryption and Decryption Fundamentals",
                "Malware Types and Defense Strategies",
                "Cyber Ethics and Safe Browsing"
            ]
        },
        "ICT": {
            "Computer Architecture": [
                "CPU Fetch-Decode-Execute Cycle",
                "Memory Hierarchy (Cache, RAM, Secondary Storage)",
                "Logic Gates & Boolean Algebra",
                "Input, Output and Storage Peripherals"
            ],
            "Operating Systems & Software": [
                "Process Management & CPU Scheduling",
                "File Systems and Directory Structures",
                "System Security & User Access Controls",
                "Application Software vs System Software"
            ],
            "Web Technologies & Networking": [
                "The Internet and World Wide Web Architecture",
                "HTML, CSS and Client-Side Scripting",
                "IP Addressing, DNS and Packet Routing",
                "Network Protocols (TCP/IP, HTTP, FTP)"
            ],
            "Information Systems & Productivity": [
                "Spreadsheets and Data Analysis Tools",
                "Word Processing and Presentation Software",
                "Database Management and Information Retrieval",
                "Impact of ICT in Society and E-Commerce"
            ]
        },
        "Robotics": {
            "Sensors & Actuators": [
                "Ultrasonic, Infrared and Proximity Sensors",
                "Servo, Stepper and DC Motors Control",
                "Feedback Control Loops and PID Controllers",
                "Analog vs Digital Sensor Interfacing"
            ],
            "Kinematics & Microcontrollers": [
                "Forward and Inverse Robot Kinematics",
                "Microcontroller Architecture (Arduino, ESP32)",
                "PWM Signal Modulation and Motor Drivers",
                "Embedded Systems Programming in C/C++"
            ],
            "Automation & AI in Robotics": [
                "Autonomous Navigation and Obstacle Avoidance",
                "Computer Vision for Robotics",
                "Machine Learning in Robotic Systems",
                "Industrial Automation and Safety Protocols"
            ]
        }
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
        st.session_state.exam_question_logs = []
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

# Initialize session state stores & session time logs
if "user_sessions" not in st.session_state:
    st.session_state.user_sessions = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "greeted" not in st.session_state:
    st.session_state.greeted = False

# Session time tracking variables
if "session_login_time" not in st.session_state:
    st.session_state.session_login_time = datetime.datetime.now()
if "exam_question_logs" not in st.session_state:
    st.session_state.exam_question_logs = []

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
        f" I am Sir OK, your AI tutor ready to help you master {selected_subject}"
        " for WAEC. How can I assist you today?"
    )
    if learning_mode == "📝 WAEC Exam Practice":
        initial_greeting += f"\n\n👉 **Exam Practice Ready:** Please type your desired topic and number of questions below (e.g., *'Networking, 2 questions'*)."
    elif learning_mode == "🎨 Whiteboard Concept Studio":
        initial_greeting += f"\n\n🎨 **Whiteboard Studio Ready:** Explore comprehensive animated computing and ICT concepts with tailored SVG visualizations, curated YouTube video explanations under 121 seconds, synchronized audio, and live responsive word highlighting!"

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
# 🎨 WHITEBOARD CONCEPT STUDIO (FULLY RESPONSIVE & LIVE AUDIO-TO-TEXT TRANSCRIPTION)
# ==========================================================
if learning_mode == "🎨 Whiteboard Concept Studio":
    curr_hierarchy = get_curriculum_hierarchy()
    subject_topics_dict = curr_hierarchy.get(selected_subject, {
        "General Concepts": ["Introduction and Fundamental Principles"]
    })

    # Main layout split: Left viewport container (78%), Right custom control sidebar (22%)
    col_main_vp, col_side_ctrl = st.columns([78, 22])

    with col_side_ctrl:
        st.markdown("### 🎛️ Studio Controls")
        chosen_topic = st.selectbox("Select Topic", list(subject_topics_dict.keys()), key="wb_topic_select")
        available_subtopics = subject_topics_dict.get(chosen_topic, ["General Overview"])
        chosen_subtopic = st.selectbox("Select Subtopic", available_subtopics, key="wb_subtopic_select")

    # Function providing tailored animations or curated YouTube video IDs under 121 seconds (<121s)
    def get_whiteboard_content(subject, topic, subtopic):
        svg = ""
        text = ""
        use_video_fallback = False
        yt_video_id = ""

        if subject == "Computing":
            if "Network Topologies" in topic:
                if "Star" in subtopic:
                    svg = """
                        <circle cx="300" cy="130" r="30" fill="#1a1a1a" stroke="#00ffcc" stroke-width="3" />
                        <text x="300" y="126" fill="#00ffcc" font-size="8" font-weight="bold" text-anchor="middle">CENTRAL</text>
                        <text x="300" y="138" fill="#00ffcc" font-size="8" font-weight="bold" text-anchor="middle">SWITCH</text>
                        <line x1="300" y1="130" x2="100" y2="55" stroke="#555" stroke-width="2" stroke-dasharray="4"/>
                        <line x1="300" y1="130" x2="500" y2="55" stroke="#555" stroke-width="2" stroke-dasharray="4"/>
                        <line x1="300" y1="130" x2="100" y2="205" stroke="#555" stroke-width="2" stroke-dasharray="4"/>
                        <line x1="300" y1="130" x2="500" y2="205" stroke="#555" stroke-width="2" stroke-dasharray="4"/>
                        <circle r="6" fill="#ff0055"><animateMotion path="M 300,130 L 100,55" dur="8s" repeatCount="indefinite"/></circle>
                        <g transform="translate(100, 55)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">PC 1</text></g>
                        <g transform="translate(500, 55)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">PC 2</text></g>
                        <g transform="translate(100, 205)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">PC 3</text></g>
                        <g transform="translate(500, 205)"><rect x="-22" y="-14" width="44" height="28" rx="4" fill="#2a2a2a" stroke="#fff" stroke-width="2"/><text x="0" y="4" fill="#fff" font-size="8" font-weight="bold" text-anchor="middle">PC 4</text></g>
                    """
                    text = "In a Star Network Topology, every node device connects directly to a central hub or switch. If one cable fails, only that workstation is disconnected, ensuring high reliability across WAEC network standards."
                elif "Bus" in subtopic:
                    svg = """
                        <line x1="50" y1="120" x2="550" y2="120" stroke="#ffbb00" stroke-width="6"/>
                        <text x="300" y="105" fill="#ffbb00" font-size="10" font-weight="bold" text-anchor="middle">CENTRAL BACKBONE CABLE (BUS)</text>
                        <line x1="150" y1="120" x2="150" y2="60" stroke="#fff" stroke-width="2"/><rect x="125" y="30" width="50" height="30" rx="4" fill="#222" stroke="#00ffcc" stroke-width="2"/><text x="150" y="48" fill="#00ffcc" font-size="8" text-anchor="middle">Node 1</text>
                        <line x1="300" y1="120" x2="300" y2="60" stroke="#fff" stroke-width="2"/><rect x="275" y="30" width="50" height="30" rx="4" fill="#222" stroke="#00ffcc" stroke-width="2"/><text x="300" y="48" fill="#00ffcc" font-size="8" text-anchor="middle">Node 2</text>
                        <line x1="450" y1="120" x2="450" y2="60" stroke="#fff" stroke-width="2"/><rect x="425" y="30" width="50" height="30" rx="4" fill="#222" stroke="#00ffcc" stroke-width="2"/><text x="450" y="48" fill="#00ffcc" font-size="8" text-anchor="middle">Node 3</text>
                        <circle r="5" fill="#ff0055"><animateMotion path="M 150,120 L 450,120" dur="6s" repeatCount="indefinite"/></circle>
                    """
                    text = "The Bus Network Topology utilizes a single shared communication line known as the backbone. Data broadcasted travels along the entire cable length until it reaches the intended recipient workstation."
                else:
                    use_video_fallback = True
                    yt_video_id = "fZW_qA8c3Gg"  # Curated computing video <121s
                    text = "Curated WAEC educational video explaining network topology architectural layouts under 121 seconds with live audio-to-text transcript sync."
            elif "Database Systems" in topic:
                if "Entity-Relationship" in subtopic:
                    svg = """
                        <rect x="60" y="80" width="120" height="70" rx="4" fill="#1e1e1e" stroke="#00ffcc" stroke-width="2"/>
                        <text x="120" y="112" fill="#00ffcc" font-size="11" font-weight="bold" text-anchor="middle">STUDENT</text>
                        <text x="120" y="130" fill="#aaa" font-size="8" text-anchor="middle">Entity Type</text>
                        <ellipse cx="60" cy="30" rx="35" ry="16" fill="#1a1a1a" stroke="#ffbb00" stroke-width="1.5"/>
                        <text x="60" y="33" fill="#ffbb00" font-size="8" text-anchor="middle">StudentID (PK)</text>
                        <line x1="60" y1="46" x2="90" y2="80" stroke="#555" stroke-width="1.5"/>
                        <polygon points="260,115 310,85 360,115 310,145" fill="#1e1e1e" stroke="#ff0055" stroke-width="2"/>
                        <text x="310" y="118" fill="#ff0055" font-size="9" font-weight="bold" text-anchor="middle">ENROLLS</text>
                        <line x1="180" y1="115" x2="260" y2="115" stroke="#fff" stroke-width="2"/><text x="210" y="105" fill="#fff" font-size="8">1</text>
                        <rect x="420" y="80" width="120" height="70" rx="4" fill="#1e1e1e" stroke="#00ffcc" stroke-width="2"/>
                        <text x="480" y="112" fill="#00ffcc" font-size="11" font-weight="bold" text-anchor="middle">COURSE</text>
                        <text x="480" y="130" fill="#aaa" font-size="8" text-anchor="middle">Entity Type</text>
                        <line x1="360" y1="115" x2="420" y2="115" stroke="#fff" stroke-width="2"/><text x="390" y="105" fill="#fff" font-size="8">N</text>
                        <circle r="5" fill="#00ffcc"><animateMotion path="M 120,115 L 310,115 L 480,115" dur="8s" repeatCount="indefinite"/></circle>
                    """
                    text = "Entity-Relationship (ER) Diagrams visually map database architecture. Rectangles represent entity types like Student and Course, diamonds indicate relationships, and ellipses define attributes."
                else:
                    use_video_fallback = True
                    yt_video_id = "HWD904aX5bs"  # Curated database video <121s
                    text = "Database normalization and SQL constraints explained via curated WAEC instructional video with live audio transcription."
            else:
                use_video_fallback = True
                yt_video_id = "9Q6isjw02Us"  # Cybersecurity overview <121s
                text = "Cybersecurity fundamentals and data protection principles explained in under 121 seconds with live speech transcript."
        elif subject == "ICT":
            if "Computer Architecture" in topic:
                svg = """
                    <rect x="60" y="90" width="100" height="80" rx="6" fill="#222" stroke="#00ffcc" stroke-width="2"/>
                    <text x="110" y="135" fill="#00ffcc" font-size="10" font-weight="bold" text-anchor="middle">MEMORY</text>
                    <rect x="240" y="90" width="120" height="80" rx="6" fill="#222" stroke="#ffbb00" stroke-width="2"/>
                    <text x="300" y="125" fill="#ffbb00" font-size="9" font-weight="bold" text-anchor="middle">CONTROL</text>
                    <text x="300" y="140" fill="#ffbb00" font-size="9" font-weight="bold" text-anchor="middle">UNIT (CU)</text>
                    <rect x="440" y="90" width="100" height="80" rx="6" fill="#222" stroke="#ff0055" stroke-width="2"/>
                    <text x="490" y="135" fill="#ff0055" font-size="10" font-weight="bold" text-anchor="middle">ALU</text>
                    <line x1="165" y1="120" x2="235" y2="120" stroke="#fff" stroke-width="2"/>
                    <line x1="365" y1="120" x2="435" y2="120" stroke="#fff" stroke-width="2"/>
                    <circle r="5" fill="#00ffcc"><animateMotion path="M 165,120 L 235,120 L 365,120 L 435,120" dur="8s" repeatCount="indefinite"/></circle>
                """
                text = "The CPU Fetch-Decode-Execute cycle is the fundamental operational process where instructions are retrieved from memory, decoded by the control unit, and executed by the ALU."
            else:
                use_video_fallback = True
                yt_video_id = "GcDshWEDDHM"  # ICT hardware video <121s
                text = "Curated video explaining ICT hardware components and peripherals under 121 seconds with live speech transcription."
        else:
            use_video_fallback = True
            yt_video_id = "8HYvFejdGSQ"  # Robotics video <121s
            text = "Robotic sensors and microcontrollers explained in under 121 seconds with live active transcription."

        return svg, text, use_video_fallback, yt_video_id

    current_svg, current_text, use_video_fallback, yt_video_id = get_whiteboard_content(selected_subject, chosen_topic, chosen_subtopic)

    with col_main_vp:
        player_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <style>
            * {{ box-sizing: border-box; }}
            body {{
                background-color: #121212;
                color: #ffffff;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 2px;
                width: 100%;
                overflow-x: hidden;
            }}
            .player-container {{
                background: #1e1e1e;
                border: 2px solid #00ffcc;
                border-radius: 8px;
                padding: 12px;
                box-shadow: 0 4px 20px rgba(0,255,204,0.2);
                width: 100%;
                max-width: 100%;
            }}
            .player-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
                border-bottom: 1px solid #333;
                padding-bottom: 6px;
                flex-wrap: wrap;
                gap: 5px;
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
                font-size: 0.75em;
                color: #00ffcc;
                font-weight: bold;
            }}
            .screen {{
                position: relative;
                width: 100%;
                padding-bottom: 38%;
                background: #0a0a0a;
                border-radius: 6px;
                overflow: hidden;
                border: 1px solid #333;
            }}
            .screen svg, .screen iframe {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                border: none;
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
                gap: 6px;
                align-items: center;
                width: 100%;
                justify-content: space-between;
            }}
            button {{
                background: #333;
                color: #fff;
                border: 1px solid #555;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
                cursor: pointer;
                font-size: 0.78em;
                transition: all 0.2s;
                flex-grow: 1;
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
            .transcript-box {{
                margin-top: 8px;
                background: #181818;
                border-left: 3px solid #00ffcc;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 0.85em;
                line-height: 1.4;
                color: #ddd;
                max-height: 75px;
                overflow-y: auto;
                width: 100%;
            }}
            .transcript-title {{
                font-weight: bold;
                color: #00ffcc;
                margin-bottom: 2px;
                font-size: 0.75em;
                text-transform: uppercase;
            }}
            .highlighted-word {{
                background: #00ffcc;
                color: #000;
                padding: 0 2px;
                border-radius: 3px;
                font-weight: bold;
            }}
            .past-word {{ color: #888; }}
            .future-word {{ color: #ddd; }}
        </style>
        </head>
        <body>
        <div class="player-container">
            <div class="player-header">
                <span class="brand">📺 SIR OK STUDIO ({ "YOUTUBE LIVE TRANSCRIPTION" if use_video_fallback else "WHITEBOARD ANIMATION" })</span>
                <span class="topic-badge">{chosen_topic} &gt; {chosen_subtopic}</span>
            </div>
            
            <div class="screen" id="screenContainer">
                {"<iframe src='https://www.youtube.com/embed/" + yt_video_id + "?rel=0&autoplay=1' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture' allowfullscreen></iframe>" if use_video_fallback else "<svg id='wbSvg' viewBox='0 0 600 240'>" + current_svg + "</svg>"}
            </div>

            <div class="controls-bar">
                <div class="btn-group">
                    <button id="playBtn" onclick="togglePlayPause()">⏸️ Pause Audio</button>
                    <button onclick="restartAudio()">🔄 Restart Explanation</button>
                </div>
            </div>
            <div class="transcript-box" id="transcriptBox">
                <div class="transcript-title">🎙️ {"Live YouTube Audio Transcription" if use_video_fallback else "Synchronized Audio Explanation & Word Highlighting"}</div>
                <div id="liveCaptionText"></div>
            </div>
        </div>

        <script>
            const fullText = "{current_text}";
            let words = fullText.split(/\\s+/);
            let currentWordIndex = 0;
            let isPlaying = true;
            let utterance = null;
            const playBtn = document.getElementById('playBtn');
            const liveCaptionText = document.getElementById('liveCaptionText');

            function renderTranscript() {{
                if (!liveCaptionText) return;
                liveCaptionText.innerHTML = words.map((w, i) => {{
                    let cls = 'future-word';
                    if (i === currentWordIndex) cls = 'highlighted-word active-word';
                    else if (i < currentWordIndex) cls = 'past-word';
                    return `<span class="${{cls}}">${{w}}</span>`;
                }}).join(' ');
            }}

            function playSpeech() {{
                if (!('speechSynthesis' in window)) return;
                window.speechSynthesis.cancel();

                const remainingWords = words.slice(currentWordIndex);
                if (remainingWords.length === 0) return;

                utterance = new SpeechSynthesisUtterance(remainingWords.join(' '));
                utterance.rate = 0.90;
                window.chunkStartIndex = currentWordIndex;

                utterance.onboundary = function(event) {{
                    if (event.name === 'word') {{
                        const textUpToChar = event.target.text.substring(0, event.charIndex);
                        const wordsBeforeChar = textUpToChar.trim() === '' ? 0 : textUpToChar.trim().split(/\\s+/).length;
                        currentWordIndex = window.chunkStartIndex + wordsBeforeChar;
                        if (currentWordIndex >= words.length) currentWordIndex = words.length - 1;
                        renderTranscript();
                    }}
                }};

                utterance.onend = function() {{
                    isPlaying = false;
                    if (playBtn) playBtn.innerHTML = '▶️ Play Audio';
                }};

                window.speechSynthesis.speak(utterance);
            }}

            function togglePlayPause() {{
                isPlaying = !isPlaying;
                if (isPlaying) {{
                    if (playBtn) playBtn.innerHTML = '⏸️ Pause Audio';
                    playSpeech();
                }} else {{
                    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
                    if (playBtn) playBtn.innerHTML = '▶️ Play Audio';
                }}
            }}

            function restartAudio() {{
                if ('speechSynthesis' in window) window.speechSynthesis.cancel();
                currentWordIndex = 0;
                renderTranscript();
                isPlaying = true;
                if (playBtn) playBtn.innerHTML = '⏸️ Pause Audio';
                playSpeech();
            }}

            window.addEventListener('load', () => {{
                renderTranscript();
                playSpeech();
            }});
        </script>
        </body>
        </html>
        """

        components.html(player_html, height=480, scrolling=False)

else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Export and Print controls side-by-side if report card / revision guide exists (WAEC Exams Mode Only)
if learning_mode == "📝 WAEC Exam Practice" and st.session_state.last_revision_guide:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 WAEC Report Card & Study Tools")
    
    col_exp1, col_exp2 = st.sidebar.columns(2)
    with col_exp1:
        st.download_button(
            label="📥 Download PDF",
            data=st.session_state.last_revision_guide,
            file_name=f"SirOK_{selected_subject}_WAEC_ReportCard.txt",
            mime="text/plain",
        )
    with col_exp2:
        if st.button("🖨️ Print Report"):
            st.toast("Report ready for printing! Use your browser print menu (Ctrl+P).")

user_query = None

# Bottom chat input available in all modes
if input_method == "⌨️ Type Question":
    if learning_mode == "🎨 Whiteboard Concept Studio":
        prompt_label = "Or type topic and subtopic separated by space or comma:"
    elif learning_mode == "📝 WAEC Exam Practice" and st.session_state.exam_state_stage == "awaiting_config":
        prompt_label = "Type your desired topic and number of questions (e.g., 'Databases, 2 questions'):"
    elif learning_mode == "📝 WAEC Exam Practice" and st.session_state.exam_state_stage == "in_progress":
        prompt_label = f"Type your answer for Question {st.session_state.current_question_num} of {st.session_state.total_questions} (A, B, C, or D)..."
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
                            st.session_state.exam_question_logs = []
                            st.session_state.current_correct_option = None
                            st.session_state.topic_performance = {}
                            st.session_state.last_revision_guide = None

                            start_prompt = (
                                f"You are Sir OK, an expert WAEC Examiner in"
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

                            # Log question item details for exportable WAEC report card
                            q_text_summary = st.session_state.asked_questions[-1] if st.session_state.asked_questions else f"Question {current_q}"
                            q_summary_snippet = q_text_summary.split("\n")[0][:80]
                            question_log_entry = {
                                "q_num": current_q,
                                "topic": active_topic,
                                "item_desc": q_summary_snippet,
                                "user_answer": user_ans_clean,
                                "correct_answer": expected_letter,
                                "result": "CORRECT" if is_correct else "INCORRECT",
                                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            st.session_state.exam_question_logs.append(question_log_entry)

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

                            eval_and_next_prompt = (
                                f"You are Sir OK, an expert WAEC Examiner in"
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
                                session_logout_time = datetime.datetime.now()
                                session_duration = session_logout_time - st.session_state.session_login_time
                                duration_mins = int(session_duration.total_seconds() // 60)
                                duration_secs = int(session_duration.total_seconds() % 60)

                                test_items_table = "\n### 📋 Detailed Test Items Breakdown\n"
                                test_items_table += "| Q# | Topic | Item Summary | Your Ans | Correct | Status |\n"
                                test_items_table += "|---|---|---|---|---|---|\n"
                                for log in st.session_state.exam_question_logs:
                                    test_items_table += f"| {log['q_num']} | {log['topic']} | {log['item_desc']} | {log['user_answer']} | {log['correct_answer']} | {log['result']} |\n"

                                evaluation_summary_score = (
                                    f"\n\n### 📊 WAEC Examination & Session Summary\n"
                                    f"- **Student Name:** {student_full_name}\n"
                                    f"- **School:** {student_school}\n"
                                    f"- **Session Date:** {st.session_state.session_login_time.strftime('%Y-%m-%d')}\n"
                                    f"- **Login Time:** {st.session_state.session_login_time.strftime('%H:%M:%S')}\n"
                                    f"- **Logout Time:** {session_logout_time.strftime('%H:%M:%S')}\n"
                                    f"- **Total Time Spent:** {duration_mins} minutes {duration_secs} seconds\n"
                                    f"- **Total Questions:** {total_q}\n"
                                    f"- **Correct Answers:** {st.session_state.correct_count}\n"
                                    f"- **Wrong Answers:** {st.session_state.wrong_count}\n"
                                    f"- **Overall Performance Score:** {(st.session_state.correct_count / total_q) * 100:.1f}%\n"
                                    + test_items_table
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
                                session_logout_time = datetime.datetime.now()
                                session_duration = session_logout_time - st.session_state.session_login_time
                                duration_mins = int(session_duration.total_seconds() // 60)
                                duration_secs = int(session_duration.total_seconds() % 60)

                                report_card_text_table = "DETAILED TEST ITEMS BREAKDOWN:\n"
                                for log in st.session_state.exam_question_logs:
                                    report_card_text_table += f"Q{log['q_num']} | Topic: {log['topic']} | Your Ans: {log['user_answer']} | Correct: {log['correct_answer']} | Status: {log['result']}\n"

                                st.session_state.last_revision_guide = (
                                    f"SIR OK AI TUTOR - OFFICIAL WAEC REPORT CARD & TIME LOG\n"
                                    f"Student: {student_full_name} | School: {student_school}\n"
                                    f"Subject: {selected_subject}\n"
                                    f"Session Date: {st.session_state.session_login_time.strftime('%Y-%m-%d')}\n"
                                    f"Login Time: {st.session_state.session_login_time.strftime('%H:%M:%S')}\n"
                                    f"Logout Time: {session_logout_time.strftime('%H:%M:%S')}\n"
                                    f"Total Time Spent: {duration_mins} mins {duration_secs} secs\n"
                                    f"Overall Score: {(st.session_state.correct_count / total_q) * 100:.1f}% ({st.session_state.correct_count}/{total_q})\n"
                                    f"--------------------------------------------------\n\n"
                                    f"{report_card_text_table}\n\n"
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
                            f"You are Sir OK, an expert SHS AI Tutor helping"
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
