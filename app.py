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

# Page configuration with wide layout to utilize 80%+ of available space
st.set_page_config(
    page_title="SHS Computing AI Tutor", page_icon="💻", layout="wide"
)

# Inject custom CSS to maximize width and responsiveness across all modes
st.markdown("""
    <style>
    .block-container {
        max-width: 85% !important;
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .stChatInput {
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
        initial_greeting += f"\n\n🎨 **Whiteboard Studio Ready:** Explore comprehensive animated computing and ICT concepts with tailored SVG visualizations, external curriculum embeds, synchronized audio explanations, and live responsive word highlighting!"

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
# 🎨 WHITEBOARD CONCEPT STUDIO (DEDICATED SUBTOPIC ANIMATIONS, EMBEDDED REFERENCES, LIMITATION HANDLING)
# ==========================================================
if learning_mode == "🎨 Whiteboard Concept Studio":
    st.markdown("### 🎨 Sir O.K Animated Whiteboard Studio")

    curr_hierarchy = get_curriculum_hierarchy()
    subject_topics_dict = curr_hierarchy.get(selected_subject, {
        "General Concepts": ["Introduction and Fundamental Principles"]
    })

    # Dependent cascading dropdowns: Subtopics depend entirely on the selected topic under the current subject
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        chosen_topic = st.selectbox("Select Topic", list(subject_topics_dict.keys()), key="wb_topic_select")
    with col_t2:
        available_subtopics = subject_topics_dict.get(chosen_topic, ["General Overview"])
        chosen_subtopic = st.selectbox("Select Subtopic", available_subtopics, key="wb_subtopic_select")

    # Function to generate tailored non-overlapping animations, external iframe embed references, or explicit limitation notices
    def get_whiteboard_content(subject, topic, subtopic):
        svg = ""
        text = ""
        is_iframe = False
        iframe_url = ""
        is_unsupported = False

        # Specific tailored content per subtopic to avoid cross-sitting/duplication
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
                elif "Ring" in subtopic:
                    svg = """
                        <circle cx="300" cy="120" r="70" fill="none" stroke="#00ffcc" stroke-width="3" stroke-dasharray="6"/>
                        <g transform="translate(300, 50)"><rect x="-25" y="-12" width="50" height="24" fill="#222" stroke="#ffbb00" stroke-width="2"/><text x="0" y="3" fill="#ffbb00" font-size="8" text-anchor="middle">Node A</text></g>
                        <g transform="translate(370, 120)"><rect x="-25" y="-12" width="50" height="24" fill="#222" stroke="#ffbb00" stroke-width="2"/><text x="0" y="3" fill="#ffbb00" font-size="8" text-anchor="middle">Node B</text></g>
                        <g transform="translate(300, 190)"><rect x="-25" y="-12" width="50" height="24" fill="#222" stroke="#ffbb00" stroke-width="2"/><text x="0" y="3" fill="#ffbb00" font-size="8" text-anchor="middle">Node C</text></g>
                        <g transform="translate(230, 120)"><rect x="-25" y="-12" width="50" height="24" fill="#222" stroke="#ffbb00" stroke-width="2"/><text x="0" y="3" fill="#ffbb00" font-size="8" text-anchor="middle">Node D</text></g>
                        <circle r="5" fill="#ff0055"><animateMotion path="M 300,50 A 70,70 0 1,1 299,50" dur="8s" repeatCount="indefinite"/></circle>
                    """
                    text = "In a Ring Network Topology, devices are connected in a circular loop configuration. Data packets circulate in one specific direction from node to node until reaching the destination address."
                elif "Mesh" in subtopic:
                    svg = """
                        <polygon points="200,60 400,60 500,160 300,210 100,160" fill="none" stroke="#555" stroke-width="2"/>
                        <line x1="200" y1="60" x2="300" y2="210" stroke="#00ffcc" stroke-width="2"/>
                        <line x1="400" y1="60" x2="300" y2="210" stroke="#00ffcc" stroke-width="2"/>
                        <line x1="100" y1="160" x2="400" y2="60" stroke="#00ffcc" stroke-width="2"/>
                        <line x1="500" y1="160" x2="200" y2="60" stroke="#00ffcc" stroke-width="2"/>
                        <circle cx="200" cy="60" r="16" fill="#222" stroke="#ffbb00" stroke-width="2"/><text x="200" y="64" fill="#ffbb00" font-size="8" text-anchor="middle">N1</text>
                        <circle cx="400" cy="60" r="16" fill="#222" stroke="#ffbb00" stroke-width="2"/><text x="400" y="64" fill="#ffbb00" font-size="8" text-anchor="middle">N2</text>
                        <circle cx="500" cy="160" r="16" fill="#222" stroke="#ffbb00" stroke-width="2"/><text x="500" y="164" fill="#ffbb00" font-size="8" text-anchor="middle">N3</text>
                        <circle cx="300" cy="210" r="16" fill="#222" stroke="#ffbb00" stroke-width="2"/><text x="300" y="214" fill="#ffbb00" font-size="8" text-anchor="middle">N4</text>
                        <circle cx="100" cy="160" r="16" fill="#222" stroke="#ffbb00" stroke-width="2"/><text x="100" y="164" fill="#ffbb00" font-size="8" text-anchor="middle">N5</text>
                    """
                    text = "A Mesh Network Topology features redundant point-to-point connections between every node or across multiple nodes, offering supreme fault tolerance and path redundancy."
                else:
                    svg = """
                        <rect x="150" y="90" width="300" height="80" rx="8" fill="#1e1e1e" stroke="#00ffcc" stroke-width="2"/>
                        <text x="300" y="125" fill="#00ffcc" font-size="11" font-weight="bold" text-anchor="middle">COMPLEX HYBRID TOPOLOGY</text>
                        <text x="300" y="145" fill="#aaa" font-size="9" text-anchor="middle">Combining Star, Bus, and Ring Architectures</text>
                    """
                    text = "Hybrid Topologies combine two or more distinct network structures, such as star-ring or star-bus networks, accommodating large-scale institutional computing requirements."
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
                elif "JOIN" in subtopic:
                    svg = """
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
                    text = "SQL JOIN operations combine records from two or more tables based on a related column between them, facilitating complex relational database queries."
                elif "Normalization" in subtopic:
                    svg = """
                        <rect x="150" y="80" width="300" height="90" rx="8" fill="#1e1e1e" stroke="#ff0055" stroke-width="2"/>
                        <text x="300" y="110" fill="#ff0055" font-size="10" font-weight="bold" text-anchor="middle">DATABASE NORMALIZATION</text>
                        <text x="300" y="135" fill="#fff" font-size="9" text-anchor="middle">1NF ➔ Remove Repeating Groups</text>
                        <text x="300" y="155" fill="#fff" font-size="9" text-anchor="middle">2NF &amp; 3NF ➔ Eliminate Partial &amp; Transitive Dependencies</text>
                    """
                    text = "Database Normalization systematically structures tables to minimize data redundancy and dependency anomalies across First, Second, and Third Normal Forms (1NF, 2NF, 3NF)."
                else:
                    is_iframe = True
                    iframe_url = "https://www.youtube.com/embed/HWD904aX5bs"
                    text = "Exploring core relational database constraints, primary keys, foreign keys, and RDBMS architectural properties according to WAEC computing standards."
            elif "Programming & Algorithms" in topic:
                if "Flowcharts" in subtopic:
                    svg = """
                        <ellipse cx="300" cy="35" rx="55" ry="18" fill="#222" stroke="#00ffcc" stroke-width="2"/><text x="300" y="38" fill="#00ffcc" font-size="8" text-anchor="middle">Start Program</text>
                        <line x1="300" y1="53" x2="300" y2="80" stroke="#fff" stroke-width="2"/>
                        <polygon points="300,80 370,120 300,160 230,120" fill="#222" stroke="#ffbb00" stroke-width="2"/><text x="300" y="123" fill="#ffbb00" font-size="8" text-anchor="middle">Score &gt;= 50?</text>
                        <line x1="370" y1="120" x2="450" y2="120" stroke="#fff" stroke-width="2"/><text x="410" y="112" fill="#00ffcc" font-size="8">YES</text>
                        <rect x="450" y="100" width="100" height="40" rx="4" fill="#222" stroke="#00ffcc" stroke-width="2"/><text x="500" y="124" fill="#00ffcc" font-size="8" text-anchor="middle">Pass Grade</text>
                        <circle r="5" fill="#ff0055"><animateMotion path="M 300,35 L 300,80 L 370,120 L 450,120" dur="6s" repeatCount="indefinite"/></circle>
                    """
                    text = "Flowcharts use standardized geometric symbols connected by arrows to illustrate algorithm logic, decision branching, and execution flow prior to writing actual source code."
                elif "Control Structures" in subtopic:
                    svg = """
                        <rect x="150" y="70" width="300" height="100" rx="8" fill="#1e1e1e" stroke="#00ffcc" stroke-width="2"/>
                        <text x="300" y="100" fill="#00ffcc" font-size="10" font-weight="bold" text-anchor="middle">CONTROL STRUCTURES</text>
                        <text x="300" y="125" fill="#fff" font-size="9" text-anchor="middle">Conditional: if / else / elif branching</text>
                        <text x="300" y="145" fill="#fff" font-size="9" text-anchor="middle">Iterative: for loops &amp; while loops</text>
                    """
                    text = "Control structures dictate the order in which individual instructions or code blocks are evaluated and executed within programs."
                else:
                    is_unsupported = True
                    text = "Advanced data structures and object-oriented programming principles for this specific subtopic do not have sufficient internal curriculum dataset items for live SVG rendering. Refer to your WAEC syllabus textbook for detailed implementation guidelines."
            else:
                is_iframe = True
                iframe_url = "https://www.youtube.com/embed/9Q6isjw02Us"
                text = "Cybersecurity fundamentals, data privacy, encryption, and defense against malicious software threats in digital environments."
        elif subject == "ICT":
            if "Computer Architecture" in topic:
                if "CPU" in subtopic:
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
                elif "Memory Hierarchy" in subtopic:
                    svg = """
                        <polygon points="300,50 450,190 150,190" fill="#1e1e1e" stroke="#ffbb00" stroke-width="2"/>
                        <text x="300" y="90" fill="#ffbb00" font-size="9" font-weight="bold" text-anchor="middle">CPU Registers &amp; Cache</text>
                        <text x="300" y="130" fill="#00ffcc" font-size="9" font-weight="bold" text-anchor="middle">Main Memory (RAM)</text>
                        <text x="300" y="170" fill="#ff0055" font-size="9" font-weight="bold" text-anchor="middle">Secondary Storage (SSD/HDD)</text>
                    """
                    text = "The Memory Hierarchy balances speed, capacity, and cost, ranging from lightning-fast processor registers and cache down to high-capacity secondary storage drives."
                else:
                    is_iframe = True
                    iframe_url = "https://www.youtube.com/embed/GcDshWEDDHM"
                    text = "Exploring computer hardware peripherals, logic gates, and boolean algebra principles for WAEC ICT."
            elif "Operating Systems & Software" in topic:
                svg = """
                    <rect x="150" y="75" width="300" height="90" rx="8" fill="#1e1e1e" stroke="#00ffcc" stroke-width="2"/>
                    <text x="300" y="105" fill="#00ffcc" font-size="11" font-weight="bold" text-anchor="middle">OPERATING SYSTEM KERNEL</text>
                    <text x="300" y="130" fill="#fff" font-size="9" text-anchor="middle">Process Management &amp; CPU Scheduling</text>
                    <text x="300" y="150" fill="#fff" font-size="9" text-anchor="middle">File Systems &amp; Memory Allocation</text>
                """
                text = "The operating system manages computer hardware resources, provides common services for application software, and controls process scheduling."
            else:
                is_unsupported = True
                text = "Specific web technology and productivity subtopics under ICT currently lack sufficient internal dataset instructional blocks for custom animation generation. Please consult the official GES ICT syllabus manual."
        else:
            if "Sensors & Actuators" in topic:
                svg = """
                    <rect x="80" y="90" width="120" height="60" rx="6" fill="#222" stroke="#00ffcc" stroke-width="2"/>
                    <text x="140" y="125" fill="#00ffcc" font-size="9" font-weight="bold" text-anchor="middle">ULTRASONIC SENSOR</text>
                    <line x1="205" y1="120" x2="395" y2="120" stroke="#fff" stroke-width="2" stroke-dasharray="4"/>
                    <rect x="400" y="90" width="120" height="60" rx="6" fill="#222" stroke="#ffbb00" stroke-width="2"/>
                    <text x="460" y="125" fill="#ffbb00" font-size="9" font-weight="bold" text-anchor="middle">ARDUINO / ESP32</text>
                    <circle r="5" fill="#ff0055"><animateMotion path="M 205,120 L 395,120" dur="5s" repeatCount="indefinite"/></circle>
                """
                text = "Robotic sensors collect environmental data such as distance, light, or proximity, transmitting signals to microcontrollers for automated response and feedback control."
            else:
                is_unsupported = True
                text = "Advanced robotics kinematics and AI automation subtopics do not have sufficient local training text entries to construct a standalone animation. Refer to industrial robotics engineering textbooks."

        return svg, text, is_iframe, iframe_url, is_unsupported

    current_svg, current_text, is_iframe, iframe_url, is_unsupported = get_whiteboard_content(selected_subject, chosen_topic, chosen_subtopic)

    player_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
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
            padding: 16px;
            box-shadow: 0 4px 20px rgba(0,255,204,0.2);
            width: 100%;
            box-sizing: border-box;
        }}
        .player-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            border-bottom: 1px solid #333;
            padding-bottom: 8px;
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
            border-radius: 4px;
            font-size: 0.8em;
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
        .unsupported-box {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 20px;
            text-align: center;
            background: #151515;
            color: #ffbb00;
            font-size: 0.95em;
            box-sizing: border-box;
        }}
        .controls-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 12px;
            background: #252525;
            padding: 10px 14px;
            border-radius: 6px;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .btn-group {{
            display: flex;
            gap: 8px;
            align-items: center;
            width: 100%;
            justify-content: space-between;
        }}
        button {{
            background: #333;
            color: #fff;
            border: 1px solid #555;
            padding: 8px 14px;
            border-radius: 4px;
            font-weight: bold;
            cursor: pointer;
            font-size: 0.8em;
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
            margin-top: 10px;
            background: #181818;
            border-left: 3px solid #00ffcc;
            padding: 10px 14px;
            border-radius: 4px;
            font-size: 0.9em;
            line-height: 1.5;
            color: #ddd;
            max-height: 80px;
            overflow-y: auto;
            box-sizing: border-box;
        }}
        .transcript-title {{
            font-weight: bold;
            color: #00ffcc;
            margin-bottom: 3px;
            font-size: 0.8em;
            text-transform: uppercase;
        }}
        .highlighted-word {{
            background: #00ffcc;
            color: #000;
            padding: 0 3px;
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
            <span class="brand">📺 SIR O.K. STUDIO</span>
            <span class="topic-badge">{chosen_topic} &gt; {chosen_subtopic}</span>
        </div>
        
        <div class="screen" id="screenContainer">
            {"<iframe src='" + iframe_url + "' allowfullscreen></iframe>" if is_iframe else ("<div class='unsupported-box'>⚠️ " + current_text + "</div>" if is_unsupported else "<svg id='wbSvg' viewBox='0 0 600 240'>" + current_svg + "</svg>")}
        </div>

        <div class="controls-bar">
            <div class="btn-group">
                <button id="playBtn" onclick="togglePlayPause()">⏸️ Pause Audio</button>
                <button onclick="restartAudio()">🔄 Restart Explanation</button>
            </div>
        </div>

        {"<!--" if is_unsupported else ""}
        <div class="transcript-box" id="transcriptBox">
            <div class="transcript-title">🎙️ Synchronized Audio Explanation &amp; Word Highlighting</div>
            <div id="liveCaptionText"></div>
        </div>
        {"-->" if is_unsupported else ""}
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
            if ({str(is_unsupported).lower()}) return;
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

# Export and Print controls side-by-side if report card / revision guide exists
if st.session_state.last_revision_guide:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Report Card & Study Tools")
    
    # Report Card PDF generator helper function using HTML/ReportLab simulation or direct HTML download
    def generate_report_card_html():
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>Sir O.K. Report Card - {student_full_name}</title></head>
        <body style="font-family: Arial; padding: 30px; color: #333;">
            <h1 style="color: #0055ff;">SIR O.K. COMPUTING ACADEMY</h1>
            <h2>Official Student Academic Report Card &amp; Revision Guide</h2>
            <hr/>
            <p><b>Student Name:</b> {student_full_name}</p>
            <p><b>School:</b> {student_school}</p>
            <p><b>Subject Area:</b> {selected_subject}</p>
            <p><b>Date:</b> September 2026</p>
            <hr/>
            <h3>Performance Summary</h3>
            <pre style="background: #f4f4f4; padding: 15px; border-radius: 5px;">{st.session_state.last_revision_guide}</pre>
        </body>
        </html>
        """

    col_exp1, col_exp2 = st.sidebar.columns(2)
    with col_exp1:
        st.download_button(
            label="📥 Download PDF",
            data=st.session_state.last_revision_guide,
            file_name=f"SirOK_{selected_subject}_ReportCard.txt",
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
                                    f"SIR O.K AI TUTOR - OFFICIAL REPORT CARD & REVISION GUIDE\n"
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
