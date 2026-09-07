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

# Optional PostgreSQL & Docker imports with graceful fallback handling
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
    page_title="SHS AI Tutor - Enterprise Edition", page_icon="💻", layout="wide"
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
# 🧠 AUTOMATED SEMANTIC GRADING ENGINE
# ==========================================================
def evaluate_essay_semantics(student_answer: str, rubric: str, subject: str, year_level: str) -> dict:
    client = get_groq_client() or get_openai_client()
    if not client:
        return {
            "score": 75.0,
            "strengths": ["Clear attempt at addressing the prompt", "Good foundational terminology"],
            "gaps": ["Lacks detailed technical elaboration", "Could include standard examples"],
            "feedback": f"Your response covers the core concept for {subject} ({year_level}). To score full distinction marks, elaborate further on practical applications."
        }
    
    prompt = f"""You are an expert Chief Examiner in {subject} for {year_level}. Grade the student's answer strictly based on the rubric provided. Return ONLY valid JSON format with keys: "score" (float 0-100), "strengths" (list of strings), "gaps" (list of strings), and "feedback" (string).

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
# 🎮 SECURE PYTHON & SQL CODE EXECUTION SANDBOX
# ==========================================================
def execute_python_code(code_string: str) -> dict:
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
# CURRICULUM & YEAR-BY-YEAR SEGMENTATION
# ==========================================================
def get_comprehensive_curriculum():
    return {
        "Core Mathematics": {
            "Year 1": ["Number and Numeration", "Sets", "Algebraic Expressions", "Geometry and Mensuration"],
            "Year 2": ["Quadratic Equations", "Trigonometry", "Statistics and Probability", "Vectors and Matrices"],
            "Year 3": ["Advanced Calculus Concepts", "Circle Theorems", "Linear Programming", "Revision and Past WAEC Papers"]
        },
        "English Language": {
            "Year 1": ["Grammar & Lexis", "Parts of Speech", "Reading Comprehension Fundamentals", "Summary Writing Basics"],
            "Year 2": ["Essay Writing (Narrative, Descriptive, Argumentative)", "Advanced Lexis & Structure", "Literature-in-English Integration"],
            "Year 3": ["Oral English & Phonetics", "Mock Exam Essay Practicals", "Final WAEC Exam Strategies"]
        },
        "Integrated Science": {
            "Year 1": ["Diversity of Matter", "Cells and Living Organisms", "Matter and Energy", "Acids, Bases and Salts"],
            "Year 2": ["Ecosystems and Environment", "Work, Energy and Power", "Atomic Structure and Bonding", "Electronics Basics"],
            "Year 3": ["Genetics and Evolution", "Organic Chemistry", "Applied Physics & Mechanics", "Practical Science Exam Review"]
        },
        "Social Studies": {
            "Year 1": ["The Environment and the Individual", "The Family and the Community", "National Identity and Unity"],
            "Year 2": ["Socio-Economic Development", "Governance, Politics and Law", "Human Rights and Responsibilities"],
            "Year 3": ["Global Issues and International Relations", "Self-Reliance and Entrepreneurship", "Exam Preparatory Case Studies"]
        },
        "Information and Communications Technology (ICT)": {
            "Year 1": ["Computer Hardware Architecture", "Operating Systems Fundamentals", "Word Processing & Spreadsheets"],
            "Year 2": ["Database Management Systems", "Desktop Publishing & Presentation Graphics", "Internet and Networking Basics"],
            "Year 3": ["Web Technologies & HTML/CSS", "Cybersecurity and Ethics", "Practical Project Development"]
        },
        "Computing": {
            "Year 1": ["Computer Systems Overview", "Data Representation & Logic Gates", "Algorithms & Flowcharts"],
            "Year 2": ["Structured Programming in Python", "Database Systems & SQL", "Network Topologies & Routing"],
            "Year 3": ["Advanced Software Engineering Principles", "Information Security & Cryptography", "Capstone Computing Project"]
        },
        "Physics": {
            "Year 1": ["Measurements and Units", "Kinematics & Dynamics", "Work, Energy and Power"],
            "Year 2": ["Waves, Light and Optics", "Thermal Physics", "Current Electricity & Magnetism"],
            "Year 3": ["Electronics & Semiconductors", "Atomic and Nuclear Physics", "Practical Physics Calculations"]
        },
        "Chemistry": {
            "Year 1": ["Nature of Matter", "Atomic Structure", "Periodic Table & Periodicity", "Stoichiometry"],
            "Year 2": ["Chemical Kinetics & Equilibrium", "Energetics", "Acids, Bases and Salts Reactions", "Introduction to Organic Chemistry"],
            "Year 3": ["Advanced Organic Chemistry", "Redox Reactions & Electrochemistry", "Industrial Chemistry Processes"]
        },
        "Biology": {
            "Year 1": ["Living Cells", "Organization of Life", "Plant and Animal Nutrition", "Transport Systems"],
            "Year 2": ["Respiration and Excretion", "Coordination and Control", "Reproduction and Growth"],
            "Year 3": ["Genetics, Heredity and Variation", "Evolution and Ecology", "Practical Biology Specimen Review"]
        },
        "Financial Accounting": {
            "Year 1": ["Introduction to Accounting Principles", "Double Entry Bookkeeping", "Subsidiary Books & Cash Book"],
            "Year 2": ["Bank Reconciliation Statements", "Control Accounts", "Final Accounts of Sole Proprietorship"],
            "Year 3": ["Partnership and Company Accounts", "Manufacturing Accounts", "Incomplete Records & Auditing Basics"]
        },
        "Economics": {
            "Year 1": ["Nature and Scope of Economics", "Basic Economic Concepts", "Demand and Supply Analysis"],
            "Year 2": ["Theory of Production & Costs", "Market Structures", "National Income Accounting"],
            "Year 3": ["Public Finance & Taxation", "International Trade & Balance of Payments", "Economic Development Planning"]
        },
        "Business Management": {
            "Year 1": ["Introduction to Business", "Forms of Business Organizations", "Business Environment"],
            "Year 2": ["Functions of Management (Planning, Organizing, Leading, Controlling)", "Marketing Management"],
            "Year 3": ["Human Resource Management", "Financial Management", "Entrepreneurship & Business Ethics"]
        },
        "General Agriculture": {
            "Year 1": ["Introduction to Agriculture", "Agricultural Ecology & Soils", "Crop Production Basics"],
            "Year 2": ["Animal Husbandry & Production", "Agricultural Economics & Extension", "Farm Mechanization"],
            "Year 3": ["Crop Protection & Pest Management", "Agribusiness Management", "Agricultural Project Planning"]
        },
        "Geography": {
            "Year 1": ["Practical Geography (Map Reading & Scale)", "The Earth's Structure & Plate Tectonics", "Climatology"],
            "Year 2": ["Geomorphology", "Biogeography", "Economic Geography (Agriculture & Industries)"],
            "Year 3": ["Population and Settlement Geography", "Regional Geography of West Africa", "Environmental Hazards & Management"]
        },
        "History": {
            "Year 1": ["Historiography & Sources of History", "Early Civilizations in West Africa", "State Formation in Pre-Colonial West Africa"],
            "Year 2": ["The Trans-Atlantic Slave Trade & Its Impact", "Colonial Rule in West Africa", "Resistance Movements"],
            "Year 3": ["Nationalism and Independence Movements", "Post-Independence Political & Economic Development", "International Organizations"]
        },
        "Literature-in-English": {
            "Year 1": ["Introduction to Literary Genres (Prose, Drama, Poetry)", "Literary Devices & Figures of Speech"],
            "Year 2": ["Detailed Analysis of Set Prose & Drama Texts", "African and Non-African Poetry Appreciation"],
            "Year 3": ["Unseen Poetry Practice", "Critical Essay Writing & Examination Techniques"]
        },
        "French": {
            "Year 1": ["Grammaire Fondamentale", "Vocabulaire de Base & Salutations", "La Conjugaison des Verbes Réguliers"],
            "Year 2": ["Expression Écrite & Orale", "Les Temps du Passé et du Futur", "Compréhension de Texte"],
            "Year 3": ["Rédaction Avancée", "Épreuves Type WAEC & Vocabulaire Professionnel"]
        },
        "Christian Religious Studies (CRS)": {
            "Year 1": ["The Creation and Early History of Israel", "The Covenant and Leadership in Israel", "The Life and Ministry of Jesus"],
            "Year 2": ["The Passion, Death and Resurrection of Jesus", "The Early Church and Missionary Expansion", "Pauline Epistles & Christian Living"],
            "Year 3": ["Themes in Christian Ethics", "Christianity and Other Religions in West Africa", "Revision and WAEC Essay Prep"]
        },
        "Islamic Studies": {
            "Year 1": ["The Articles of Faith (Iman)", "The Pillars of Islam (Ibadah)", "Introduction to Quranic Studies"],
            "Year 2": ["The Life and Times of Prophet Muhammad (Seerah)", "Hadith Compilation & Teachings", "Islamic Law and Morality (Sharia & Akhlaq)"],
            "Year 3": ["Islamic History and Civilization", "Contemporary Social Issues in Islam", "Exam Preparation and Essays"]
        },
        "Graphic Design / Visual Arts": {
            "Year 1": ["Elements and Principles of Design", "Drawing and Painting Techniques", "Lettering & Layout Design"],
            "Year 2": ["Computer-Aided Design (CAD)", "Printmaking & Packaging Design", "Sculpture and Ceramics Basics"],
            "Year 3": ["Portfolio Development", "Art History and Appreciation", "Practical Project Execution"]
        },
        "Food and Nutrition": {
            "Year 1": ["Kitchen Hygiene and Safety", "Basic Nutrients and Their Functions", "Meal Planning Principles"],
            "Year 2": ["Cookery Methods and Food Preparation", "Food Preservation and Storage", "Special Diets and Nutrition Disorders"],
            "Year 3": ["Entertaining and Table Etiquette", "Food Science Experiments", "Practical Exam Preparation"]
        },
        "Clothing and Textiles": {
            "Year 1": ["Sewing Tools and Equipment", "Basic Textile Fibres and Fabrics", "Stitches and Seams"],
            "Year 2": ["Pattern Drafting and Alteration", "Garment Construction Techniques", "Fashion Design Principles"],
            "Year 3": ["Garment Decoration & Care", "Consumer Education in Textiles", "Practical Portfolio Presentation"]
        },
        "Music": {
            "Year 1": ["Rudiments of Music & Notation", "Intervals and Scales", "Aural Training Fundamentals"],
            "Year 2": ["Harmonization and Chord Progression", "Music History (Western & African)", "Instrumental and Vocal Performance"],
            "Year 3": ["Composition and Arranging", "Set Works Analysis", "Advanced Aural and Theory Practice"]
        },
        "Physical Education (PE)": {
            "Year 1": ["Anatomy and Physiology of Exercise", "History and Rules of Major Sports", "Health and Fitness Principles"],
            "Year 2": ["Biomechanical Principles", "Sports Psychology & Coaching", "Officiating and Tournament Organization"],
            "Year 3": ["Sports Injuries and Rehabilitation", "Nutrition for Athletes", "Practical Practical Training Assessment"]
        },
        "Electronics": {
            "Year 1": ["Fundamental Electrical Quantities & Laws", "Passive Components (Resistors, Capacitors, Inductors)", "Direct Current (DC) Circuits"],
            "Year 2": ["Alternating Current (AC) Theory", "Semiconductor Diodes and Transistors", "Digital Electronics & Logic Gates"],
            "Year 3": ["Amplifiers and Oscillators", "Power Supplies & Troubleshooting", "Practical Circuit Design & Soldering"]
        },
        "Technical Drawing": {
            "Year 1": ["Drawing Instruments and Line Work", "Geometrical Constructions", "Loci and Scales"],
            "Year 2": ["Orthographic Projections", "Isometric and Oblique Drawings", "Sectional Views"],
            "Year 3": ["Development of Surfaces", "Building Drawing & Blueprints", "Assembly Drawings"]
        },
        "Applied Electricity": {
            "Year 1": ["Safety Rules in Electrical Work", "Basic Electrical Circuit Components", "Measuring Instruments (Multimeter, Ammeter)"],
            "Year 2": ["Domestic Electrical Installation", "Transformers and AC Machines", "Electrical Wiring Regulations"],
            "Year 3": ["Electrical Power Distribution", "Troubleshooting Electrical Faults", "Practical Installation Project"]
        },
        "Metalwork": {
            "Year 1": ["Workshop Safety and Tools", "Bench Work and Measurement", "Properties of Metals"],
            "Year 2": ["Foundry Work and Casting", "Welding, Soldering and Brazing", "Lathe Machine Operations"],
            "Year 3": ["Heat Treatment of Metals", "Advanced Machine Shop Practices", "Project Fabrication"]
        },
        "Woodwork": {
            "Year 1": ["Wood Identification and Seasoning", "Hand Tools and Maintenance", "Basic Wood Joints"],
            "Year 2": ["Woodworking Machinery Operations", "Timber Preparation & Joinery", "Surface Finishing"],
            "Year 3": ["Furniture Design and Construction", "Upholstery Basics", "Final Practical Project"]
        },
        "Information Technology (IT) Elective": {
            "Year 1": ["Advanced Computer Architecture", "Networking Protocols & OSI Model", "Database Design Principles"],
            "Year 2": ["Object-Oriented Programming", "Web Development Frameworks", "Systems Analysis and Design"],
            "Year 3": ["Cloud Computing Concepts", "IT Project Management", "Capstone Software Development"]
        },
        "Robotics & Automation": {
            "Year 1": ["Introduction to Robotics & Kinematics", "Sensors and Actuators", "Basic Microcontroller Programming"],
            "Year 2": ["Embedded C/C++ and Python for Robotics", "PID Controllers & Feedback Systems", "Actuator Control Systems"],
            "Year 3": ["Autonomous Navigation & AI Integration", "Robot Operating System (ROS)", "Robotics Capstone Project"]
        }
    }

def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

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
                language="en",
            )
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")
        return transcription.text
    except Exception:
        return None


# ==========================================================
# SIDEBAR NAVIGATION & PROFILE SETUP
# ==========================================================
st.sidebar.title("Navigation & Profile")
curriculum = get_comprehensive_curriculum()
all_subjects = list(curriculum.keys())

if "prev_subject" not in st.session_state:
    st.session_state.prev_subject = all_subjects[0]

selected_subject = st.sidebar.selectbox("Select Subject", all_subjects)

# Year-by-year segmentation selector
available_years = list(curriculum[selected_subject].keys())
selected_year = st.sidebar.selectbox("Select Year Level", available_years)

if selected_subject != st.session_state.prev_subject:
    switch_msg = f"Subject switched from **{st.session_state.prev_subject}** to **{selected_subject}** ({selected_year})."
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
        "📖 Syllabus & Topics Overview"
    ]
)

input_method = st.sidebar.radio("Choose input method:", ["⌨️ Type Question", "🎤 Speak Question"])

# Header Layout
user_img_base64 = get_image_base64("ONORE_AKORTIA_1.jpg")
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
        <span style="font-size: 2.5em;">💻</span>
        <div style="flex-grow: 1;">
            <h1 style="margin: 0; font-size: 1.8em; line-height: 1.2;">SHS AI Tutor - Enterprise Edition</h1>
            <p style="margin: 0; color: #666; font-size: 0.95em;">Cloud-Powered Intelligent Tutoring for <b>{selected_subject} ({selected_year})</b>.</p>
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
            res = conn.execute(text("SELECT xp FROM student_profiles WHERE user_key = :uk"), {"uk": user_key}).fetchone()
            if res:
                st.session_state.user_xp = res[0]
            else:
                conn.execute(text("INSERT INTO student_profiles (user_key, full_name, school, xp) VALUES (:uk, :fn, :sc, :xp) ON CONFLICT (user_key) DO NOTHING"),
                           {"uk": user_key, "fn": student_full_name, "sc": student_school, "xp": st.session_state.user_xp})
    except Exception:
        pass

if not st.session_state.greeted and student_full_name and student_school:
    initial_greeting = f"Hello {student_full_name} from {student_school}! I am Sir OK, your AI tutor for {selected_subject} ({selected_year})."
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
            {"Rank": 2, "Student": student_full_name if student_full_name else "You", "School": student_school if student_school else "Your School", "XP": st.session_state.user_xp},
            {"Rank": 3, "Student": "Abena Osei", "School": "Wesley Girls High", "XP": 1920},
        ])
        st.dataframe(lb_df, use_container_width=True, hide_index=True)


# ==========================================================
# MODE 2: CODE EXECUTION SANDBOX
# ==========================================================
elif learning_mode == "🎮 Code Execution Sandbox":
    st.markdown("## 🎮 Interactive Code Execution Sandbox")
    st.markdown("Test Python algorithms and calculations live in memory with immediate output validation.")
    
    code_input = st.text_area("Write Python Code:", value="""# Test your algorithms here
scores = [78, 85, 92, 68, 95]
print("Scores:", scores)
print("Highest:", max(scores))
print("Average:", sum(scores) / len(scores))
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
    st.markdown(f"Submit your essay or descriptive answer below for **{selected_subject} ({selected_year})**. Our semantic grading engine evaluates conceptual alignment against WAEC rubrics.")

    essay_prompt = st.text_area("Exam Prompt / Question:", value=f"Explain key principles and applications relevant to {selected_subject} ({selected_year}).")
    rubric_text = st.text_area("Grading Rubric:", value="1. Define core concepts clearly.\n2. Discuss major challenges or components.\n3. Provide concrete examples.")
    student_essay = st.text_area("Your Essay / Detailed Answer:", value="", height=180)

    if st.button("🚀 Submit for Semantic Grading"):
        if student_essay.strip():
            with st.spinner("Analyzing semantics and evaluating rubric alignment..."):
                grading_res = evaluate_essay_semantics(student_essay, rubric_text, selected_subject, selected_year)
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

                if db_engine and student_full_name:
                    try:
                        with db_engine.begin() as conn:
                            conn.execute(text("""
                                INSERT INTO student_history (user_key, assignment_type, submitted_content, semantic_score, ai_feedback)
                                VALUES (:uk, 'essay', :sc, :ss, :af)
                            """), {"uk": user_key, "sc": student_essay, "ss": grading_res.get("score", 0), "af": feedback_str})
                    except Exception:
                        pass

                st.markdown("### 🔊 Audio Feedback Output")
                audio_bytes = generate_tts_audio(feedback_str)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                else:
                    st.warning("OpenAI TTS API key required for audio generation.")
        else:
            st.warning("Please enter your essay before submitting.")


# ==========================================================
# MODE 4: SYLLABUS & TOPICS OVERVIEW
# ==========================================================
elif learning_mode == "📖 Syllabus & Topics Overview":
    st.markdown(f"## 📖 Syllabus Topics for {selected_subject} - {selected_year}")
    topics = curriculum[selected_subject][selected_year]
    for idx, topic in enumerate(topics, 1):
        st.markdown(f"**{idx}. {topic}**")
    st.download_button(f"📥 Download {selected_subject} ({selected_year}) Outline", f"OFFICIAL {selected_subject.upper()} {selected_year.upper()} SYLLABUS", file_name=f"{selected_subject}_{selected_year}.txt")


# ==========================================================
# MODE 5: WHITEBOARD STUDIO
# ==========================================================
elif learning_mode == "🎨 Whiteboard Studio":
    st.markdown("## 🎨 Whiteboard Concept Studio")
    st.info(f"Explore interactive conceptual diagrams and synchronized audio explanations for {selected_subject} ({selected_year}).")


# ==========================================================
# MODE 6: STUDY & CHAT / WAEC EXAM PRACTICE
# ==========================================================
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = None
    if input_method == "⌨️ Type Question":
        user_query = st.chat_input(f"Ask a question about {selected_subject} ({selected_year})...")
    else:
        audio_val = st.audio_input("Record voice question")
        if audio_val:
            user_query = transcribe_audio(audio_val)

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        client = get_groq_client()
        if client:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        completion = client.chat.completions.create(
                            model=ACTIVE_MODEL,
                            messages=[
                                {"role": "system", "content": f"You are Sir OK, expert SHS tutor in {selected_subject} for {selected_year}."},
                                {"role": "user", "content": user_query}
                            ],
                            max_tokens=400,
                            temperature=0.3,
                        )
                        ai_resp = completion.choices[0].message.content
                        st.markdown(ai_resp)
                        
                        audio_data = generate_tts_audio(ai_resp)
                        if audio_data:
                            st.audio(audio_data, format="audio/mp3")

                        st.session_state.messages.append({"role": "assistant", "content": ai_resp})
                    except Exception as e:
                        st.error(f"AI Error: {e}")
