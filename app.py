import os
import pandas as pd
from groq import Groq
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="SHS Computing AI Tutor", page_icon="🤖", layout="centered"
)

st.title("🤖 SHS Computing AI Textbook Tutor")
st.markdown(
    "Hello dear! Happy to see you. Kindly tell me your name and that of your school please..."
)

# Load the API key securely from Streamlit secrets
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    groq_api_key = None

if not groq_api_key:
    st.error("Please configure your Groq API key in Streamlit Secrets.")
    st.stop()

# Initialize Groq Client
client = Groq(api_key=groq_api_key)


# Load Dataset Function
@st.cache_data
def load_data():
    # Make sure 'waec_qa_dataset.csv' is in the same folder
    if os.path.exists("waec_qa_dataset.csv"):
        return pd.read_csv("waec_qa_dataset.csv")
    return None


df = load_data()

if df is None:
    st.error(
        "Dataset 'waec_qa_dataset.csv' not found! Please place it in the project folder."
    )
    st.stop()


# AI Tutor Logic
def ask_ai_tutor(query, dataset):
    # Context building from dataset rows
    context_text = "\n".join(
        [
            f"Q: {row['question_text']}\nA: {row['answer_text']}"
            for _, row in dataset.head(15).iterrows()
        ]
    )

    prompt = f"""
    You are an expert AI tutor named Sir O.K. You are created by Mr. ONORE AKORTIA, a teacher at OLA SHS,HO.You are a friendly AI tutor for Ghanaian Senior High School (SHS) Computing students.
    Use the following official curriculum context to answer the student's question accurately. 
    
    Let only your first response contain a welcome or greating message to the learner and do not repeat it. Also, do not ask for there names if they do not provide it as the inscription under your name already politely ask them to tell you their name and school.
    If the answer isn't directly in the context, use your knowledge aligned with WAEC standards.

    Context:
    {context_text}

    Student Question: {query}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with AI tutor: {e}"


# Chat Interface State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Field
if user_query := st.chat_input("What is a data type?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("AI Tutor is thinking..."):
            answer = ask_ai_tutor(user_query, df)
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
