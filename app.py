import os
from openai import OpenAI

# Initialize the Groq client with OpenAI compatibility
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY", "your-groq-api-key-here"),
    base_url="https://api.groq.com/openai/v1"
)

# ==========================================
# 1. WAEC Exam Practice Initialization
# ==========================================
def initialize_waec_practice():
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are an expert WAEC exam tutor. Initialize Question 1."
            },
            {
                "role": "user",
                "content": "Start the WAEC Exam Practice mode."
            }
        ],
        temperature=0.7,
        max_tokens=1024
    )
    return response.choices[0].message.content


# ==========================================
# 2. WAEC Exam Practice Evaluation
# ==========================================
def evaluate_waec_answer(user_answer, current_question):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "Evaluate the student's answer for the WAEC exam question, provide feedback, and serve the next question."
            },
            {
                "role": "user",
                "content": f"Question: {current_question}\nStudent's Answer: {user_answer}"
            }
        ],
        temperature=0.7,
        max_tokens=1024
    )
    return response.choices[0].message.content


# ==========================================
# 3. Study & Chat Mode
# ==========================================
def study_chat_response(prompt_text):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful educational AI assistant for study and chat sessions."
            },
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        temperature=0.7,
        max_tokens=2048
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("Full updated script configured successfully using model='openai/gpt-oss-20b'.")
