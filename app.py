import os
import json
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="SHS AI Tutor",
    page_icon="🎓",
    layout="wide"
)

@st.cache_data
def load_curriculum_index():
    """Loads the master curriculum index from the root directory."""
    index_path = "curriculum_index.json"
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_subject_data(file_name):
    """Loads an individual subject JSON file directly from the root directory."""
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def render_parsed_text(full_text):
    """Slices the continuous textbook text into readable sections using built-in markers."""
    if not isinstance(full_text, str) or not full_text.strip():
        st.info("No text content available for this section.")
        return

    # Split the giant string cleanly by the document's section markers
    chunks = full_text.split("SECTION ")
    
    # Render front matter / intro if present
    if chunks[0].strip():
        with st.expander("📖 Front Matter & Table of Contents", expanded=False):
            st.markdown(chunks[0].strip())
    
    # Iterate through actual sections and format into expanders
    for chunk in chunks[1:]:
        lines = chunk.strip().split("\n")
        section_title = f"Section {lines[0]}" if lines else "Section Details"
        section_body = "\n".join(lines[1:]) if len(lines) > 1 else chunk
        
        with st.expander(f"📑 {section_title}", expanded=False):
            st.markdown(section_body)

def main():
    st.title("🎓 SHS AI Tutor")
    st.markdown("Your interactive intelligent tutoring system powered by curriculum data.")

    # Load the index from the root directory
    index = load_curriculum_index()

    if not index:
        st.error("⚠️ `curriculum_index.json` was not found in the root directory. Please run `convert_pdfs.py` first.")
        return

    # Sidebar Navigation
    st.sidebar.title("Curriculum Navigation")
    
    years = list(index.keys())
    selected_year = st.sidebar.selectbox("Select Year / Level", years)
    
    if selected_year:
        subjects = list(index[selected_year].keys())
        selected_subject = st.sidebar.selectbox("Select Subject", subjects)
        
        if selected_subject:
            json_filename = index[selected_year][selected_subject]
            subject_content = load_subject_data(json_filename)
            
            st.header(f"{selected_year} — {selected_subject}")
            st.caption(f"Active Source: `{json_filename}`")
            
            if subject_content:
                # Feature Integration: Tabbed Workspace (Reader + AI Chat)
                tab1, tab2 = st.tabs(["📖 Textbook Reader & Content", "🤖 AI Tutor Chat"])
                
                with tab1:
                    st.subheader("Curriculum Text Overview")
                    for category_name, full_text in subject_content.items():
                        if len(subject_content) > 1:
                            st.markdown(f"### Stream: {category_name}")
                        render_parsed_text(full_text)
                
                with tab2:
                    st.subheader(f"Ask the AI Tutor about {selected_subject}")
                    st.info(f"Ask questions based on the loaded curriculum content for {selected_year} {selected_subject}.")
                    
                    # Chat session state initialization
                    if "messages" not in st.session_state:
                        st.session_state.messages = []

                    # Display chat history
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                    # Chat input loop
                    if prompt := st.chat_input("Ask a question from this syllabus..."):
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        # Response generation hook utilizing current subject context
                        response = f"I am your SHS AI Tutor for **{selected_subject} ({selected_year})**. You asked: *'{prompt}'*. (This response space integrates with your Gemini API/curriculum context vector search)."
                        
                        with st.chat_message("assistant"):
                            st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.warning(f"Could not load contents from file: `{json_filename}`")

if __name__ == "__main__":
    main()
