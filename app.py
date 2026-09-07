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


def main():
    st.title("🎓 SHS AI Tutor")
    st.markdown("Select a year and subject from the sidebar to explore the curriculum content.")

    # Load the index from the root directory
    index = load_curriculum_index()

    if not index:
        st.error(
            "⚠️ `curriculum_index.json` was not found in the root directory. Please ensure your converted JSON files are generated in the root folder.")
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
            st.caption(f"Loaded from root file: `{json_filename}`")

            if subject_content:
                for section_name, text_content in subject_content.items():
                    with st.expander(f"Section: {section_name}", expanded=True):
                        st.write(text_content)
            else:
                st.warning(f"Could not load contents from file: `{json_filename}`")


if __name__ == "__main__":
    main()
