
import streamlit as st
import tempfile
import os
from pipeline import pipeline_for_video
from gpt_utils import combine_transcripts_with_gpt, assess_transcript_quality

st.set_page_config(page_title="Nursing Student Simulation Transcription & Assessment", layout="wide")

# --- Custom CSS for a prettier design ---
st.markdown(
    """
    <style>
    .main {
        background-color: #f7fafc;
    }
    .stApp {
        background: linear-gradient(120deg, #e0f7fa 0%, #fce4ec 100%);
    }
    .title-text {
        font-size: 2.6rem;
        font-weight: 700;
        color: #1976d2;
        margin-bottom: 0.5em;
        text-align: center;
    }
    .subtitle-text {
        font-size: 1.3rem;
        color: #333;
        text-align: center;
        margin-bottom: 2em;
    }
    .stButton>button {
        background-color: #1976d2;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1.5em;
    }
    .stDownloadButton>button {
        background-color: #43a047;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1.5em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title and Description ---
st.markdown('<div class="title-text">🩺 Nursing Student Simulation: Video Transcription & AI Quality Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Upload your simulation videos, get accurate transcriptions using Whisper, combine them with GPT, and receive an AI-powered quality assessment. Perfect for nursing education, simulation debrief, and reflective learning.</div>', unsafe_allow_html=True)

# --- Sidebar for API Key ---
st.sidebar.header("🔑 Settings")
st.sidebar.markdown("Enter your <b>OpenAI API Key</b> to enable transcript combination and assessment.", unsafe_allow_html=True)
api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=st.session_state.get('api_key', ''))
if api_key:
    st.session_state['api_key'] = api_key

# --- File Uploader ---
st.markdown("### 1️⃣ Upload Simulation Video Files (MP4)")
video_files = st.file_uploader("Upload one or more simulation videos (MP4)", type=["mp4"], accept_multiple_files=True, help="You can upload several simulation recordings at once.")


if video_files and api_key:
    st.markdown("### 2️⃣ Transcription Progress")
    if 'transcripts' not in st.session_state:
        st.session_state['transcripts'] = []
    if st.button("📝 Start Transcription", help="Click to transcribe all uploaded videos."):
        transcripts = []
        for idx, uploaded_file in enumerate(video_files):
            with st.expander(f"Video {idx+1}: {uploaded_file.name}", expanded=True):
                st.info("Transcribing... This may take a few minutes depending on video length.")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                try:
                    transcript_lines = pipeline_for_video(tmp_path)
                    transcript_text = "\n".join(transcript_lines)
                    transcripts.append(transcript_text)
                    st.success(f"Transcription complete for {uploaded_file.name}")
                    st.text_area("Transcript", transcript_text, height=150)
                except Exception as e:
                    st.error(f"Failed to process {uploaded_file.name}: {e}")
                finally:
                    os.remove(tmp_path)
        st.session_state['transcripts'] = transcripts

    if st.session_state.get('transcripts'):
        st.markdown("### 3️⃣ Combine Transcripts & Assess Quality with GPT")
        if st.button("🤖 Combine & Assess with GPT", help="Uses your OpenAI key. No data is stored."):
            with st.spinner("Combining transcripts with GPT..."):
                combined = combine_transcripts_with_gpt(st.session_state['transcripts'], api_key)
                st.session_state['combined'] = combined
                st.subheader("📝 Combined Transcript")
                st.text_area("Combined Transcript", combined, height=300)
                st.download_button("⬇️ Download Combined Transcript", combined, file_name="combined_transcript.txt")
            with st.spinner("Assessing transcript quality with GPT..."):
                assessment = assess_transcript_quality(combined, api_key)
                st.session_state['assessment'] = assessment
                st.subheader("📊 Quality Assessment")
                st.write(assessment)
        elif st.session_state.get('combined'):
            st.subheader("📝 Combined Transcript")
            st.text_area("Combined Transcript", st.session_state['combined'], height=300)
            st.download_button("⬇️ Download Combined Transcript", st.session_state['combined'], file_name="combined_transcript.txt")
            if st.session_state.get('assessment'):
                st.subheader("📊 Quality Assessment")
                st.write(st.session_state['assessment'])
else:
    st.info("Please upload simulation video files and enter your OpenAI API key in the sidebar.")
