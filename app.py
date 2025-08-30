
import streamlit as st
from pathlib import Path
# Text models utilities (updated to GPT-4o)
from gpt_utils import combine_transcripts_with_gpt, assess_transcript_quality
import os, tempfile, subprocess
# Optional local pipeline (keep if you still want it)
try:
    from pipeline import pipeline_for_video  # your existing local Whisper path
except Exception:
    pipeline_for_video = None
from typing import List, Tuple
# ------------- OpenAI STT (forced to gpt-4o-transcribe) -------------
def _openai_client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key)

def transcribe_with_openai_single(file_path: str, api_key: str) -> str:
    client = _openai_client(api_key)
    with open(file_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",   # forced (no mini)
            file=f,
            response_format="text",
        )
    return str(resp)

def extract_audio_ffmpeg(video_path: str) -> str:
    out_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    import subprocess
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", out_wav]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_wav

def split_audio(audio_path: str, chunk_len_ms: int = 5 * 60 * 1000) -> List[Tuple[str, int]]:
    from pydub import AudioSegment
    audio = AudioSegment.from_file(audio_path)
    chunks: List[Tuple[str, int]] = []
    for i in range(0, len(audio), chunk_len_ms):
        segment = audio[i:i+chunk_len_ms]
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        segment.export(fp, format="wav")
        chunks.append((fp, i // 1000))
    return chunks

def transcribe_long_with_openai(video_path: str, api_key: str) -> List[str]:
    def _fmt_time(seconds: int) -> str:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
    audio_path = extract_audio_ffmpeg(video_path)
    try:
        chunks = split_audio(audio_path)
        lines: List[str] = []
        for chunk_path, start_sec in chunks:
            try:
                txt = transcribe_with_openai_single(chunk_path, api_key)
                for sent in [s.strip() for s in txt.replace("\r"," ").split("\n") if s.strip()]:
                    lines.append(f"[{_fmt_time(start_sec)}] {sent}")
            finally:
                try: os.remove(chunk_path)
                except Exception: pass
        return lines
    finally:
        try: os.remove(audio_path)
        except Exception: pass

# ------------------------- UI -------------------------
st.set_page_config(page_title="Nursing Simulation: Transcribe & Assess", layout="wide")
st.title("🩺 Nursing Simulation: Transcribe & Assess")
st.markdown("""
> This tool is designed to transcribe and evaluate nursing simulations in Hebrew. It turns your recordings into a clear Nurse–Patient transcript and gives a quick score with brief notes on the nurse’s use of empathetic language.
""")

st.markdown("""

**How to use:** 1) Upload video/audio → 2) Enter your OpenAI key → 3) Click **Transcribe (and auto-combine)** → 4) (Optional) **Assess empathy (GPT-4o)**.
""")
st.info("Heads-up: Transcript is auto-generated and may not be fully accurate.")



st.sidebar.header("Settings")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Keep engine toggle if you still want local option; default to OpenAI
engine = st.sidebar.radio(
    "Transcription engine",
    ["OpenAI API (gpt-4o-transcribe)", "Local Whisper (requires GPU)"],
    index=0
)

uploaded = st.file_uploader(
    "Upload one or more simulation video files (MP4/MOV/WEBM/MP3/WAV)",
    type=["mp4","mov","webm","mkv","mp3","wav","m4a","mpeg4"],
    accept_multiple_files=True
)

# session state
if "raw_transcripts" not in st.session_state:
    st.session_state["raw_transcripts"] = []
if "combined" not in st.session_state:
    st.session_state["combined"] = ""
if "assessment" not in st.session_state:
    st.session_state["assessment"] = ""

col1, col2 = st.columns(2)
with col1:
    go_btn = st.button("Transcribe (and auto-combine)")
with col2:
    assess_btn = st.button("Assess empathy (GPT-4o)")

# --------- Transcribe -> Auto-Combine (no raw shown) ----------
if go_btn:
    if not uploaded:
        st.warning("Please upload at least one file.")
    elif engine.startswith("OpenAI") and not api_key:
        st.warning("Enter your OpenAI API key to use the API engine.")
    elif engine.startswith("Local") and pipeline_for_video is None:
        st.error("Local Whisper pipeline is not available.")
    else:
        st.session_state["raw_transcripts"] = []
        progress = st.progress(0)
        for i, f in enumerate(uploaded, start=1):
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
                tmp.write(f.getbuffer())
                tmp_path = tmp.name
            st.write(f"**Processing:** {f.name}")
            try:
                if engine.startswith("OpenAI"):
                    with st.spinner("Transcribing with OpenAI (gpt-4o-transcribe)..."):
                        lines = transcribe_long_with_openai(tmp_path, api_key)
                else:
                    with st.spinner("Transcribing locally with Whisper (small)..."):
                        lines = pipeline_for_video(tmp_path)
                transcript_text = "\n".join(lines) if isinstance(lines, list) else str(lines)
                st.session_state["raw_transcripts"].append(transcript_text)
                st.success(f"Finished: {f.name}")
            except Exception as e:
                st.error(f"Failed on {f.name}: {e}")
            finally:
                try: os.remove(tmp_path)
                except Exception: pass
            progress.progress(i / len(uploaded))

        # AUTO-COMBINE with GPT-4o (no raw transcript shown)
        if not api_key:
            st.warning("OpenAI API key is required to combine transcripts.")
        else:
            with st.spinner("Combining transcripts with GPT-4o..."):
                st.session_state["combined"] = combine_transcripts_with_gpt(
                    st.session_state["raw_transcripts"], api_key
                )
            st.subheader("📝 Combined Transcript")
            st.text_area("Combined", st.session_state["combined"], height=300)
            st.download_button(
                "Download combined transcript",
                st.session_state["combined"].encode("utf-8"),
                file_name="combined_transcript.txt",
            )

# --------- Assess (uses GPT-4o) ----------
if assess_btn:
    if not st.session_state.get("combined"):
        st.warning("No combined transcript yet. Click 'Transcribe (and auto-combine)' first.")
    elif not api_key:
        st.warning("Enter your OpenAI API key.")
    else:
        with st.spinner("Assessing empathy with GPT-4o..."):
            st.session_state["assessment"] = assess_transcript_quality(
                st.session_state["combined"], api_key
            )
        st.subheader("📊 Empathy Assessment")
        st.write(st.session_state["assessment"])