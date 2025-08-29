import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import streamlit as st

# Text models utilities (GPT-4o, empathy template)
from gpt_utils import combine_transcripts_with_gpt, assess_transcript_quality

# Optional: local Whisper pipeline, if you still keep it around
try:
    from pipeline import pipeline_for_video  # your local Whisper path (optional)
except Exception:
    pipeline_for_video = None

# ------------------ ffmpeg helper (cloud-safe) ------------------
def ensure_ffmpeg() -> str:
    """
    Ensure an ffmpeg binary exists (downloads a static one if needed).
    Also wires pydub to use that binary.
    """
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["IMAGEIO_FFMPEG_EXE"] = ff
    try:
        from pydub import AudioSegment
        AudioSegment.converter = ff
    except Exception:
        pass
    return ff

# ------------------ OpenAI helpers ------------------
def _openai_client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key)

def transcribe_with_openai_single(file_path: str, api_key: str) -> str:
    """
    Send a single (small) audio file to OpenAI STT and return text.
    """
    client = _openai_client(api_key)
    with open(file_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",   # forced
            file=f,
            response_format="text",
        )
    return str(resp)

def extract_audio_ffmpeg(video_path: str) -> str:
    """
    Extract mono 16kHz WAV from a video/audio file using ffmpeg.
    """
    ff = ensure_ffmpeg()
    out_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    import subprocess
    cmd = [ff, "-y", "-i", video_path, "-ac", "1", "-ar", "16000", out_wav]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_wav

def split_audio(audio_path: str, chunk_len_ms: int = 5 * 60 * 1000) -> List[Tuple[str, int]]:
    """
    Split audio into ~5-minute chunks using pydub.
    Returns list of (chunk_file_path, start_offset_seconds)
    """
    ensure_ffmpeg()  # makes sure pydub picks up the binary
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
    """
    Extract audio, split into chunks, transcribe each via OpenAI, and return timestamped lines.
    """
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
                # Simple line split; keeps UI readable and merges well later
                for sent in [s.strip() for s in txt.replace("\r", " ").split("\n") if s.strip()]:
                    lines.append(f"[{_fmt_time(start_sec)}] {sent}")
            finally:
                try:
                    os.remove(chunk_path)
                except Exception:
                    pass
        return lines
    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Nursing Simulation: Transcribe & Assess", layout="wide")
st.title("🩺 Nursing Simulation: Transcribe & Assess")

# Short, friendly explanation (very short)
st.markdown("""
What this does:Transcribes your Hebrew simulation and gives a quick empathy check on the nurse’s lines.

How to use: 1) Upload video/audio → 2) Enter your OpenAI key → 3) Click **Transcribe (and auto-combine)** → 4) (Optional) **Assess empathy (GPT-4o)**.
""")
st.info("Heads-up: Transcript is auto-generated and may not be completely accurate.")

# Sidebar: settings + reset
st.sidebar.header("Settings")
api_key_input = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))

# Optional local mode toggle (kept for flexibility)
engine = st.sidebar.radio(
    "Transcription engine",
    ["OpenAI API (gpt-4o-transcribe)", "Local Whisper (requires GPU)"],
    index=0
)

# Reset controls
st.sidebar.markdown("---")
reset_all = st.sidebar.button("🔄 Reset app (clear uploads & results)")

# Session state
if "raw_transcripts" not in st.session_state:
    st.session_state["raw_transcripts"] = []
if "combined" not in st.session_state:
    st.session_state["combined"] = ""
if "assessment" not in st.session_state:
    st.session_state["assessment"] = ""
if "upload_key" not in st.session_state:
    st.session_state["upload_key"] = 0

if reset_all:
    st.session_state["raw_transcripts"] = []
    st.session_state["combined"] = ""
    st.session_state["assessment"] = ""
    st.session_state["upload_key"] += 1  # force uploader to reset
    st.rerun()


# Uploader (keyed so it can reset cleanly)
uploaded = st.file_uploader(
    "Upload one or more simulation video files (MP4/MOV/WEBM/MKV/MP3/WAV/M4A)",
    type=["mp4","mov","webm","mkv","mp3","wav","m4a","mpeg4"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['upload_key']}",
)

# Buttons
col1, col2 = st.columns(2)
with col1:
    go_btn = st.button("Transcribe (and auto-combine)")
with col2:
    assess_btn = st.button("Assess empathy (GPT-4o)")

# Actions
if go_btn:
    if not uploaded:
        st.warning("Please upload at least one file.")
    elif engine.startswith("OpenAI") and not api_key_input:
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
                        lines = transcribe_long_with_openai(tmp_path, api_key_input)
                else:
                    with st.spinner("Transcribing locally with Whisper (requires GPU)..."):
                        lines = pipeline_for_video(tmp_path)

                transcript_text = "\n".join(lines) if isinstance(lines, list) else str(lines)
                st.session_state["raw_transcripts"].append(transcript_text)
                st.success(f"Finished: {f.name}")
            except Exception as e:
                st.error(f"Failed on {f.name}: {e}")
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            progress.progress(i / len(uploaded))

        # Auto-combine (no raw transcripts shown)
        if not api_key_input:
            st.warning("OpenAI API key is required to combine transcripts.")
        else:
            with st.spinner("Combining transcripts with GPT-4o..."):
                st.session_state["combined"] = combine_transcripts_with_gpt(
                    st.session_state["raw_transcripts"], api_key_input
                )
            st.subheader("📝 Combined Transcript")
            st.text_area("Combined", st.session_state["combined"], height=320)
            st.download_button(
                "Download combined transcript",
                st.session_state["combined"].encode("utf-8"),
                file_name="combined_transcript.txt",
            )

if assess_btn:
    if not st.session_state.get("combined"):
        st.warning("No combined transcript yet. Click 'Transcribe (and auto-combine)' first.")
    elif not api_key_input:
        st.warning("Enter your OpenAI API key.")
    else:
        with st.spinner("Assessing empathy with GPT-4o..."):
            st.session_state["assessment"] = assess_transcript_quality(
                st.session_state["combined"], api_key_input
            )
        st.subheader("📊 Empathy Assessment")
        st.write(st.session_state["assessment"])
