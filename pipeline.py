# pipeline.py

import os
import tempfile
import subprocess
from datetime import timedelta

import whisper
import torch
from pydub import AudioSegment
import Levenshtein


# --- add to pipeline.py ---
from openai import OpenAI
import re

def _transcribe_chunk_openai(chunk_path: str, api_key: str, model: str = "gpt-4o-transcribe") -> str:
    client = OpenAI(api_key=api_key)
    with open(chunk_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=model,      # or "gpt-4o-mini-transcribe"
            file=f,
            response_format="text"
        )
    # SDK returns a plain string for response_format="text"
    return str(resp)

def pipeline_for_video_openai(video_path: str, api_key: str, model: str = "gpt-4o-transcribe"):
    audio_path = extract_audio(video_path)
    try:
        chunks = split_audio(audio_path)  # you already return (path, offset_seconds)
        lines = []
        for chunk_path, offset in chunks:
            text = _transcribe_chunk_openai(chunk_path, api_key, model=model)
            # slap coarse timestamps on sentences so your UI stays the same shape
            for sent in filter(None, [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)]):
                lines.append(f"[{format_time(offset)}] {sent}")
        # optionally reuse your duplicate cleaner
        return remove_duplicate_lines(lines)
    finally:
        try: os.remove(audio_path)
        except Exception: pass


# --- Settings to mirror your original script ---
LANGUAGE = "he"
DEFAULT_MODEL = "large"           # same default as your working script
CHUNK_LENGTH_MS = 5 * 60 * 1000   # 5 minutes


def extract_audio(video_path: str) -> str:
    """Extract audio to a temp mp3 using ffmpeg (raises with stderr if it fails)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    audio_output_path = tmp.name

    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "mp3", audio_output_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed.\n\nffmpeg stderr:\n{result.stderr}")
    return audio_output_path


def split_audio(audio_path: str, chunk_length_ms: int = CHUNK_LENGTH_MS):
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        temp_chunk = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        chunk.export(temp_chunk.name, format="mp3")
        chunks.append((temp_chunk.name, i // 1000))  # store offset in seconds
    return chunks


def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def is_valid_segment(text, previous_segments, min_length=3, similarity_threshold=0.8):
    text = str(text).strip()
    if len(text) < min_length:
        return False
    # punctuation-only
    if text.replace('.', '').replace(',', '').replace('?', '').replace('!', '').strip() == '':
        return False
    for prev_text in previous_segments[-5:]:
        prev_text = str(prev_text)
        similarity = Levenshtein.ratio(text.lower(), prev_text.lower())
        if similarity > similarity_threshold:
            return False
    return True


def transcribe_chunks(chunks, model_name: str = DEFAULT_MODEL):
    # keep behavior but avoid fp16 crash on CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)
    use_fp16 = (device == "cuda")

    transcript = []
    recent_segments = []

    for chunk_path, offset in chunks:
        result = model.transcribe(
            chunk_path,
            language=LANGUAGE,
            word_timestamps=True,
            condition_on_previous_text=False,
            temperature=0.0,
            beam_size=5,
            best_of=5,
            patience=1.0,
            no_speech_threshold=0.6,
            suppress_blank=True,
            fp16=use_fp16,
            verbose=False,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
        )

        for segment in result.get("segments", []):
            text = str(segment.get("text", "")).strip()
            if not is_valid_segment(text, recent_segments):
                continue
            # avoid exact repetition within last few lines
            if text in [str(line.split('] ', 1)[1]) for line in transcript[-3:] if '] ' in line]:
                continue

            start_time = format_time(segment['start'] + offset)
            line = f"[{start_time}] {text}"
            transcript.append(line)
            recent_segments.append(text)

            if len(recent_segments) > 10:
                recent_segments.pop(0)

    return remove_duplicate_lines(transcript)


def remove_duplicate_lines(transcript, similarity_threshold=0.85):
    """Remove duplicate or very similar consecutive lines (preserves your output format)."""
    if not transcript:
        return transcript

    cleaned_transcript = [transcript[0]]
    for i in range(1, len(transcript)):
        current_line = transcript[i]
        current_text = str(current_line.split('] ', 1)[1] if '] ' in current_line else current_line)
        is_duplicate = False

        for j in range(max(0, len(cleaned_transcript) - 3), len(cleaned_transcript)):
            prev_line = cleaned_transcript[j]
            prev_text = str(prev_line.split('] ', 1)[1] if '] ' in prev_line else prev_line)

            if current_text == prev_text:
                is_duplicate = True
                break

            similarity = Levenshtein.ratio(current_text.lower(), prev_text.lower())
            if similarity > similarity_threshold:
                # keep the longer one (same rule as your script)
                if len(current_text) > len(prev_text):
                    cleaned_transcript[j] = current_line
                is_duplicate = True
                break

        if not is_duplicate:
            cleaned_transcript.append(current_line)

    return cleaned_transcript


def pipeline_for_video(video_path: str, model_name: str = DEFAULT_MODEL):
    """Main entry: returns list of lines like '[HH:MM:SS] text' (same as your script)."""
    audio_path = extract_audio(video_path)
    try:
        chunks = split_audio(audio_path)
        try:
            transcript_lines = transcribe_chunks(chunks, model_name=model_name)
            return transcript_lines
        finally:
            # cleanup chunk files
            for chunk_path, _ in chunks:
                try:
                    os.remove(chunk_path)
                except Exception:
                    pass
    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass
