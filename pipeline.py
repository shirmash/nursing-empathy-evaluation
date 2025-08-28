import os
import tempfile
import whisper
from pydub import AudioSegment
from datetime import timedelta
import Levenshtein

def extract_audio(video_path):
    audio_output_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    result = os.system(f'ffmpeg -y -i "{video_path}" -vn -acodec mp3 "{audio_output_path}"')
    if result != 0:
        raise RuntimeError("Audio extraction failed.")
    return audio_output_path

def split_audio(audio_path, chunk_length_ms=5 * 60 * 1000):
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        temp_chunk = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        chunk.export(temp_chunk.name, format="mp3")
        chunks.append((temp_chunk.name, i // 1000))
    return chunks

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def is_valid_segment(text, previous_segments, min_length=3, similarity_threshold=0.8):
    text = str(text).strip()
    if len(text) < min_length:
        return False
    if text.replace('.', '').replace(',', '').replace('?', '').replace('!', '').strip() == '':
        return False
    for prev_text in previous_segments[-5:]:
        prev_text = str(prev_text)
        similarity = Levenshtein.ratio(text.lower(), prev_text.lower())
        if similarity > similarity_threshold:
            return False
    return True

def transcribe_chunks(chunks, model_name="large"): 
    model = whisper.load_model(model_name)
    transcript = []
    recent_segments = []
    for chunk_path, offset in chunks:
        result = model.transcribe(
            chunk_path,
            language="he",
            word_timestamps=True,
            condition_on_previous_text=False,
            temperature=0.0,
            beam_size=5,
            best_of=5,
            patience=1.0,
            no_speech_threshold=0.6,
            suppress_blank=True,
            fp16=True,
            verbose=False,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
        )
        for segment in result["segments"]:
            text = str(segment['text']).strip()
            if not is_valid_segment(text, recent_segments):
                continue
            if text in [str(line.split('] ', 1)[1]) for line in transcript[-3:] if '] ' in line]:
                continue
            start_time = format_time(segment['start'] + offset)
            line = f"[{start_time}] {text}"
            transcript.append(str(line))
            recent_segments.append(text)
            if len(recent_segments) > 10:
                recent_segments.pop(0)
    transcript = remove_duplicate_lines(transcript)
    return transcript

def remove_duplicate_lines(transcript, similarity_threshold=0.85):
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
                if len(current_text) > len(prev_text):
                    cleaned_transcript[j] = current_line
                is_duplicate = True
                break
        if not is_duplicate:
            cleaned_transcript.append(current_line)
    return cleaned_transcript

def pipeline_for_video(video_path):
    audio_path = extract_audio(video_path)
    chunks = split_audio(audio_path)
    transcript_lines = transcribe_chunks(chunks)
    os.remove(audio_path)
    for chunk_path, _ in chunks:
        os.remove(chunk_path)
    return transcript_lines
