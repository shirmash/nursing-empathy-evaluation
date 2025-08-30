# Nursing Empathy Evaluation

This tool is designed to transcribe and evaluate nursing simulations in Hebrew. It turns your recording into a clear Nurse–Patient transcript and gives a quick score with brief notes on the nurse’s use of empathetic language.

> **Heads-up:** Transcripts are auto-generated and might miss or mangle a few words. Use clinical judgment and fix anything that looks off.

---

## Live app
Replace this with your deployed URL (optional):
```
https://nursing-empathy-evaluation-il.streamlit.app/
```

## How it works (quick guide)

1) **Upload** your simulation recording(s) — MP4/MOV/WEBM/MKV/MP3/WAV/M4A.  
2) **Enter your OpenAI key** in the left sidebar.  
3) Click **Transcribe (and auto-combine)**.  
   - The app turns each file into text and **merges everything** into one clean, time-ordered conversation with roles: **Nurse / Patient** (OOC if needed).  
4) Review the **Combined Transcript** (you can download it as `.txt`).  
5) (Optional) Click **Assess empathy (GPT-4o)** to get a **1–5 score** with a short reason focused on the nurse’s lines.

**What you get**
- A clear transcript for debriefs (no raw chunks shown).
- A quick empathy check on the nurse’s communication.

**Tips**
- Multiple uploads are fine — they’ll be stitched into one timeline.  
- Very long/quiet audio may be harder to transcribe; check the text and edit if needed.  
- Keep patient privacy in mind when sharing transcripts.


> **Heads-up:** Transcripts are auto-generated and can miss a word here or there. Use your clinical judgment and correct anything that looks off.


- **OpenAI API key** – required for:
  - **Combine**: merges all transcripts into a single, time-ordered Nurse/Patient conversation (uses **GPT-4o**).
  - **Empathy check**: 1–5 score with brief justification focused on the nurse’s lines (uses **GPT-4o**).

**How to create an API key (official guide):**  
See OpenAI’s step-by-step “Create and export an API key” instructions.  
https://platform.openai.com/docs/quickstart/create-and-export-an-api-key


## Features
- 🎙️ **Speech-to-text:** Uses OpenAI **gpt-4o-transcribe** to convert video/audio to text.
- 🧩 **Smart merge:** Combines multiple uploads into a single, timestamped, role-tagged transcript (`Nurse`, `Patient`, with `(OOC)` when needed).
- 🧠 **Empathy check:** One-line 1–5 score + short justification (Hebrew template) using **GPT-4o**.
- 💾 **Export:** Download the combined transcript as `.txt`.
- 🔌 **local option available** Works fully with the API; optional local Whisper model usage if gpu is available.

---

## Quick start (local)

1. **Python 3.8–3.12** and **ffmpeg** (see FFmpeg below).
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI key (or use the sidebar in the app):
   ```bash
   export OPENAI_API_KEY="sk-..."   # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
   ```
4. Run:
   ```bash
   streamlit run app.py
   ```
5. In the app: upload `.mp4/.mov/.webm/.mkv/.mp3/.wav/.m4a`, click **Transcribe (and auto-combine)**, then **Assess empathy (GPT-4o)** if you want the score.

---

## Deploy on Streamlit Cloud

1. Push these files to your repo:
   - `app.py`
   - `gpt_utils.py`
   - `requirements.txt`
   - `README.md`
2. Create a new Streamlit app pointing to `app.py`.
3. Add your secret: **Manage app → Settings → Secrets**
   ```toml
   OPENAI_API_KEY="sk-..."
   ```
4. Build runs automatically. `imageio-ffmpeg` fetches a static **ffmpeg** at runtime—no manual install needed.

---

## Configuration

- **Models (fixed by this app):**
  - Speech-to-text: `gpt-4o-transcribe`
  - Text (merge + empathy): `gpt-4o`
- **Behavior:**
  - Only the **combined** transcript is shown (raw per-file transcripts are hidden).
  - Role tags are derived from content; do not alternate mechanically.
- **Environment variable:** `OPENAI_API_KEY` (or paste it in the sidebar).

---

## Requirements

Put this in `requirements.txt`:

```txt
streamlit>=1.36
openai>=1.40.0
pydub>=0.25.1
imageio-ffmpeg>=0.4.9
```

> Optional local path (only if you want Whisper on your own machine):
> ```txt
> git+https://github.com/openai/whisper.git
> torch
> ```

---

## FFmpeg

- **Cloud:** no action needed.
- **Local:** install ffmpeg if missing.
  - **Windows:** download static build from https://www.gyan.dev/ffmpeg/builds/ → extract → add `.../bin` to PATH → `ffmpeg -version`
  - **macOS:** `brew install ffmpeg`
  - **Ubuntu/Debian:** `sudo apt-get update && sudo apt-get install -y ffmpeg`

---

## Project structure

```
.
├─ app.py            # Streamlit UI (auto-combine + assess)
├─ gpt_utils.py      # OpenAI v1 helpers (GPT-4o, empathy template)
├─ pipeline.py       # (optional) local Whisper path if you keep it
├─ requirements.txt
└─ README.md
```

---

## How it works (high level)

1) **Transcribe**: Extract audio → split into ~5-minute chunks → call `gpt-4o-transcribe` → stitch text with timestamps.  
2) **Combine**: Merge all transcripts into a single, chronological, role-tagged dialogue.  
3) **Assess**: Apply the Hebrew empathy template to **Nurse** lines only and return a one-line score with reasons.

---

