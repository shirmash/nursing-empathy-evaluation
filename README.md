# Nursing Empathy Evaluation

Turn nursing simulation recordings into an easy-to-read conversation and a quick snapshot of empathy.

> **Heads-up:** Transcripts are auto-generated and might miss or mangle a few words. Use clinical judgment and fix anything that looks off.

---

## Live app
Replace this with your deployed URL (optional):
```
https://your-deployment-link-here
```

---

## Features
- 🎙️ **Speech-to-text:** Uses OpenAI **gpt-4o-transcribe** to convert video/audio to text.
- 🧩 **Smart merge:** Combines multiple uploads into a single, timestamped, role-tagged transcript (`Nurse`, `Patient`, with `(OOC)` when needed).
- 🧠 **Empathy check:** One-line 1–5 score + short justification (Hebrew template) using **GPT-4o**.
- 💾 **Export:** Download the combined transcript as `.txt`.
- 🔌 **No local Whisper required:** Works fully with the API; optional local Whisper path can remain in the repo if you want it.

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

- **Cloud:** handled automatically by `imageio-ffmpeg` (no action needed).
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

## Troubleshooting

- `ImportError: from openai import OpenAI` → Upgrade the SDK:
  ```bash
  pip install --upgrade openai
  ```
- `APIRemovedInV1: ChatCompletion` → Remove any `openai.ChatCompletion.create(...)` calls; this app uses the v1 client:
  `OpenAI(...).chat.completions.create(...)`.
- `ffmpeg not found` (local) → Install ffmpeg and ensure it’s in PATH, or rely on `imageio-ffmpeg` in cloud.
- Empty/partial transcription → very long/quiet file; try re-uploading, check chunking, and verify API quota.

---

## Privacy

Only upload material you’re allowed to use. Avoid PHI unless you have explicit consent and follow your policy. Auto-generated transcripts may contain errors—review critically.

---

## License

MIT (or your preferred license)
