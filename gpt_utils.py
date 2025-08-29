# gpt_utils.py — Python 3.8 safe + OpenAI v1 SDK, forced to GPT-4o
# pip install --upgrade openai

from typing import List
from openai import OpenAI
import re

# === Hebrew Empathy Evaluation ===
EMPATHY_PROMPT_TEMPLATE = r"""
את/ה בוחן/ת איכות תקשורת של סטודנטית לסיעוד בתרגול סימולציה.

המשימה: להעריך את **השפה האמפתית** של הסטודנטית.
התבסס/י רק על שורות המסומנות `Nurse` בתמליל המצורף. 
מה נחשב לשפה אמפתית? ביטויים שמכירים ברגש, משקפים את דברי המטופל ומאשרים את תחושותיו (למשל הכרה בכאב/מבוכה, נרמול, הזמנה לשיתוף, הבטחת זמינות, שיקוף תוכן).

הוראות:
1) קרא/י את התמליל (הסופי והמאוחד).
2) אל תוסיף/י תוכן שלא מופיע בטקסט.
3) בנימוקים יש לציין בקצרה גם **מה הוריד את הציון או מה היה חסר** (למשל: מעט/בלי הכרה ברגש; אין שיקוף; ללא הזמנה לשיתוף; היעדר הבטחת זמינות; טון פקודי/מהיר מדי).

פלט מבוקש (שורה אחת בלבד):
שפה אמפתית: [ציון 1–5] – [נימוקים קצרים עם ציטוטים רלוונטיים מהשורות של Nurse. יש לזכור שאיסוף מידע רפואי ומתן טיפול הם חלק טבעי מהמפגש, ולכן אינם מורידים מהציון. ההערכה מתמקדת בשאלה האם שולבו גם ביטויים של שפה אמפתית כגון הכרה ברגש, נרמול, שיקוף, הזמנה לשיתוף והבטחת זמינות.]
תמליל:
{final_transcript}
""".strip() + "\n"

# === Merge/clean prompt (N transcripts supported) ===
PROMPT_PREAMBLE = """
You are reviewing a Hebrew-language nursing simulation dialogue.
Each of the three transcripts comes from a different camera angle or microphone and may contain only part of the conversation.
The transcriptions were generated automatically using Whisper, so they may include errors such as:
- Missing or incomplete sentences
- Repeated or disfluent phrases
- Slight timing misalignment

Your task is to reconstruct a **clean, coherent, and chronologically accurate** transcript using the following rules:

1. **Use the timestamps ([HH:MM:SS])** to place lines in the correct order. Reconstruct the conversation sequence based on time, even if the lines appear in different transcripts.
2. **Do NOT add or imagine content.** Only use what appears in the provided transcripts.
3. **Fix transcription errors where appropriate, but retain as much of the original information as possible.** Prioritize keeping all clinically or contextually meaningful content from the transcripts, even if phrased imperfectly. Only omit redundant, broken, or clearly meaningless lines.
4. **Assign speaker roles** based on content and context. Use only these roles:
   - `Nurse`
   - `Patient`
5. Do NOT alternate roles mechanically. Assign roles based on what is said.
6. If the same line appears in more than one transcript, **merge or choose the clearest version**. Avoid duplication.
7. Keep the transcript in a natural, readable flow that resembles a real dialogue.
8. **Fix malformed Hebrew words** when clearly misrecognized (e.g., "להתעברר" → "להתאוורר"), but do not invent content.
9. Preserve **simulation-side comments** (like when the nurse speaks to herself, to colleagues, or refers to the patient in third person such as “אותה”). These are **in-character clinical planning remarks**, not spoken to the patient. Keep them in the flow and tag them as `Nurse`, not as OOC.
10. If someone speaks **outside of character (OOC)** — for example, asking instructors, reacting to simulation errors, or breaking the scene — keep the line, and **tag it clearly** as one of:
    - `Nurse (OOC)`
    - `Patient (OOC)`
11. Treat repeated "תודה רבה" at the end or start of the transcripts as likely Whisper artifacts — omit them from the final merged transcript unless clearly part of the dialogue.

Return only the cleaned and role-tagged transcript in the following format:
[HH:MM:SS] Role: Sentence
""".strip()

def _client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def _build_merge_prompt(transcripts: List[str]) -> str:
    if isinstance(transcripts, str):
        transcripts = [transcripts]
    parts = [PROMPT_PREAMBLE, "\nHere are the raw transcripts:"]
    if not transcripts:
        parts.append("\n(Empty input — no transcripts provided.)")
    else:
        for i, t in enumerate(transcripts, 1):
            parts.append(f"\nTRANSCRIPT {i}:\n{t}\n")
    return "\n".join(parts)

def combine_transcripts_with_gpt(transcripts, api_key):
    """Merge transcripts into a single clean, role-tagged transcript (GPT-4o)."""
    prompt = _build_merge_prompt(transcripts)
    client = _client(api_key)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You merge transcripts faithfully. Output ONLY the cleaned, role-tagged transcript."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=4000,
    )
    return resp.choices[0].message.content.strip()

def assess_transcript_quality(final_transcript: str, api_key: str):
    """Evaluate empathy using the Hebrew template (GPT-4o). Returns ONE line."""
    client = _client(api_key)
    user_prompt = EMPATHY_PROMPT_TEMPLATE.format(final_transcript=final_transcript)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Assess empathetic language concisely. Return exactly ONE line."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=600,
    )
    text = resp.choices[0].message.content.strip()
    return re.sub(r'\s*\n+\s*', ' ', text)