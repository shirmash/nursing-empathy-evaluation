import openai
import os
import streamlit as st

def combine_transcripts_with_gpt(transcripts, api_key, model="gpt-3.5-turbo"):
    prompt = (
        "You are an expert at summarizing and merging transcripts. "
        "Combine the following transcripts into a single, coherent transcript, removing duplicates and making it readable. "
        "Keep all important information.\n\n"
    )
    joined = "\n\n".join(transcripts)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": joined}
    ]
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=4096,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def assess_transcript_quality(transcript, api_key, model="gpt-3.5-turbo"):
    prompt = (
        "You are an expert in transcript quality assessment. "
        "Given the following transcript, provide a short quality assessment (accuracy, completeness, clarity, and any issues):\n\n"
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": transcript}
    ]
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=512,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()
