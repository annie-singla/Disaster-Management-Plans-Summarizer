# requirements.txt:
# streamlit
# openai>=1.0.0
# PyMuPDF
# tiktoken

import streamlit as st
import fitz  # PyMuPDF
import openai
import os
import tiktoken
from openai import OpenAI

# 🔑 Load OpenAI API Key from Streamlit secrets or hardcoded (NOT recommended)
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else "YOUR_API_KEY_HERE"
client = OpenAI(api_key=openai_api_key)

st.title("📄 Disaster Management Plan Summarizer")

# ---------- Utilities ----------

def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        return " ".join(page.get_text() for page in doc)

def chunk_text(text, max_tokens=8000):
    enc = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks = []
    chunk = []
    tokens = 0
    for word in words:
        word_tokens = len(enc.encode(word))
        if tokens + word_tokens > max_tokens:
            chunks.append(" ".join(chunk))
            chunk = [word]
            tokens = word_tokens
        else:
            chunk.append(word)
            tokens += word_tokens
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def summarize_chunk(chunk):
    prompt = f"""
You are an expert in emergency planning. Summarize the following disaster management content into:
1. Executive Summary
2. Key Actionable Plans
3. Stakeholders
4. Preparedness, Response, and Recovery Points

Text:
{chunk}
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content

# ---------- UI Flow ----------

uploaded_pdf = st.file_uploader("Upload Disaster Management PDF", type="pdf")

if uploaded_pdf:
    st.info("Extracting text from PDF...")
    text = extract_text_from_pdf(uploaded_pdf)

    st.success("PDF loaded. Splitting and summarizing...")

    chunks = chunk_text(text)
    summaries = []
    for i, chunk in enumerate(chunks):
        with st.spinner(f"Summarizing chunk {i+1}/{len(chunks)}..."):
            summary = summarize_chunk(chunk)
            summaries.append(summary)

    # Final summary prompt
    final_prompt = "Combine and condense the following summaries into a cohesive executive summary of the disaster plan:\n\n" + "\n\n".join(summaries)

    final_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.2,
    )
    final_summary = final_response.choices[0].message.content

    st.subheader("📝 Final Summary")
    st.write(final_summary)

    st.download_button("Download Summary", final_summary, file_name="summary.txt")
