import streamlit as st
import fitz  # PyMuPDF
import tiktoken
from openai import OpenAI

# --- Setup your API Key securely ---
api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else "your_openai_key_here"
client = OpenAI(api_key=api_key)

# --- Streamlit UI ---
st.title("📄 Disaster Management Plan Summarizer")

uploaded_pdf = st.file_uploader("Upload a Disaster Management PDF file", type="pdf")

# --- Utility: Extract Text from PDF ---
def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return " ".join(page.get_text() for page in doc)

# --- Utility: Token-based Chunking ---
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

# --- Utility: GPT Summarizer ---
def summarize_chunk(chunk):
    prompt = f"""
You are an expert in disaster management planning. Summarize the following text into:

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

# --- Main Processing ---
if uploaded_pdf:
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_pdf)

    st.success("Text extracted. Summarizing...")

    chunks = chunk_text(text)
    summaries = []

    for i, chunk in enumerate(chunks):
        with st.spinner(f"Summarizing chunk {i+1}/{len(chunks)}"):
            summary = summarize_chunk(chunk)
            summaries.append(summary)

    final_prompt = "Combine the following summaries into one comprehensive executive summary of the disaster plan:\n\n" + "\n\n".join(summaries)

    with st.spinner("Generating final summary..."):
        final_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.2,
        )
        final_summary = final_response.choices[0].message.content

    st.subheader("📝 Final Summary")
    st.write(final_summary)

    st.download_button("Download Summary", final_summary, file_name="disaster_plan_summary.txt")
