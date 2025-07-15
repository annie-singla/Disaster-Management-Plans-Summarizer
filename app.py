# 🔹 Step 2: Import required modules
import os
from PyPDF2 import PdfReader
from openai import AzureOpenAI
from google.colab import files
from IPython.display import display, Markdown

# 🔹 Step 3: Set your Azure OpenAI credentials
API_VERSION = "2024-12-01-preview"
ENDPOINT = "https://ai-gershonavi3238ai913103279029.openai.azure.com/"
DEPLOYMENT = "gpt-35-turbo"
API_KEY = "ETteF1ZBx3xMaD0aE5zkJQTWEMiXH6bWK9HohX0z7Qmf9fqFRNgZJQQJ99BDACHYHv6XJ3w3AAAAACOGvgwJ"

# 🔹 Step 4: Initialize OpenAI client
client = AzureOpenAI(
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
)

# 🔹 Step 5: Define helper functions
def split_text(text, max_tokens=3000):
    paragraphs = text.split('\n')
    chunks, current_chunk = [], ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_tokens:
            current_chunk += para + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def summarize_text(text, chunk_id):
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are an expert in summarizing government disaster management plans."},
            {"role": "user", "content": f"Summarize this section:\n\n{text}"}
        ],
        max_tokens=1024,
        temperature=0.7,
        top_p=1.0
    )
    return response.choices[0].message.content.strip()
