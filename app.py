import os
import tempfile
import streamlit as st
import yt_dlp
from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()
ASR_MODEL = os.getenv("HUGGINGFACE_ASR_REPO", "openai/whisper-small")
LLM_MODEL = os.getenv("HUGGINGFACE_LLM_REPO", "google/gemma-2-2b-it")

# you’ll also need your token in env
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.set_page_config(layout="centered", page_title="YouTube → MP3")

st.title("🎵 YouTube → MP3")

url = st.text_input("Enter YouTube URL")

def download_audio_mp3(youtube_url, out_dir=None):
    if out_dir is None:
        out_dir = tempfile.mkdtemp()
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(out_dir, '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info.get('id')
        mp3_path = os.path.join(out_dir, f"{video_id}.mp3")
        if os.path.exists(mp3_path):
            return mp3_path
        mp3s = list(Path(out_dir).glob("*.mp3"))
        if mp3s:
            return str(mp3s[0])
    raise FileNotFoundError("Failed to download mp3")

def transcribe_with_huggingface(mp3_path, model_id):
    from huggingface_hub import InferenceClient
    client = InferenceClient(model=model_id, token=HUGGINGFACE_TOKEN)
    # Use automatic_speech_recognition, pass path
    result = client.automatic_speech_recognition(mp3_path)
    # sometimes result has .get("text")
    return result.get("text", "")

def summarize_with_llm(text, repo_id):
    llm = HuggingFaceEndpoint(repo_id=repo_id)
    chat = ChatHuggingFace(llm=llm)
    prompt = f"Summarize the following transcript in 5 bullet points:\n\n{text[:4000]}"
    resp = chat.invoke(prompt)
    return resp.content

if st.button("Start") and url.strip():
    with st.spinner("Downloading..."):
        mp3_file = download_audio_mp3(url)
    st.success(f"Downloaded: {mp3_file}")
    st.audio(mp3_file)

    with st.spinner("Transcribing with Hugging Face..."):
        transcript_text = transcribe_with_huggingface(mp3_file, ASR_MODEL)
    st.subheader("Transcript")
    st.write(transcript_text[:2000])

    with st.spinner("Summarizing with Hugging Face..."):
        summary = summarize_with_llm(transcript_text, LLM_MODEL)
    st.subheader("LLM Summary")
    st.write(summary)
