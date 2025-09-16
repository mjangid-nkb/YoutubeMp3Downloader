import os
import tempfile
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

import yt_dlp
import imageio_ffmpeg as ffmpeg  # For ffmpeg in Streamlit Cloud

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()
ASR_MODEL = os.getenv("HUGGINGFACE_ASR_REPO", "openai/whisper-small")
LLM_MODEL = os.getenv("HUGGINGFACE_LLM_REPO", "google/gemma-2-2b-it")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Streamlit page config
st.set_page_config(layout="centered", page_title="YouTube → MP3")
st.title("🎵 YouTube → MP3 Converter")

url = st.text_input("Enter YouTube URL")


def download_audio_mp3(youtube_url, out_dir=None):
    """Download YouTube video as MP3 using yt_dlp with static ffmpeg."""
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
        'ffmpeg_location': ffmpeg.get_ffmpeg_exe(),
        'nocheckcertificate': True,
        'noplaylist': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
        'geo_bypass': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info.get('id')
        mp3_path = os.path.join(out_dir, f"{video_id}.mp3")
        if os.path.exists(mp3_path):
            return mp3_path
        # Fallback: first mp3 in directory
        mp3s = list(Path(out_dir).glob("*.mp3"))
        if mp3s:
            return str(mp3s[0])
    raise FileNotFoundError("Failed to download mp3")


def transcribe_with_huggingface(mp3_path, model_id):
    """Transcribe MP3 to text using Hugging Face InferenceClient."""
    client = InferenceClient(token=HUGGINGFACE_TOKEN)
    with open(mp3_path, "rb") as f:
        result = client.audio_to_text(model=model_id, file=f)
    return result.get("text", "")


def summarize_with_llm(text, repo_id):
    """Summarize transcript using LangChain HuggingFace LLM."""
    llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=HUGGINGFACE_TOKEN)
    chat = ChatHuggingFace(llm=llm)
    prompt = f"Summarize the following transcript in 5 bullet points:\n\n{text[:4000]}"
    resp = chat.invoke(prompt)
    return resp.content


# Streamlit button action
if st.button("Start") and url.strip():
    with st.spinner("Downloading MP3..."):
        try:
            mp3_file = download_audio_mp3(url)
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()

    st.success(f"Downloaded: {mp3_file}")
    st.audio(mp3_file)

    with st.spinner("Transcribing..."):
        try:
            transcript_text = transcribe_with_huggingface(mp3_file, ASR_MODEL)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            st.stop()

    st.subheader("Transcript")
    st.write(transcript_text[:2000])

    with st.spinner("Summarizing..."):
        try:
            summary = summarize_with_llm(transcript_text, LLM_MODEL)
        except Exception as e:
            st.error(f"Summarization failed: {e}")
            st.stop()

    st.subheader("LLM Summary")
    st.write(summary)
