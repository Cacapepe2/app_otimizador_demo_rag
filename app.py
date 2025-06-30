import os
import streamlit as st
import pandas as pd
import openai
import whisper
import yt_dlp

from lightrag import LightRAG

# Configuração do OpenRouter
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

st.set_page_config(page_title="RAG Otimizador Técnico", layout="wide")

# Proteção com senha
st.markdown("### 🔐 Acesso Restrito")
senha = st.text_input("Digite a senha para acessar:", type="password")

if senha != os.getenv("APP_SENH"):
    st.warning("Acesso negado. Digite a senha correta para continuar.")
    st.stop()
st.title("📡 Otimizador Inteligente com RAG")
st.markdown("Envie planilhas, documentos ou links de vídeo e pergunte sobre sua rede.")

# Carregando modelo de áudio
model = whisper.load_model("base")

# Transcrição YouTube
@st.cache_data
def transcrever_audio_do_youtube(url):
    with yt_dlp.YoutubeDL({'format': 'bestaudio', 'outtmpl': 'audio.%(ext)s'}) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    result = model.transcribe(filename)
    return result["text"]

docs = []

# 📊 Upload CSV
csv_file = st.file_uploader("📈 Envie um arquivo CSV com os dados", type=["csv"])
if csv_file:
    df = pd.read_csv(csv_file)
    st.success("CSV carregado com sucesso!")
    st.dataframe(df.head())

    for _, row in df.iterrows():
        entrada = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(entrada)

# 📄 Upload de documentos
uploaded_docs = st.file_uploader("📄 Envie arquivos .txt ou .pdf", type=["txt", "pdf"], accept_multiple_files=True)
if uploaded_docs:
    for file in uploaded_docs:
        content = file.read().decode("utf-8", errors="ignore")
        docs.append(content)

# 📺 Link do YouTube
youtube_link = st.text_input("🎥 Cole um link de vídeo do YouTube para transcrição automática:")
if youtube_link:
    with st.spinner("Transcrevendo áudio do vídeo..."):
        try:
            transcricao = transcrever_audio_do_youtube(youtube_link)
            st.success("Transcrição concluída!")
            st.text_area("📝 Texto extraído do vídeo:", transcricao, height=200)
            docs.append(transcricao)
        except Exception as e:
            st.error(f"Erro ao transcrever vídeo: {e}")

# RAG e Pergunta
if docs:
    rag = LightRAG(docs)
    rag.create_index()

    user_question = st.text_input("🧠 Faça uma pergunta técnica:")
    if user_question:
        contexto = "\n".join(docs[:15])
        prompt = f"""
Você é um especialista técnico em redes móveis.

DADOS:
{contexto}

PERGUNTA:
{user_question}

OBSERVAÇÃO:
Valores altos (ex: 10) em colunas como 'índice_taxa' e 'ue_medio' representam pior desempenho da rede.
Use esse conhecimento para recomendar ações de melhoria.
"""

        try:
            response = openai.ChatCompletion.create(
                model="openchat/openchat-3.5-0106",
                messages=[{"role": "user", "content": prompt}]
            )
            st.markdown("### ✅ Resposta da IA:")
            st.success(response['choices'][0]['message']['content'])
        except Exception as e:
            st.error(f"Erro ao consultar a IA: {e}")
