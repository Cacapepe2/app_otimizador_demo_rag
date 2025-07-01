import os
import streamlit as st
import pandas as pd
import openai
import whisper
import yt_dlp
import tempfile

from lightrag import LightRAG

# Configuração do OpenRouter
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

st.set_page_config(page_title="RAG Otimizador Técnico", layout="wide")

# Proteção com senha
st.markdown("### 🔐 Acesso Restrito")
senha = st.text_input("Digite a senha para acessar:", type="password")

if senha != os.getenv("APP_SENHA"):
    st.warning("Acesso negado. Digite a senha correta para continuar.")
    st.stop()

st.title("📡 Otimizador Inteligente com RAG")
st.markdown("Envie planilhas, documentos ou links de vídeo e pergunte sobre sua rede.")

# 🔊 Carregamento do modelo Whisper apenas sob demanda
@st.cache_resource(show_spinner="🔊 Carregando modelo Whisper...")
def carregar_modelo_whisper():
    return whisper.load_model("tiny")

# 📺 Transcrição YouTube (otimizada)
def transcrever_audio_do_youtube(url):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = None
                for f in os.listdir(temp_dir):
                    if f.endswith(".mp3"):
                        filename = os.path.join(temp_dir, f)
                        break

            if filename:
                model = carregar_modelo_whisper()
                result = model.transcribe(filename)
                return result["text"]
            else:
                return "Erro ao processar o áudio."
    except Exception as e:
        return f"Erro: {e}"

# 📚 Coleta de documentos
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
        transcricao = transcrever_audio_do_youtube(youtube_link)
        st.success("Transcrição concluída!")
        st.text_area("📝 Texto extraído do vídeo:", transcricao, height=200)
        docs.append(transcricao)

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
                model="openchat/openchat-3.5-0106",  # Pode trocar por outro modelo do OpenRouter
                messages=[{"role": "user", "content": prompt}]
            )
            st.markdown("### ✅ Resposta da IA:")
            st.success(response['choices'][0]['message']['content'])
        except Exception as e:
            st.error(f"Erro ao consultar a IA: {e}")
else:
    st.info("📥 Envie arquivos ou links para começar.")
