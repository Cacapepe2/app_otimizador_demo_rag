import os
import streamlit as st
import pandas as pd
from openai import OpenAI
import whisper
import yt_dlp
import tempfile

from lightrag import LightRAG

# Configuração do OpenRouter
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
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
        'format': 'bestaudio',
        'outtmpl': 'audio.%(ext)s',
        'quiet': True,
    }

    if os.path.exists("youtube_cookies.txt"):
        ydl_opts["cookies"] = "youtube_cookies.txt"
],
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
    try:
        df = pd.read_csv(csv_file)
    except UnicodeDecodeError:
        try:
            # Cria novo buffer com o conteúdo original
            csv_file_data = csv_file.getvalue()
            df = pd.read_csv(pd.io.common.BytesIO(csv_file_data), encoding='ISO-8859-1', delimiter=';')
        except Exception as e:
            st.error(f"Erro ao tentar ler o arquivo CSV: {e}")
            df = None

    if df is not None:
        st.success("CSV carregado com sucesso!")
        st.dataframe(df.head())
        for chunk in pd.read_csv(csv_file, chunksize=1000):
            for _, row in chunk.iterrows():
                entrada = " | ".join([f"{col}: {row[col]}" for col in chunk.columns])
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

        # 🔍 Detectar colunas técnicas presentes
        colunas_relevantes = ['índice_taxa', 'ue_medio', 'site', 'bts']
        colunas_presentes = [col for col in colunas_relevantes if any(col in d.lower() for d in docs)]

        if colunas_presentes:
            observacao = "OBSERVAÇÃO: Valores altos em colunas como " + ", ".join(colunas_presentes) + " representam pior desempenho da rede."
        else:
            observacao = ""  # não adiciona se não for relevante

        # 🔧 Prompt final adaptado
        prompt = f"""
Você é um especialista técnico em redes móveis.

DADOS:
{contexto}

PERGUNTA:
{user_question}

{observacao}
"""

        try:
            response = client.chat.completions.create(
    model="deepseek/deepseek-chat-v3-0324:free",
    messages=[{"role": "user", "content": prompt}]
)

if response and hasattr(response, "choices") and response.choices:
    st.markdown("### ✅ Resposta da IA:")
    st.success(response.choices[0].message.content)
else:
    st.warning("⚠️ A resposta da IA veio vazia ou malformada.")
        except Exception as e:
            st.error(f"Erro ao consultar a IA: {e}")
else:
    st.info("📥 Envie arquivos ou links para começar.")
