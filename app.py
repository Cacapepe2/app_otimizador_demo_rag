import streamlit as st
import os
import pandas as pd
from lightrag import RAGPipeline
from lightrag.embeddings import HuggingFaceEmbedding
from lightrag.llms import OpenRouterLLM
from lightrag.vectorstores import FAISS
from lightrag.document_loaders import SimpleDirectoryReader

# Autenticação com senha (segura com variável de ambiente)
def autenticar():
    senha_correta = os.getenv("APP_SENHA", "sem_senha")
    senha = st.text_input("🔐 Digite a senha para acessar:", type="password")
    if senha != senha_correta:
        st.warning("Senha incorreta ou ausente.")
        st.stop()

autenticar()

st.title("📄 RAG Técnico para Telecomunicações")

# Upload de arquivos
st.subheader("📤 Envie seus documentos técnicos e planilhas com dados")
uploaded_files = st.file_uploader("Arraste ou envie arquivos PDF, TXT, CSV", accept_multiple_files=True)

dados_df = None

if uploaded_files:
    os.makedirs("data", exist_ok=True)
    for file in uploaded_files:
        file_path = os.path.join("data", file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

        # Se for CSV, processa como DataFrame também
        if file.name.endswith(".csv"):
            try:
                df_temp = pd.read_csv(file_path)
                if dados_df is None:
                    dados_df = df_temp
                else:
                    dados_df = pd.concat([dados_df, df_temp], ignore_index=True)
            except Exception as e:
                st.warning(f"Erro ao ler {file.name} como CSV: {e}")

    st.success("Documentos salvos com sucesso!")

    # Inicializa o pipeline RAG
    st.info("🔄 Indexando os documentos...")
    documents = SimpleDirectoryReader("data").load()

    embed_model = HuggingFaceEmbedding("BAAI/bge-base-en-v1.5")
    vectorstore = FAISS.from_documents(documents, embed_model)

    llm = OpenRouterLLM(
        api_key=os.getenv("OPENROUTER_API_KEY"),  # variável de ambiente segura
        model="openai/gpt-3.5-turbo"
    )

    pipeline = RAGPipeline(retriever=vectorstore.as_retriever(), llm=llm)
    st.success("Documentos prontos para consulta.")

    # Interface de perguntas
    st.subheader("🔎 Faça perguntas sobre os documentos e dados carregados")
    query = st.text_input("Digite sua pergunta técnica:")
    if query:
        with st.spinner("Consultando documentos e dados..."):
            contexto_extra = ""
            if dados_df is not None:
                colunas_criticas = [col for col in dados_df.columns if any(palavra in col.lower() for palavra in ["taxa", "ue", "indice"])]
                dados_criticos = dados_df.copy()
                for col in colunas_criticas:
                    try:
                        dados_criticos = dados_criticos[dados_criticos[col] >= 8]
                    except:
                        pass  # ignora colunas não numéricas
                contexto_extra = dados_criticos.to_string(index=False)
            result = pipeline.invoke(query + "\n\nDados críticos:\n" + contexto_extra)
            st.markdown("### 🧠 Resposta")
            st.write(result)

# Exemplo de perguntas sugeridas
with st.expander("💡 Exemplos de perguntas técnicas"):
    st.markdown("""
    - Quais sites com índice_taxa ou ue_medio igual a 10 precisam de atenção?
    - Quais são os piores valores registrados por município?
    - O que fazer com um site com alto ue_medio e índice de retransmissão?
    - Existe alguma recomendação nos documentos para valores críticos?
    - Como melhorar um site com valores ruins em várias colunas?
    """)
