import streamlit as st
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuración de página
st.set_page_config(page_title="Chatea con tu PDF", page_icon="📄")
st.title("Charla con tu PDF")

# Clave API desde secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Cache para el modelo de lenguaje (LLM)
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=GOOGLE_API_KEY
    )

llm = load_llm()

# Cache para los embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

embeddings = load_embeddings()

# Función para obtener (o crear) el vectorstore correspondiente a un PDF
@st.cache_resource
def get_vectorstore(pdf_path):
    # Ruta de la carpeta donde se guarda el índice (sin extensión)
    index_path = os.path.splitext(pdf_path)[0]
    
    if os.path.exists(index_path):
        # Cargar índice existente
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # Crear nuevo índice a partir del PDF
        with st.spinner("Procesando PDF y generando índices..."):
            loader = PyPDFLoader(file_path=pdf_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(
                chunk_size=1000, chunk_overlap=30, separator="\n"
            )
            docs = text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(index_path)
    return vectorstore

# Listar archivos PDF en el directorio raíz
pdf_files = glob.glob("*.pdf")
if not pdf_files:
    st.sidebar.warning("No se encontraron archivos PDF en el directorio.")
    st.stop()

# Selector de archivo en la barra lateral
selected_pdf = st.sidebar.selectbox("Selecciona un archivo PDF", pdf_files)

# Reiniciar el chat si cambia el archivo seleccionado
if "last_pdf" not in st.session_state:
    st.session_state.last_pdf = selected_pdf
if st.session_state.last_pdf != selected_pdf:
    st.session_state.messages = []
    st.session_state.last_pdf = selected_pdf
    st.rerun()

# Obtener el vectorstore (cacheado) y el retriever
vectorstore = get_vectorstore(selected_pdf)
retriever = vectorstore.as_retriever()

# Inicializar historial de mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes previos
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Función para generar respuesta usando LCEL
def generar_respuesta(query):
    # Plantilla del prompt
    system_prompt = (
        "Usa el siguiente contexto para responder la pregunta. "
        "Si no sabes la respuesta, simplemente di que no lo sabes. "
        "Usa tres oraciones como máximo y mantén la respuesta concisa.\n\n"
        "Contexto: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Función auxiliar para unir documentos
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Cadena RAG:
    # 1. Obtiene contexto del retriever y lo formatea, y pasa la pregunta sin cambios.
    # 2. Aplica el prompt.
    # 3. Llama al LLM.
    # 4. Parsea la salida a string.
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Recuperamos los documentos por separado para mostrarlos en el expander
    retrieved_docs = retriever.invoke(query)
    answer = rag_chain.invoke(query)
    return answer, retrieved_docs

# Entrada del usuario
user_input = st.chat_input("¿Qué quieres saber?")
if user_input:
    # Mostrar mensaje del usuario
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generar respuesta
    with st.spinner("Pensando..."):
        try:
            answer, docs = generar_respuesta(user_input)
            with st.chat_message("assistant"):
                st.write(answer)
                # Mostrar contexto usado (documentos recuperados)
                with st.expander("Contexto utilizado"):
                    for doc in docs:
                        st.write(doc.page_content)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")