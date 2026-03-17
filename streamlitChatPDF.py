import streamlit as st
import os
import glob
import pypdf

# Document Loaders y Splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# Embeddings y Vector Store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

# Prompts y Modelos
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Chains (IMPORTACIÓN SIMPLIFICADA)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(
    page_title="Chatea con tu PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Para instalar estas bibliotecas, usa pip:
# pip install streamlit langchain langchain-huggingface langchain-community faiss langchain-google-genai

st.title("Charla con tu PDF")

# Obtiene la clave API de Google desde los secretos de Streamlit
GOOGLE_API_KEY=st.secrets["GOOGLE_API_KEY"]
# Inicializa el modelo de lenguaje grande (LLM) de Google Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", # Especifica el modelo a usar
    temperature=0, # Controla la aleatoriedad de la salida (0 = determinista)
    max_tokens=None, # Sin límite en el número de tokens generados
    timeout=None, # Sin límite de tiempo
    max_retries=2, # Número máximo de reintentos en caso de error
    api_key=GOOGLE_API_KEY # La clave API
)

# Define una función para generar una base de datos vectorial
def generarBDVectorial(archivoPDF):
    # Crea la ruta para el índice FAISS
    rutaIndice=archivoPDF.replace(".pdf","")
    # Inicializa las incrustaciones de Hugging Face
    embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Especifica el modelo de incrustaciones a usar
            )
    with st.spinner("Generando índices..."):
        # Carga el índice FAISS si existe
        if os.path.exists(rutaIndice):
            vectorstore =  FAISS.load_local(rutaIndice, embeddings, allow_dangerous_deserialization=True)
        # Si no existe, crea uno nuevo
        else:        
            loader = PyPDFLoader(file_path=archivoPDF) # Carga el PDF
            documents = loader.load() # Lee el documento
            text_splitter = CharacterTextSplitter(
                chunk_size=1000, chunk_overlap=30, separator="\n" # Configura el divisor de texto
            )
            docs = text_splitter.split_documents(documents=documents) # Divide el documento en trozos
            
            vectorstore = FAISS.from_documents(docs, embeddings) # Crea el índice FAISS
            vectorstore.save_local(rutaIndice) # Guarda el índice localmente
    retriever = VectorStoreRetriever(vectorstore=vectorstore) # Crea un objeto retriever para acceder al índice

    return retriever # Devuelve el retriever

# Define una función para generar una consulta
def generarConsulta(query,llm,retriever):
    # Define el prompt del sistema
    system_prompt = (
        "Use el contexto dado para responder la pregunta"
        "Si no sabe la respuesta, digamos que no lo sabe"        
        "Use tres oraciones como máximo y mantenga la respuesta concisa"
        "Contexto: {context}"
    )
    # Crea una plantilla de prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    # Crea una cadena para combinar documentos
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # Crea una cadena de recuperación
    chain = create_retrieval_chain(retriever, question_answer_chain)

    return chain.invoke({"input": query}) # Devuelve la respuesta a la consulta

# Inicializa el historial de chat en la sesión de Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Inicializa el nombre del archivo en la sesión de Streamlit
if "archivo" not in st.session_state:
    st.session_state.archivo =""

# Muestra el historial de chat
with st.container():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Obtiene la lista de archivos PDF
archivos =glob.glob("*.pdf")
# Muestra un selector de archivos en la barra lateral
parArchivo = st.sidebar.selectbox('Archivo',options=archivos,index=0)
# Reinicia la aplicación si se selecciona un archivo diferente
if st.session_state.archivo!=parArchivo:
    st.session_state.archivo=parArchivo
    st.session_state.messages = []
    st.rerun()
    
# Genera la base de datos vectorial
retriever = generarBDVectorial(parArchivo)
# Muestra la entrada de chat
prompt=st.chat_input("Qué quieres saber?")

# Procesa la entrada del usuario
if prompt:
    # Muestra el mensaje del usuario
    st.chat_message("user").markdown(prompt)
    # Guarda el mensaje del usuario en el historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:        
        # Genera la respuesta
        chat_completion =generarConsulta(prompt,llm,retriever)                
        # Muestra la respuesta del asistente
        with st.chat_message("assistant"):            
            st.write(chat_completion["answer"])
            full_response=chat_completion["answer"]
            # Muestra el contexto usado para generar la respuesta
            with st.expander("Contexto"):
                for contexto in chat_completion["context"]:
                    st.write(contexto)
        # Guarda la respuesta del asistente en el historial
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    except Exception as e: # Maneja las excepciones
        st.error(e)