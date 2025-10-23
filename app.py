import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# ==============================
# CONFIGURACIÓN GENERAL
# ==============================
st.set_page_config(page_title="Asistente Futbolero ⚽", page_icon="⚽", layout="wide")

# --- CABECERA PRINCIPAL ---
st.markdown("""
# ⚽ Asistente Futbolero: Estrategias, Tácticas y Análisis
Bienvenido al **Asistente Futbolero**, tu analista de confianza para entender jugadas, tácticas o estrategias del fútbol profesional.  
Sube un PDF con análisis de equipos, manuales técnicos o reglas del juego y hazle preguntas como si hablaras con un **entrenador profesional**.  
""")

st.caption(f"Versión de Python: {platform.python_version()}")

# --- IMAGEN DE PORTADA ---
try:
    image = Image.open('futbol.jpg')
    st.image(image, width=400, caption="El fútbol no solo se juega, también se estudia ⚽")
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📋 Panel del Director Técnico")
    st.info("Carga tu documento táctico o guía y analiza jugadas, estilos de juego o estrategias.")
    st.markdown("**Consejo:** Ideal para entrenadores, estudiantes de deporte o fanáticos del fútbol analítico.")

# --- CLAVE DE API ---
ke = st.text_input('🔑 Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar.")

# --- CARGA DEL PDF ---
pdf = st.file_uploader("📄 Carga tu documento de fútbol (estrategias, reglas o análisis)", type="pdf")

# ==============================
# PROCESAMIENTO DEL DOCUMENTO
# ==============================
if pdf is not None and ke:
    try:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.success(f"Texto extraído correctamente. Total de {len(text)} caracteres de contenido futbolero 📝")

        # Dividir en fragmentos
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=25,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.info(f"Documento dividido en {len(chunks)} secciones de análisis táctico.")

        # Crear base de conocimiento
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # ==============================
        # PREGUNTAS RÁPIDAS
        # ==============================
        st.subheader("🏟️ Preguntas rápidas del entrenador")
        st.markdown("Selecciona una pregunta o escribe la tuya:")

        col1, col2, col3 = st.columns(3)
        user_question = None
        with col1:
            if st.button("¿Cómo mejorar la posesión del balón?"):
                user_question = "¿Qué estrategias ayudan a mejorar la posesión del balón?"
            if st.button("Errores comunes en defensa"):
                user_question = "¿Cuáles son los errores defensivos más comunes y cómo corregirlos?"
        with col2:
            if st.button("Cómo presionar alto"):
                user_question = "¿Cómo se implementa una presión alta efectiva?"
            if st.button("Transiciones rápidas"):
                user_question = "¿Qué tácticas sirven para hacer transiciones rápidas en ataque?"
        with col3:
            if st.button("Cómo mantener la moral del equipo"):
                user_question = "¿Qué métodos ayudan a mantener la moral del equipo en partidos difíciles?"

        # Campo para pregunta personalizada
        user_custom_question = st.text_area("📢 O haz tu propia pregunta:", placeholder="Ejemplo: ¿Qué formación es más efectiva contra un 4-3-3?")
        if user_custom_question.strip():
            user_question = user_custom_question

        # ==============================
        # RESPUESTA DEL ASISTENTE
        # ==============================
        if user_question:
            st.markdown("---")
            st.markdown("### 🧠 Análisis del Asistente Futbolero:")

            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI(temperature=0.2, model_name="gpt-4o")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            st.success(response)

    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar.")
else:
    st.info("📘 Carga un documento de fútbol para comenzar el análisis táctico.")
