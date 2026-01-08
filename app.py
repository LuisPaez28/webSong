import streamlit as st
import librosa
import numpy as np
import whisper
import os
import subprocess
import imageio_ffmpeg
from docx import Document
from io import BytesIO

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Traductor de Canciones", layout="wide")

# --- NUEVA FUNCIÓN CON CACHÉ (Ponla aquí, antes del bloque principal) ---
@st.cache_resource
def load_whisper_model(model_size):
    """Carga el modelo una sola vez y lo guarda en memoria dinamica"""
    return whisper.load_model(model_size, device="cpu")

CHORD_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def get_chord(chroma):
    idx = np.argmax(chroma)
    suffix = "" if chroma[(idx + 4) % 12] > chroma[(idx + 3) % 12] else "m"
    return f"{CHORD_NAMES[idx]}{suffix}"

# --- INTERFAZ ---
st.title("Recuperador de acordes y letras de canciones")

st.sidebar.header("Configuración")
modelo_seleccionado = st.sidebar.selectbox("Modelo de Whisper", ["base (equilibrado)", "tiny (rapido)", "small (exacto)"], index=0)

uploaded_file = st.file_uploader("Sube tu video o audio", type=["mp4", "mov", "wav", "mp3"])

if uploaded_file is not None:
    # 1. Creamos el botón
    if st.button("Comenzar Análisis"):
        with st.spinner("Cargando modelo y procesando..."):
            
            # --- AQUÍ USAMOS LA FUNCIÓN DE CACHÉ ---
            model = load_whisper_model(modelo_seleccionado)
            
            # Guardar archivo temporal
            temp_input = "temp_input" + os.path.splitext(uploaded_file.name)[1]
            with open(temp_input, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Convertir a Audio
            audio_path = "temp_audio.wav"
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run([ffmpeg_exe, "-y", "-i", temp_input, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path], capture_output=True)

            # Transcribir
            st.info("Transcribiendo letra...")
            result = model.transcribe(audio_path, fp16=False)

            # Acordes
            st.info("Analizando acordes...")
            y, sr = librosa.load(audio_path, sr=None)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=1024)
            times = librosa.times_like(chroma, sr=sr, hop_length=1024)

            # Generar Word y Mostrar resultados
            doc = Document()
            doc.add_heading('Cancionero Recuperado', 0)
            
            for seg in result['segments']:
                idx_start = np.searchsorted(times, seg['start'])
                idx_end = np.searchsorted(times, seg['end'])
                
                # Homologación (máximo 4 acordes)
                paso = max(1, (idx_end - idx_start) // 4)
                acordes_list = []
                ultimo = ""
                for i in range(idx_start, idx_end, paso):
                    act = get_chord(chroma[:, i])
                    if act != ultimo:
                        acordes_list.append(act)
                        ultimo = act
                
                acordes_str = "   ".join(acordes_list)
                tiempo = f"[{int(seg['start']//60):02d}:{int(seg['start']%60):02d}]"
                texto = seg['text'].strip()

                st.markdown(f"**{tiempo} {acordes_str}**")
                st.write(texto)
                
                p = doc.add_paragraph()
                run = p.add_run(f"{tiempo}  {acordes_str}")
                run.bold = True
                doc.add_paragraph(texto)

            # Descarga
            buffer = BytesIO()
            doc.save(buffer)
            st.download_button(
                label="📄 Descargar",
                data=buffer.getvalue(),
                file_name="cancionero.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            
            # Limpiar archivos
            if os.path.exists(temp_input): os.remove(temp_input)
            if os.path.exists(audio_path): os.remove(audio_path)