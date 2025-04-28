import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImageOps
from keras.models import load_model
import platform

# 🌸 Configuración de página
st.set_page_config(
    page_title="Detector Cute 🌸",
    page_icon="🎀",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 🌸 Un poquito de CSS para hacerlo aún más tierno y oscuro
st.markdown("""
    <style>
    body {
        background-color: #121212; /* Fondo oscuro */
    }
    .stApp {
        background: #121212; /* Fondo del contenido */
        color: #f8f8f2; /* Color de texto general */
    }
    .title {
        color: #ff69b4; /* Título en rosita */
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    .sidebar .sidebar-content {
        background: #2c2c2c; /* Sidebar oscuro */
    }
    .css-1d391kg, .css-1v3fvcr {
        background-color: #2c2c2c;
        color: #f8f8f2;
    }
    </style>
    """, unsafe_allow_html=True)

# 🌸 Mostrar versión de Python
st.write("🐍 Versión de Python:", platform.python_version())

# 🌸 Cargar modelo
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# 🌸 Título cute
st.markdown("<h1 class='title'>🎀 Reconocimiento de Imágenes Cute 🎀</h1>", unsafe_allow_html=True)

# 🌸 Imagen de portada
image = Image.open('OIG5.jpg')
st.image(image, width=350, caption="✨ ¡Sonríe! Estás a punto de ser detectadx ✨")

# 🌸 Sidebar amoroso
with st.sidebar:
    st.markdown("<h2 style='color: #ff69b4;'>🌸 Bienvenidx 🌸</h2>", unsafe_allow_html=True)
    st.write("Usa un modelo entrenado con Teachable Machine para identificar tus gestos o poses de manera mágica ✨.")

# 🌸 Cámara para capturar imagen
img_file_buffer = st.camera_input("📸 ¡Toma una fotito linda!")

if img_file_buffer is not None:
    # Preparar imagen
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)

    # Normalizar imagen
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Realizar predicción
    prediction = model.predict(data)
    print(prediction)

    # Mostrar resultados súper cute
    if prediction[0][0] > 0.5:
        st.success(f"🌸 ¡Detectado movimiento hacia la Izquierda! 🌸\n\n✨ Probabilidad: {prediction[0][0]:.2f}")
    if prediction[0][1] > 0.5:
        st.success(f"🌸 ¡Detectado movimiento hacia Arriba! 🌸\n\n✨ Probabilidad: {prediction[0][1]:.2f}")
    # Si quieres activar la derecha también:
    # if prediction[0][2] > 0.5:
    #     st.success(f"🌸 ¡Detectado movimiento hacia la Derecha! 🌸\n\n✨ Probabilidad: {prediction[0][2]:.2f}")

# 🌸 Footer cute
st.markdown("---")
st.markdown("<center><h4 style='color: #ff69b4;'>Hecho con mucho 💖 usando Streamlit ✨</h4></center>", unsafe_allow_html=True)

