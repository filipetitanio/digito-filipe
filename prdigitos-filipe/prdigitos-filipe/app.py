import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from streamlit_drawable_canvas import st_canvas
import plotly.express as px
import pandas as pd

# Configuração da página
st.set_page_config(page_title="Reconhecimento de Dígitos Manuscritos - Filipe Tchivela", layout="wide")

# Inicializando o estado da sessão para histórico de previsões
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Carregar o modelo localmente
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('prdigitos-filipe/mnist_cnn_model.h5')
    return model

model = load_model()

# Função para pré-processar a imagem
def preprocess_image(image):
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0
    image_array = 1.0 - image_array  # Inverter (fundo preto, dígito branco)
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array

# Título e descrição
st.title("📝 Reconhecimento de Dígitos Manuscritos")
st.markdown("""
**Desenvolvido por Filipe Tchivela**  
Esta aplicação utiliza uma Rede Neural Convolucional (CNN) treinada no dataset MNIST para reconhecer dígitos manuscritos com acurácia ≥99%.  
Desenhe um dígito (0-9) no canvas ou faça upload de uma imagem para testar o modelo!
""")

# Layout com três colunas
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("Desenhe um Dígito")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Faça Upload de uma Imagem")
    uploaded_file = st.file_uploader("Escolha uma imagem (PNG/JPG)", type=["png", "jpg", "jpeg"])

with col3:
    st.subheader("Instruções")
    st.markdown("""
    - **Desenhar**: Use o canvas à esquerda para desenhar um dígito (0-9) com fundo preto e traço branco.
    - **Upload**: Envie uma imagem de um dígito (idealmente 28x28, fundo preto, dígito branco).
    - **Prever**: Clique no botão abaixo para obter a previsão.
    - **Histórico**: Veja as últimas previsões na seção abaixo.
    """)

# Botão para prever
if st.button("Prever Dígito"):
    if canvas_result.image_data is not None or uploaded_file is not None:
        if canvas_result.image_data is not None:
            image = Image.fromarray(canvas_result.image_data.astype('uint8'))
        else:
            image = Image.open(uploaded_file)
        
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction)
        probabilities = prediction[0]

        # Exibir resultados
        st.subheader("Resultado da Previsão")
        st.write(f"**Dígito Predito:** {predicted_digit}")
        
        # Gráfico de barras com probabilidades
        prob_df = pd.DataFrame({
            'Dígito': [str(i) for i in range(10)],
            'Probabilidade': probabilities
        })
        fig = px.bar(prob_df, x='Dígito', y='Probabilidade', title='Probabilidades por Classe',
                     color='Probabilidade', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

        # Visualizar imagem processada
        st.image(processed_image.reshape(28, 28), caption="Imagem Processada", width=150)

        # Adicionar ao histórico
        st.session_state.prediction_history.append({
            'Dígito Predito': predicted_digit,
            'Probabilidade Máxima': float(probabilities[predicted_digit]),
            'Imagem': processed_image.reshape(28, 28)
        })
    else:
        st.warning("Por favor, desenhe um dígito ou faça upload de uma imagem!")

# Exibir histórico de previsões
if st.session_state.prediction_history:
    st.markdown("---")
    st.subheader("Histórico de Previsões")
    for i, pred in enumerate(st.session_state.prediction_history[-5:]):
        col_hist1, col_hist2 = st.columns([1, 3])
        with col_hist1:
            st.image(pred['Imagem'], caption=f"Previsão {i+1}", width=100)
        with col_hist2:
            st.write(f"**Dígito Predito:** {pred['Dígito Predito']}")
            st.write(f"**Probabilidade Máxima:** {pred['Probabilidade Máxima']:.4f}")

# Seção Sobre Mim
st.markdown("---")
st.header("Sobre Mim")
st.markdown("""
**Nome:** Filipe Tchivela  
**Curso:** Ciência da Computação, 3º Ano  
**Universidade:** Mandume  
**Descrição:** Este projeto foi desenvolvido para demonstrar o uso de redes neurais convolucionais no reconhecimento de dígitos manuscritos, alcançando acurácia superior a 99%. A aplicação Streamlit oferece uma interface interativa para testes em tempo real, ideal para apresentações acadêmicas.
""")
