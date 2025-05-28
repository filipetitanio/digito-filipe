# Reconhecimento de Dígitos Manuscritos com CNN

Este projeto implementa um sistema de reconhecimento de dígitos manuscritos usando o dataset MNIST e uma rede neural convolucional (CNN). A aplicação Streamlit permite testar o modelo com desenhos ou imagens enviadas.

## Autoria
- **Nome**: Filipe Tchivela
- **Curso**: Ciência da Computação, 3º Ano
- **Universidade**: Mandume
- **Data**: Maio de 2025

## Estrutura do Repositório
- `app.py`: Código da aplicação Streamlit.
- `mnist_cnn_model.h5`: Modelo treinado.
- `requirements.txt`: Dependências do projeto.
- `relatorio_projeto.tex`: Relatório detalhado em LaTeX.

## Como Executar Localmente
1. Clone o repositório:
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd mnist-digit-recognition
   ```
2. Crie um ambiente virtual e instale as dependências:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```
3. Execute a aplicação:
   ```bash
   streamlit run app.py
   ```
4. Acesse `http://localhost:8501` no navegador.

## Como Fazer Deploy
1. Faça push do repositório para o GitHub.
2. Acesse [Streamlit Community Cloud](https://share.streamlit.io/).
3. Crie uma nova aplicação, selecionando este repositório e o arquivo `app.py`.
4. Aguarde o deploy e acesse a URL fornecida.

## Requisitos
- Python 3.8+
- Bibliotecas listadas em `requirements.txt`
- Modelo treinado (`mnist_cnn_model.h5`)

## Como Usar
- Desenhe um dígito (0-9) no canvas ou faça upload de uma imagem (PNG/JPG).
- Clique em "Prever Dígito" para ver o resultado.
- A previsão inclui o dígito predito e as probabilidades por classe.

## Resultados
- Acurácia: ≥99% no conjunto de teste (`mnist_test.csv`).
- Interface: Streamlit com canvas interativo e upload de imagens.
- Relatório: Detalhado em LaTeX, com matriz de confusão e análise de erros.