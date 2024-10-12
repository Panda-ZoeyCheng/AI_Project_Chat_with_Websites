import streamlit as st
import openai
import pandas as pd
from utils.chart_generator import generate_chart

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("PlotBot")
st.markdown("This chatbot can generate charts based on your descriptions.")

user_input = st.text_input("Describe the chart you want to create: ", "")

if user_input:
    response = openai.Completion.create(
        engine = "text-davinci-004",
        prompt = f"Analyze this request and generate Python code for a chart: {user_input}",
        temperature = 0.5,
        max_tokens = 150,
    )
    code = response.choices[0].text.strip()

    st.code(code, language="python")

    try:
        exec(code)
    except Exception as e:
        st.error(f"Error generating chart: {e}")