import os
import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from utils.chart_generator import generate_chart

# load_dotenv()

# client = OpenAI(
#     api_key = os.getenv("OPENAI_API_KEY")
# )

# openai.api_key = st.secrets["OPENAI_API_KEY"]


st.title("PlotBot")
st.markdown("This chatbot can generate charts based on your descriptions.")

user_input = st.text_input("Describe the chart you want to create: ", "")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "message" not in st.session_state:
    st.session_state.message = []

for message in st.session_state.message:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("what's up?"):
    st.session_state.message.append({"role": "user", "content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model = st.session_state["openai_model"],
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.message
            ],
            stream = True
        ): 
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + " ")
        message_placeholder.markdown(full_response)
    
    st.session_state.message.append({"role": "assistant", "content": full_response})