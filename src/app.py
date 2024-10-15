import os
import streamlit as st
from streamlit_chat import message
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="PlotBot", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>PlotBot</h1>", unsafe_allow_html=True)

upload_file = st.file_uploader("Upload CSV file", type="csv")
if upload_file is not None:
    df = pd.read_csv(upload_file)
    schema = df.dtypes.to_string()
    query = None

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "graph" not in st.session_state:
    st.session_state["graph"] = []
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant"}
    ]

st.sidebar.title("Sidebar")
counter_placeholder = st.sidebar.empty()
clear_button = st.sidebar.button("Clear conversation", key="clear")

model_name = 'gpt-3.5-turbo'

if clear_button:
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant"}
    ]

def create_plot_from_response(response_text):
    [eval(statement) for statement in response_text.split("\n")]

def generate_response(query):
    st.session_state["messages"].append({"role": "user", "content": query})
    prompt = f""" You are a matplotlib and seaborn expert.
    You answer questions related to data
    Your also write code for creating visualization charts based on input query.
    Image name is always 'plot.png'
    You have a pandas DataFrame called df that contains data with the following schema:
    {schema}

    use appropriate figure size
    Do not add imports
    Do not return anything but code
    If plot is required then follow these steps
        To create charts you return in the similar format:
        plt.figure(figsize=(n,n))
        sns.<plot_kind>()
        plt.xlabel(<categories>)
        plt.ylabel(<values>)
        plt.title(<suitable-chart-name>)
        plt.savefig(<image-name>)
    Else give the answser to the query in code
    
    Query: {query}

    Result:"""


    # completion_params = {"temperature": 0.7, "max_tokens":250, "model": "text-davinci-003"}

    # completion = openai.Completion.create(prompt=full_prompt, **completion_params)

    response = openai.chat.completions.create(
        model = model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        stream = True,
        temperature = 0.7,
        max_tokens = 250
    )

    full_response = ""
    for chunk in response:
        chunk_content = chunk["choices"][0].get("delta, {}").get("content", "")
        full_response += chunk_content
        st.write(chunk_content)

    # global response

    # response = completion["choices"][0]["text"].strip("\n")

    if "plt" in full_response:
        create_plot_from_response(full_response)
        st.session_state["graph"].append(Image.open("/plot.png"))
        full_response = "Chart is shown below"
    else:
        st.session_state["graph"].append(None)
        # response = response
    
    st.session_state["messages"].append({"role": "assistant", "content": full_response})

response_container = st.container()
container = st.container()

with container:
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_area("You:", key="input", height=70)
        submit_button = st.form_submit_button(label="send")

    if submit_button and user_input:
        output = generate_response(user_input)
        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(output)
    

if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))

            try:
                st.image(st.session_state["graph"][i], width=10)
            except:
                pass
