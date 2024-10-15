# import os
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import openai
# from dotenv import load_dotenv
# from utils.chart_generator import generate_chart
# import matplotlib.pyplot as plt
# from io import BytesIO

# # load_dotenv()

# # client = OpenAI(
# #     api_key = os.getenv("OPENAI_API_KEY")
# # )

# openai.api_key = st.secrets["OPENAI_API_KEY"]


# st.title("PlotBot")
# st.markdown("This chatbot can generate charts based on your descriptions.")


# # if "openai_model" not in st.session_state:
# #     st.session_state["openai_model"] = "gpt-3.5-turbo"

# # if "message" not in st.session_state:
# #     st.session_state.message = []

# # for message in st.session_state.message:
# #     with st.chat_message(message["role"]):
# #         st.markdown(message["content"])

# # if prompt := st.chat_input("Please enter your description ..."):
# #     st.session_state.message.append({"role": "user", "content": prompt})
# #     with st.chat_message("user"):
# #         st.markdown(prompt)
    
# #     with st.chat_message("assistant"):
# #         message_placeholder = st.empty()
# #         full_response = ""
# #         for response in openai.chat.completions.create(
# #             model = st.session_state["openai_model"],
# #             messages = [
# #                 {"role": m["role"], "content": m["content"]}
# #                 for m in st.session_state.message
# #             ],
# #             stream = True
# #         ): 
# #             if response.choices[0].delta.content is not None:
# #                 full_response += response.choices[0].delta.content
# #                 message_placeholder.markdown(full_response + " ")

# #         message_placeholder.markdown(full_response)
# #         # message_placeholder.markdown(f"Generated code:\n\n```python\n{full_response}\n```")

# #         # try:
# #         #     exec(full_response)
# #         # except Exception as e:
# #         #     st.error(f"Error executing generated code: {e}")
    
# #     st.session_state.message.append({"role": "assistant", "content": full_response})

# generated_code = ""

# if "message" not in st.session_state:
#     st.session_state.message = []

# if "generated_code" not in st.session_state:
#     st.session_state.generated_code = ""

# chat_col, chart_col = st.columns([1, 2])

# with chat_col:
#     prompt = st.text_input("Please enter your description ...")

# if prompt:
#     st.session_state.message.append({"role": "user", "content": prompt})

#     response = openai.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a plot figure assistant that generates plotting code using Streamlit."},
#             {"role": "user", "content": f"Generate streamlit code to create a {prompt}. using plotly.express and ensure to use 'st.plotly_chart()' to display the chart."}
#         ],
#         stream = True
#     )

#     for chunk in response:
#         if chunk.choices[0].delta.content is not None:
#             generated_code += chunk.choices[0].delta.content


#     st.write(generated_code)

#     with chat_col:
#         st.markdown("### Generated Code")
#         st.code(st.session_state.generated_code, language="python")


#     try:
#         local_vars = {}
#         exec(generated_code, {"st": st, "px": px, "pd": pd})

#         # fig = local_vars.get("fig", None)
#         # if fig:
#         #     st.pyplot(fig)
#         # else:
#         #     st.warning("No figure was generated. Please ensure the code defines a 'fig' object.")
    
#     except Exception as e:
#         st.error(f"Error executing the code: {e}")


# with chat_col:
#     st.markdown("### Conversation History")
#     for message in st.session_state.message:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

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
    st.session_state["message"].append({"role": "user", "content": query})
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

    Query: """

    end_prompt = "\nResult:"

    full_prompt = prompt + query + end_prompt

    completion_params = {"temperature": 0.7, "max_tokens":250, "model": "text-davinci-003"}

    completion = openai.Completion.create(prompt=full_prompt, **completion_params)

    global response

    response = completion["choices"][0]["text"].strip("\n")

    if "plt" in response:
        create_plot_from_response(response)
        st.session_state["graph"].append(Image.open("/plot.png"))
        response = "Chart is shown below"
    else:
        st.session_state["graph"].append("no image")
        response = response
    
    st.session_state["messages"].append({"role": "assistant", "content":response})

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
