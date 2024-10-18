import os
import io
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

if "past" not in st.session_state:
    st.session_state["past"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "graph" not in st.session_state:
    st.session_state["graph"] = []
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant"}
    ]

st.sidebar.title("Sidebar")

upload_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
if upload_file is not None:
    df = pd.read_csv(upload_file)
    schema = df.dtypes.to_string()
    query = None

counter_placeholder = st.sidebar.empty()
clear_button = st.sidebar.button("Clear conversation", key="clear")


model_name = 'gpt-3.5-turbo'


if clear_button:
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant"}
    ]

plot_folder = os.path.join(os.getcwd(), "src", "plots")
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder) 

def create_plot_from_response(response_text):
    # [eval(statement) for statement in response_text.split("\n")]
    for statement in response_text.split("\n"):
        statement = statement.strip()
        if statement and not statement.startswith("```"):
            try:
                eval(statement)
            except Exception as e:
                st.error(f"Error executing statement: {statement}\n{e}")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.show()
    buf.seek(0)
    st.session_state["graph"].append(buf)

    # local_vars = {"df": df, "plt": plt, "sns": sns}
    # try:
    #     exec(response_text, globals(), local_vars)
    #     image_path = os.path.join(plot_folder, "plot.png")
    #     plt.savefig("plot.png")
    #     plt.close()
    #     st.write(f"Image saved at: {image_path}")
    # except Exception as e:
    #     st.error(f"Error executing generated code: {e}")

def generate_response(query):
    # st.session_state["messages"].append({"role": "user", "content": query})
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


    response = openai.chat.completions.create(
        model = model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        stream = True,
        temperature = 0.7
    )

    full_response = ""
    for chunk in response:
        chunk_content = chunk.choices[0].delta.content
        if chunk_content:
            full_response += chunk_content

    if "```python" in full_response:
        full_response = full_response.replace("```python", "")
    if "```" in full_response:
        full_response = full_response.replace("```", "")
    
    # st.write("Generated code:")
    # st.code(full_response, language="python")

    if "plt" in full_response:
        create_plot_from_response(full_response)
        # image_path = os.path.join(os.getcwd(), "src", "plots", "plot.png")
        # st.write(f"Image path: {image_path}")

        return "Chart is shown below"

        # if os.path.exists(image_path):
        #     st.session_state["graph"].append(Image.open(image_path))
        #     full_response = "Chart is shown below"
    else:
        # st.session_state["graph"].append(None)
        return  full_response
    
    # st.session_state["messages"].append({"role": "assistant", "content": full_response})

# response_container = st.container()
# container = st.container()

def display_messages():
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "graph":
                st.image(message["content"], width=700)
            else:
                st.markdown(message["content"])

input = st.chat_input("Please describe what graph you would like to show?")

if input:
    st.session_state["messages"].append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.markdown(input)
    # st.session_state.message.append({"role": "user", "content": input})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = generate_response(input)
        message_placeholder.markdown(full_response)

    st.session_state["messages"].append({"role": "assistant", "content": full_response})

# with container:
#     with st.form(key="my_form", clear_on_submit=True):
#         user_input = st.text_area("You:", key="input", height=70)
#         submit_button = st.form_submit_button(label="send")

#     if submit_button and user_input:
#         output = generate_response(user_input)
#         st.session_state["past"].append(user_input)
#         st.session_state["generated"].append(output)
    

# if st.session_state["generated"]:
#     with response_container:
#         for i in range(len(st.session_state["generated"])):
#             message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="identicon")
#             message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

#             try:
#                 # st.image(st.session_state["graph"][i], width=10)
#                 if st.session_state["graph"][i]:
#                     st.image(st.session_state["graph"][i], width=700)
#                 else:
#                     st.warning("No image to display.")
#             except Exception as e:
#                 st.error(f"Error displaying image: {e}")

display_messages()