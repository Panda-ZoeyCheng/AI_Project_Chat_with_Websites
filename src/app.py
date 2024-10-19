import os
import io
import streamlit as st
from streamlit_chat import message
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import openai
import plotly.express as px

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="PlotBot", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>PlotBot</h1>", unsafe_allow_html=True)

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# File upload in sidebar
upload_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.sidebar.write("CSV Data Preview:")
    st.sidebar.write(df.head())  # Display first few rows of the CSV file
    schema = df.dtypes.to_string()  # Extract data schema
else:
    df = None

# Display chat messages from history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "content" in message:
            st.markdown(message["content"])
        elif message["role"] == "assistant" and "plot" in message:
            st.plotly_chart(message["plot"], use_container_width=True)
        else:
            st.markdown(message["content"])

# Function to generate plotting code based on user input
def generate_plot_code(df, user_request):
    prompt = f"""
    You are a Python plotting assistant. The user has requested to create a plot.
    Here is the data schema: 
    {schema}
    
    Based on the user request: '{user_request}', generate appropriate code for plotting using the following steps:
    - Use 'plotly.express' for interactive plotting.
    - Do not import libraries, only write the plotting code.
    - The DataFrame is called 'df'.
    
    Please generate the plot code in Python:
    """
    
    response = openai.chat.completions.create(
        # engine=st.session_state["openai_model"],
        model = "gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
            ],
        stream = True,
        temperature=0.7,
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

    return full_response
    

# Function to execute the generated code
def execute_plot_code(plot_code):
    # Create an empty plot placeholder
    exec_locals = {}
    try:
        exec(plot_code, {'df': df, 'px': px}, exec_locals)
        if "fig" in exec_locals:
            return exec_locals["fig"]
        else:
            return None
    except Exception as e:
        st.error(f"Error executing the plot code: {e}")
        return None


# Function to generate a response based on user input
def generate_response(prompt):
    response = openai.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        temperature=0.7,
    )
    
    full_response = ""
    for chunk in response:
        chunk_content = chunk.choices[0].delta.content
        if chunk_content:
            full_response += chunk_content
    return full_response


# Accept user input for custom plot request
user_input = st.chat_input("Please describe what you want to know or create")

if user_input and df is not None:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    if any(keyword in user_input.lower() for keyword in ["plot", "chart", "graph"]):
        if df is not None:
            # Generate plot code from LLM based on user input
            plot_code = generate_plot_code(df, user_input)

            # Display the generated code (for debugging)
            st.code(plot_code, language='python')

            # Execute the generated plot code
            plot_fig = execute_plot_code(plot_code)

            # Display the plot in the assistant message container
            with st.chat_message("assistant"):
                if plot_fig:
                    st.session_state["messages"].append({"role": "assistant", "plot": plot_fig})
                    st.plotly_chart(plot_fig, use_container_width=True)
                else:
                    st.error("Failed to generate the plot.")
        else:
            st.error("Please upload a CSV file to create a plot.")
    else:
        response = generate_response(input)

        with st.chat_message("assistant"):
            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.markdown(response)


# Clear the messages
if st.sidebar.button("Clear Conversation"):
    st.session_state["messages"] = []