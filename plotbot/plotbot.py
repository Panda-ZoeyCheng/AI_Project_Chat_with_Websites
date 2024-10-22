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

st.header("PlotBot")

openai.api_key = st.secrets["OPENAI_API_KEY"]

def handle_file_upload():
    upload_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    if upload_file is not None:
        df = pd.read_csv(upload_file)
        st.session_state["df"] = df
        st.sidebar.write("CSV Data Preview:")
        st.sidebar.write(df.head()) 

        return df.dtypes.to_string()
    else:
        return None


def clear_conversation():
    if st.sidebar.button("Clear Conversation"):
        st.session_state["messages"] = [
            {"role": "system", "content": "Hello, what can I do for you today?"}
        ]

        st.session_state["plot_intent"] = False


def show_tips():
    with st.expander("***Tips***", icon="💬", expanded=True):
        st.markdown("""
            **How to Use This Tool**:
            
            1. **Upload your CSV file**: Please upload a CSV data file using the uploader in the sidebar.
            2. **Describe the plot you want**: In the chatbox, specify what kind of chart you'd like to generate (e.g., "bar chart of sales over time").
            3. **Interactive chart**: PlotBot will analyze the data and generate an interactive plot for you.

            You can ask for different types of plots based on the data, and PlotBot will attempt to create the corresponding visualizations.
            """)

def initialize_state():
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello, what can I do for you today?"}
        ]

    if "plot_intent" not in st.session_state:
        st.session_state["plot_intent"] = False


def display_chat_history():
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if "content" in message:
                st.markdown(message["content"])
            elif "plot" in message:
                st.plotly_chart(message["plot"], use_container_width=True)

def detect_intent(input):

    plot_keywords = [
        "plot", "chart", "graph", "draw", "visualize", "create plot", 
        "generate chart", "make graph", "sketch", "illustrate", 
        "display plot", "show graph", "produce chart", "construct graph", 
        "render", "depict", "map", "build plot", "outline", 
        "design chart", "draft", "trace", "plot out", "diagram", 
        "plot a figure", "build a chart", "chart out", "draw a graph", 
        "graph out", "make a figure", "draw up", "plot data", 
        "create figure", "generate figure", "figure out", "present graph", 
        "display chart", "display figure", "illustrate data", 
        "visual representation", "graphical representation", "render chart", 
        "produce graph", "represent data", "visualize chart", "graphically show", 
        "graphical illustration", "diagram out", "make a plot", "construct a diagram", 
        "design a graph", "trace a plot", "plot a diagram", "show a diagram",

        "scatter", "scatter plot", "scatter diagram", 
        "bar chart", "bar graph", "histogram", "pie chart", "line chart", 
        "area chart", "heatmap", "box plot", "violin plot", "bubble chart", 
        "density plot", "scatter matrix", "hexbin plot", "error bars", 
        "stacked bar chart", "stacked area chart", "polar chart", 
        "donut chart", "funnel chart", "distribution plot", "point plot", 
        "joint plot", "pair plot", "categorical plot", "swarm plot",

        "hit a plot", "make a scatter", "create a scatter", "show histogram", 
        "make a bar chart", "plot a histogram", "create a heatmap", 
        "visualize a pie chart", "plot a bubble chart", "build a scatter matrix"
    ]

    cancel_plot_keywords = ["cancel plot", "end plot", "cancel", "end"]

    if not input or not isinstance(input, str):
        return "chat"
    
    if any(keyword in input.lower() for keyword in plot_keywords):
        return "plot"
    elif any(keyword in input.lower() for keyword in cancel_plot_keywords):
        return "cancel_plot"
    else:
        return "chat"


def generate_plot_code(df, user_request, schema):
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
    
    try:
        response = openai.chat.completions.create(

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
    
    except Exception as e:
        st.error(f"Error in generating plot code: {e}")
        return None
    

def execute_plot_code(plot_code):
    exec_locals = {}
    try:
        exec(plot_code, {'df': st.session_state.get('df', None), 'px': px}, exec_locals)
        if "fig" in exec_locals:
            fig = exec_locals["fig"]
            return fig
        else:
            return None
    except Exception as e:
        st.error(f"Error executing the plot code: {e}")
        return None


def generate_response(prompt):
    try:
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
    
    except Exception as e:
        st.error(f"Error in generating response: {e}")
        return None


def plot_with_user_input(user_input, schema):
    intent = detect_intent(user_input)

    if intent == "plot":
        df = st.session_state.get("df", None)
        if df is not None:
            plot_code = generate_plot_code(df, user_input, schema)
            st.code(plot_code, language='python')
            plot_fig = execute_plot_code(plot_code)
            if plot_fig:
                st.session_state["messages"].append({"role": "assistant", "plot": plot_fig})
                display_chat_history()
        else:
            st.warning("No CSV file uploaded. Please upload a CSV file to proceed with plotting.")
    elif intent == "cancel_plot":
        st.session_state["plot_intent"] = False
    else:
        response = generate_response(user_input)
        if response:
            st.session_state["messages"].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

def main():
    schema = handle_file_upload()
    clear_conversation()
    show_tips()
    initialize_state()
    display_chat_history()
    user_input = st.chat_input("Please describe what you want to know or create")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        plot_with_user_input(user_input, schema)

main()
