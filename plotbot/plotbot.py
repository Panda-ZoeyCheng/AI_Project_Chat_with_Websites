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
import json

st.header("PlotBot")

openai.api_key = st.secrets["OPENAI_API_KEY"]
DEFAULT_CSV_PATH = "default.csv"

def parameter_settings():
    try:
        with open("llm_params.json", "r") as f:
            llm_params = json.load(f)
    except FileNotFoundError:
        llm_params = {
            "temperature": 0.7,
            "max_tokens": 500,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    return llm_params

llm_params = parameter_settings()


def load_default_csv():
    if os.path.exists(DEFAULT_CSV_PATH):
        df = pd.read_csv(DEFAULT_CSV_PATH)
        st.session_state["df"] = df
        return df
    else:
        st.error(f"Default CSV file not found at {DEFAULT_CSV_PATH}.")
        return None

def clean_df_plotting(df):
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception as e:
            st.warning(f"Could not convert column {col} to numeric: {e}")
    return df.dropna(axis=1, how="any")

def handle_file_upload():
    if "df" not in st.session_state:
        load_default_csv()

    upload_file = st.sidebar.file_uploader("Upload CSV file", type="csv", key="unique_file_uploader_1")

    if upload_file is not None:
        df = pd.read_csv(upload_file)
        st.session_state["df"] = df
        st.session_state["uploaded_file"] = True
        st.sidebar.write("Uploaded CSV Data Preview:")
        st.sidebar.write(df.head()) 

        # return df.dtypes.to_string()
    else:
        if "df" in st.session_state:
            st.sidebar.write("Currently using the following file:")
            st.sidebar.write(st.session_state["df"].head())

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
            
            *Note: Currently, this tool only supports CSV files with numeric data.*
            """)
        
def show_llm_params():
    with st.expander("***Current LLM Params***", icon="🔧", expanded=False):
        st.markdown(llm_params)


def initialize_state():
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello, what can I do for you today?"}
        ]

    if "plot_intent" not in st.session_state:
        st.session_state["plot_intent"] = False


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
    valid_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns.tolist()

    prompt = f"""
    You are a Python plotting assistant. The user has requested to create a plot.
    Here is the data schema: 
    {schema}
    
    The available columns in the DataFrame are:
    {', '.join(valid_columns)}. 
    Numeric columns are: {', '.join(numeric_columns)}.
    Non-numeric columns are: {', '.join(non_numeric_columns)}.
    
    Based on the user request: '{user_request}', generate appropriate code for plotting using the following steps:
    - Use 'plotly.express' for interactive plotting.
    - Do not import libraries, only write the plotting code.
    - The DataFrame is called 'df'.
    - If the request involves numeric data (like line charts, bar charts, histograms), ensure you only use numeric columns ({', '.join(numeric_columns)}).
    - If the request involves categorical or non-numeric data (like pie charts or scatter plots with labels), you can use non-numeric columns ({', '.join(non_numeric_columns)}).
    - If the user mentions a column that does not exist, ignore it and use available columns instead.
    - Ensure the code is error-free and matches the DataFrame schema.
    
    Please generate the plot code in Python:
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            temperature=llm_params["temperature"],  
            max_tokens=llm_params["max_tokens"],
            top_p=llm_params["top_p"],
            frequency_penalty=llm_params["frequency_penalty"],
            presence_penalty=llm_params["presence_penalty"]
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

        valid_columns = df.columns
        for col in valid_columns:
            if "labels_column_name" in full_response:
                full_response = full_response.replace("labels_column_name", col)

        return full_response
    
    except Exception as e:
        st.error(f"Error in generating plot code: {e}")
        return None

    

def execute_plot_code(plot_code):
    exec_locals = {}
    try:
        df = clean_df_plotting(st.session_state.get('df', None))
        exec(plot_code, {'df': df, 'px': px}, exec_locals)
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
            temperature = llm_params["temperature"],  
            max_tokens = llm_params["max_tokens"],
            top_p = llm_params["top_p"],
            frequency_penalty = llm_params["frequency_penalty"],
            presence_penalty = llm_params["presence_penalty"]
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
        st.session_state["plot_intent"] = True
        df = st.session_state.get("df", None)

        if df is not None:
            plot_code = generate_plot_code(df, user_input, schema)
            st.code(plot_code, language='python')
            plot_fig = execute_plot_code(plot_code)
            st.session_state["messages"].append({"role": "assistant", "code": plot_code})

            if plot_fig:
                st.session_state["messages"].append({"role": "assistant", "plot": plot_fig})

                config = {'displaylogo': False, 'modeBarButtonsToRemove': ['autoScale2d', 'resetScale2d']}
                with st.chat_message("assistant"):
                    st.plotly_chart(plot_fig, theme="streamlit", use_container_width=False, config=config)
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
    show_llm_params()
    initialize_state()
    
    if st.session_state["messages"]:
        for i, message in enumerate(st.session_state["messages"]):
            with st.chat_message(message["role"]):
                if "content" in message:
                    st.markdown(message["content"])
                elif "code" in message:
                    st.code(message["code"], language='python')
                elif "plot" in message:
                    config = {'displaylogo': False, 'modeBarButtonsToRemove': ['autoScale2d', 'resetScale2d']}
                    st.plotly_chart(message["plot"], theme="streamlit", use_container_width=True, config=config)

    user_input = st.chat_input("Please describe what you want to know or create")

    if user_input:

        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state["messages"].append({"role": "user", "content": user_input})  

        plot_with_user_input(user_input, schema)

main()
