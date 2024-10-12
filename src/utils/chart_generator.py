import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

def generate_chart(chart_type, data):
    """
    Parameters:
        chart_type: types of figure such as 'bar', 'line', 'scatter'
        data:  data for plot

    Returns:
        fig: figure for Streamlit
    """

    if chart_type == "bar":
        st.bar_chart(data)
    elif chart_type == "line":
        st.line_chart(data)
    elif chart_type == "scatter":
        fig = px.scatter(data, x=data.columns[0], y=data.columns[1])
        st.plotly_chart(fig)
    else:
        st.error(f"Unsupported chart type: {chart_type}")