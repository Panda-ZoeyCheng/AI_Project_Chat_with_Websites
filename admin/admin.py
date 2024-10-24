import streamlit as st
import json

if "llm_params" not in st.session_state:
    try:
        with open("llm_params.json", "r") as f:
            st.session_state["llm_params"] = json.load(f)
    except FileNotFoundError:
        st.session_state["llm_params"] = {
            "temperature": 0.7,
            "max_tokens": 500,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }

st.title("LLM Parameter Settings")


# Temperature
temperature = st.number_input(
    "Temperature", 
    min_value=0.0, 
    max_value=1.0, 
    value=round(st.session_state["llm_params"]["temperature"], 1), 
    step=0.1,
    key="temperature_input"
)
st.write("")

# Max Tokens
max_tokens = st.number_input(
    "Max Tokens", 
    min_value=100, 
    max_value=1000, 
    value=round(st.session_state["llm_params"]["max_tokens"], 1), 
    step=10,
    key="max_tokens_input"
)
st.write("")

# Top P
top_p = st.number_input(
    "Top P", 
    min_value=0.0, 
    max_value=1.0, 
    value=round(st.session_state["llm_params"]["top_p"], 1), 
    step=0.1,
    key="top_p_input"
)
st.write("")

# Frequency Penalty
frequency_penalty = st.number_input(
    "Frequency Penalty", 
    min_value=-2.0, 
    max_value=2.0, 
    value=round(st.session_state["llm_params"]["frequency_penalty"], 1), 
    step=0.1,
    key="frequency_penalty_input"
)
st.write("")

# Presence Penalty
presence_penalty = st.number_input(
    "Presence Penalty", 
    min_value=-2.0, 
    max_value=2.0, 
    value=round(st.session_state["llm_params"]["presence_penalty"], 1), 
    step=0.1,
    key="presence_penalty_input"
)
st.write("")


# Save button
if st.button("Save Settings"):
    st.session_state["llm_params"]["temperature"] = round(temperature, 1)
    st.session_state["llm_params"]["max_tokens"] = round(max_tokens, 1)
    st.session_state["llm_params"]["top_p"] = round(top_p, 1)
    st.session_state["llm_params"]["frequency_penalty"] = round(frequency_penalty, 1)
    st.session_state["llm_params"]["presence_penalty"] = round(presence_penalty, 1)

    with open("llm_params.json", "w") as f:
        json.dump(st.session_state["llm_params"], f)
    
    st.success("Settings saved successfully!")


st.write("### Current LLM Parameters:")
st.json(st.session_state["llm_params"])
