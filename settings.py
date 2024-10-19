import stramlit as st

def settings_page():
    st.header("Settings")
    st.write(f"You are logged in as {st.session_state.role}.")