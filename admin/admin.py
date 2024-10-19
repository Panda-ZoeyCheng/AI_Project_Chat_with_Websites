import streamlit as st


def admin_page():
    st.header("Admin")
    st.write(f"You are logged in as {st.session_state.role}.")