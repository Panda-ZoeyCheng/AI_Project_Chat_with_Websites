import streamlit as st

st.set_page_config(page_title="PlotBot", page_icon=":robot_face:")

if "role" not in st.session_state:
    st.session_state.role = None

ROLES = [None, "User", "Admin"]


def login():
    st.header("Log in")
    role = st.selectbox("Choose your role", ROLES)

    if st.button("Log in"):
        st.session_state.role = role
        st.rerun()


def logout():
    st.session_state.role = None
    st.rerun()

role = st.session_state.role

logout_Page = st.Page(logout, title="Log out", icon=":material/logout:")
settings_page = st.Page("settings.py", title="Settings", icon=":material/settings:")
user_page = st.Page("plotbot/plotbot.py", title="Plot Bot", icon=":material/bar_chart:", default=(role == "User"))
admin_page = st.Page("admin/admin.py", title="Admin", icon=":material/person_add:", default=(role == "Admin"))

account_pages = [logout_Page, settings_page]
plotbot_pages = [user_page]
admin_pages = [admin_page]


page_dict = {}
if st.session_state.role in ["User", "Admin"]:
    page_dict["PlotBot"] = plotbot_pages
if st.session_state.role == "Admin":
    page_dict["Admin"] = admin_pages

if len(page_dict) > 0:
    pg = st.navigation({"Account": account_pages} | page_dict)
else:
    pg = st.navigation([st.Page(login)])

pg.run()
