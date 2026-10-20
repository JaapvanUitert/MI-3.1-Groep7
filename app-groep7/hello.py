import streamlit as st

st.set_page_config(
    page_title="Hello",
    initial_sidebar_state="collapsed"
)

st.title(" Welkom bij de longkanker keuzehulp")
st.caption("Begin ow keuzehulp door hieronder op 'Start' te klikken.")

if st.button("Start", type="primary"):
    st.switch_page("pages/app.py")