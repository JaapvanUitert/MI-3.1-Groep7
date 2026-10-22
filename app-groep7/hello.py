import streamlit as st

pages = [
    st.Page("pages/start_pagina.py", title="Startpagina", default=True),
    st.Page("pages/app.py", title="keuzehulp"),
    st.Page("pages/over_ons.py", title="Over ons"),
]

pg = st.navigation(pages, position="top")
pg.run()

#  st.Page("pages/informatie.py", title="informatie"),