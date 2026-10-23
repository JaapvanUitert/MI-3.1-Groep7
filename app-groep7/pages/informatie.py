import streamlit as st


st.set_page_config(
    page_title="CDSS Longkanker Keuzehulp",
    initial_sidebar_state="collapsed"
)

st.title("Longkanker en behandelopties")

st.write("De gebruikelijke behandeling van longkanker is afhankelijk van de subtype, stadium en conditie van de patiënt. [2] "
"\n Stadium I: orgaanchirurgie, maar bij hoge leeftijd en/of slechte conditie kan een andere behandeling worden overwogen."
"\n Stadium II: vaak wordt orgaanchirurgie en adjuvante chemotherapie aangeraden bij patiënten met stadium II"
"\n Stadium III: vaak een combinatie van chemotherapie en radiotherapie. Er is ook een mogelijkheid tot orgaanchirurgie."
"\n Stadium IV: is afhankelijk of er wel of geen mutaties zijn. Bij wel mutaties kan er bijvoorbeeld chemotherapie gebruikt worden")

st.divider()
st.subheader("referenties")
st.write("1.  Longkanker [Internet]. Iknl.nl. 2018 [cited 2026 Oct 14]. Beschikbaar op: https://iknl.nl/kankersoorten/longkanker ")