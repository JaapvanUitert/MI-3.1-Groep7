import streamlit as st

st.set_page_config(
    page_title="Hello",
    initial_sidebar_state="collapsed"
)

st.title(" Welkom bij de longkanker keuzehulp")
st.caption("Begin uw keuzehulp door hieronder op 'Start' te klikken.")

if st.button("Start", type="primary"):
    st.switch_page("pages/app.py")
    

st.info(
    "**Belangrijke informatie over deze keuzehulp**\n\n"
    "• **Geen medisch advies:** Deze applicatie is uitsluitend een *keuzehulp* ter ondersteuning en voorbereiding op het gesprek met uw arts. "
    "De getoonde uitkomsten zijn **uitdrukkelijk geen medisch behandeladvies**.\n\n"
    "• **Gebaseerd op IKNL-data:** De voorspellingen worden berekend door een neuraal netwerk dat is getraind op historische patiëntgegevens van de Nederlandse Kankerregistratie (IKNL).\n\n"
    "• **Beperkingen:** De getoonde uitkomsten geven een inschatting op basis van populatiegemiddelden van vergelijkbare patiëntgroepen. "
    "Het model houdt geen rekening met al uw individuele kenmerken, zoals uw algehele fysieke conditie (performance status), bijkomende aandoeningen of specifieke tumormarkers.\n\n"
    "*Bespreek uw persoonlijke situatie en behandelkeuzes daarom altijd met uw behandelend arts of verpleegkundig specialist.*"
)