import streamlit as st


st.set_page_config(
    page_title="Bioassay Data Hub",
    page_icon="🧪",
    layout="wide",
)

st.title("Bioassay Data Hub")

st.write(
    "Aplikacja do gromadzenia i analizy danych "
    "z czytnika mikropłytek."
)

st.subheader("Planowane moduły")

st.write(
    """
    - Żywotność
    - Hemoliza
    - Wydajność translacji mRNA
    - Stężenie glukozy
    - Stężenie białka
    - Analiza pojedynczych eksperymentów
    - Podsumowanie projektu
    """
)
