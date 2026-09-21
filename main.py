import pandas as pd
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

uploaded_file = st.file_uploader(
    "Wybierz plik XLSX z czytnika",
    type=["xlsx"],
)

if uploaded_file is None:
    st.info("Wczytaj plik XLSX, aby rozpocząć analizę.")
    st.stop()

try:
    excel_file = pd.ExcelFile(
        uploaded_file,
        engine="openpyxl",
    )

    st.success("Plik został poprawnie wczytany.")

    selected_sheet = st.selectbox(
        "Wybierz arkusz",
        options=excel_file.sheet_names,
    )

    data = pd.read_excel(
        uploaded_file,
        sheet_name=selected_sheet,
        engine="openpyxl",
        header=None,
    )

    st.subheader("Podgląd zawartości pliku")

    st.write(
        f"Liczba wierszy: {data.shape[0]} | "
        f"Liczba kolumn: {data.shape[1]}"
    )

    st.dataframe(
        data,
        use_container_width=True,
    )

except Exception as error:
    st.error("Nie udało się odczytać pliku XLSX.")
    st.exception(error)