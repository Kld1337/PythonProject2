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


def extract_absorbance_plate(data: pd.DataFrame) -> pd.DataFrame:
    """
    Wyszukuje macierz absorbancji w eksporcie czytnika
    i przekształca ją do formatu:
    jeden dołek = jeden wiersz.
    """

    first_column = (
        data.iloc[:, 0]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    absorbance_rows = data.index[
        first_column == "abs"
    ].tolist()

    if not absorbance_rows:
        raise ValueError(
            "Nie znaleziono wiersza oznaczonego jako 'Abs'."
        )

    header_row = absorbance_rows[0]

    column_numbers = data.iloc[
        header_row, 1:13
    ].tolist()

    expected_columns = list(range(1, 13))

    detected_columns = [
        int(column_number)
        for column_number in column_numbers
    ]

    if detected_columns != expected_columns:
        raise ValueError(
            "Nie znaleziono kompletnego układu kolumn 1–12."
        )

    plate_data = data.iloc[
        header_row + 1:header_row + 9,
        0:13,
    ].copy()

    expected_rows = list("ABCDEFGH")

    detected_rows = (
        plate_data.iloc[:, 0]
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    if detected_rows != expected_rows:
        raise ValueError(
            "Nie znaleziono kompletnego układu wierszy A–H."
        )

    records = []

    for _, row in plate_data.iterrows():
        plate_row = str(row.iloc[0]).strip().upper()

        for position, plate_column in enumerate(
            detected_columns,
            start=1,
        ):
            raw_value = row.iloc[position]

            numeric_value = pd.to_numeric(
                raw_value,
                errors="coerce",
            )

            records.append(
                {
                    "well": f"{plate_row}{plate_column}",
                    "plate_row": plate_row,
                    "plate_column": plate_column,
                    "raw_absorbance": numeric_value,
                }
            )

    result = pd.DataFrame(records)

    if len(result) != 96:
        raise ValueError(
            f"Oczekiwano 96 dołków, ale odczytano {len(result)}."
        )

    return result


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

    raw_data = pd.read_excel(
        uploaded_file,
        sheet_name=selected_sheet,
        engine="openpyxl",
        header=None,
    )

    preview_tab, absorbance_tab = st.tabs(
        [
            "Podgląd całego arkusza",
            "Wyniki absorbancji",
        ]
    )

    with preview_tab:
        st.subheader("Podgląd zawartości pliku")

        st.write(
            f"Liczba wierszy: {raw_data.shape[0]} | "
            f"Liczba kolumn: {raw_data.shape[1]}"
        )

        st.dataframe(
            raw_data,
            use_container_width=True,
        )

    with absorbance_tab:
        try:
            plate_results = extract_absorbance_plate(
                raw_data
            )

            number_of_measurements = len(plate_results)

            missing_results = (
                plate_results["raw_absorbance"]
                .isna()
                .sum()
            )

            minimum_value = (
                plate_results["raw_absorbance"]
                .min()
            )

            maximum_value = (
                plate_results["raw_absorbance"]
                .max()
            )

            column_1, column_2, column_3, column_4 = (
                st.columns(4)
            )

            column_1.metric(
                "Liczba dołków",
                number_of_measurements,
            )

            column_2.metric(
                "Brakujące wyniki",
                missing_results,
            )

            column_3.metric(
                "Minimum",
                f"{minimum_value:.4f}",
            )

            column_4.metric(
                "Maksimum",
                f"{maximum_value:.4f}",
            )

            st.subheader(
                "Surowe wyniki absorbancji"
            )

            st.dataframe(
                plate_results.style.format(
                    {
                        "raw_absorbance": "{:.4f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            csv_result = plate_results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Pobierz 96 wyników jako CSV",
                data=csv_result,
                file_name="raw_absorbance_results.csv",
                mime="text/csv",
            )

        except ValueError as plate_error:
            st.warning(
                "Wybrany arkusz nie zawiera "
                "rozpoznanej macierzy absorbancji."
            )

            st.info(str(plate_error))

except Exception as error:
    st.error("Nie udało się odczytać pliku XLSX.")
    st.exception(error)