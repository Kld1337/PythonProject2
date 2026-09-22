import pandas as pd
import plotly.express as px
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

def create_plate_heatmap(
    plate_results: pd.DataFrame,
):
    heatmap_data = plate_results.pivot(
        index="plate_row",
        columns="plate_column",
        values="raw_absorbance",
    )

    heatmap_figure = px.imshow(
        heatmap_data,
        labels={
            "x": "Kolumna płytki",
            "y": "Wiersz płytki",
            "color": "Absorbancja",
        },
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale="Viridis",
        text_auto=".4f",
        aspect="auto",
    )

    heatmap_figure.update_layout(
        title="Rozkład surowej absorbancji na płytce",
        xaxis_title="Kolumna płytki",
        yaxis_title="Wiersz płytki",
    )

    heatmap_figure.update_xaxes(
        dtick=1,
        side="top",
    )

    return heatmap_figure

def create_empty_plate_map() -> pd.DataFrame:
    records = []

    for plate_row in "ABCDEFGH":
        for plate_column in range(1, 13):
            records.append(
                {
                    "well": f"{plate_row}{plate_column}",
                    "sample_id": "",
                    "sample_type": "Sample",
                    "replicate": None,
                    "condition": "",
                    "concentration": None,
                    "timepoint": "",
                }
            )

    return pd.DataFrame(records)
def extract_sample_plate_map(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Wyszukuje sekcję Sample i przypisuje
    identyfikatory próbek do dołków A1-H12.
    """

    first_column = (
        data.iloc[:, 0]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    sample_rows = data.index[
        first_column == "sample"
    ].tolist()

    if not sample_rows:
        raise ValueError(
            "Nie znaleziono sekcji 'Sample' "
            "z identyfikatorami dołków."
        )

    header_row = sample_rows[0]

    column_numbers = data.iloc[
        header_row, 1:13
    ].tolist()

    try:
        detected_columns = [
            int(column_number)
            for column_number in column_numbers
        ]
    except (TypeError, ValueError):
        raise ValueError(
            "Nie udało się odczytać numerów kolumn "
            "w sekcji Sample."
        )

    if detected_columns != list(range(1, 13)):
        raise ValueError(
            "Sekcja Sample nie zawiera kolumn 1-12."
        )

    plate_map_data = data.iloc[
        header_row + 1:header_row + 9,
        0:13,
    ].copy()

    expected_rows = list("ABCDEFGH")

    detected_rows = (
        plate_map_data.iloc[:, 0]
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    if detected_rows != expected_rows:
        raise ValueError(
            "Sekcja Sample nie zawiera pełnego "
            "układu wierszy A-H."
        )

    records = []

    for _, row in plate_map_data.iterrows():
        plate_row = str(row.iloc[0]).strip().upper()

        for position, plate_column in enumerate(
            detected_columns,
            start=1,
        ):
            sample_value = row.iloc[position]

            if pd.isna(sample_value):
                sample_id = ""
            else:
                sample_id = str(sample_value).strip()

                if sample_id.lower() == "none":
                    sample_id = ""

            records.append(
                {
                    "well": f"{plate_row}{plate_column}",
                    "plate_row": plate_row,
                    "plate_column": plate_column,
                    "sample_id": sample_id,
                }
            )

    result = pd.DataFrame(records)

    if len(result) != 96:
        raise ValueError(
            f"Oczekiwano 96 pozycji mapy, "
            f"ale odczytano {len(result)}."
        )

    return result
def combine_results_with_sample_map(
    plate_results: pd.DataFrame,
    plate_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Łączy wyniki absorbancji z identyfikatorami
    próbek na podstawie oznaczenia dołka.
    """

    combined_data = plate_results.merge(
        plate_map[["well", "sample_id"]],
        on="well",
        how="left",
        validate="one_to_one",
    )

    combined_data["sample_id"] = (
        combined_data["sample_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    combined_data["sample_status"] = (
        combined_data["sample_id"]
        .ne("")
        .map(
            {
                True: "Assigned",
                False: "Unassigned",
            }
        )
    )

    if len(combined_data) != 96:
        raise ValueError(
            "Po połączeniu oczekiwano 96 dołków, "
            f"ale otrzymano {len(combined_data)}."
        )

    return combined_data
def calculate_sample_statistics(
    combined_data: pd.DataFrame,
    expected_replicates: int = 3,
    cv_limit: float = 15.0,
) -> pd.DataFrame:
    """
    Oblicza statystyki dla każdego identyfikatora próbki.
    """

    assigned_data = combined_data.loc[
        combined_data["sample_id"]
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    if assigned_data.empty:
        raise ValueError(
            "Brak opisanych dołków do obliczenia statystyk."
        )

    summary = (
        assigned_data
        .groupby(
            "sample_id",
            as_index=False,
        )
        .agg(
            n=("raw_absorbance", "count"),
            mean_absorbance=("raw_absorbance", "mean"),
            sd_absorbance=("raw_absorbance", "std"),
            minimum=("raw_absorbance", "min"),
            maximum=("raw_absorbance", "max"),
        )
    )

    summary["cv_pct"] = (
        summary["sd_absorbance"]
        / summary["mean_absorbance"]
        * 100
    )

    summary.loc[
        summary["mean_absorbance"] == 0,
        "cv_pct",
    ] = pd.NA

    summary["qc_status"] = "PASS"

    summary.loc[
        summary["n"] != expected_replicates,
        "qc_status",
    ] = "CHECK_N"

    summary.loc[
        (
            summary["n"] == expected_replicates
        )
        & (
            summary["cv_pct"] > cv_limit
        ),
        "qc_status",
    ] = "HIGH_CV"

    return summary.sort_values(
        by=["qc_status", "sample_id"]
    ).reset_index(drop=True)

uploaded_files = st.file_uploader(
    "Wybierz pliki XLSX z czytnika",
    type=["xlsx"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info(
        "Wczytaj co najmniej jeden plik XLSX, "
        "aby rozpocząć analizę."
    )
    st.stop()

uploaded_file = st.selectbox(
    "Wybierz plik do podglądu",
    options=uploaded_files,
    format_func=lambda file: file.name,
)

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

    (
        preview_tab,
        absorbance_tab,
        plate_map_tab,
        combined_tab,
        statistics_tab,
    ) = st.tabs(
        [
            "Podgląd całego arkusza",
            "Wyniki absorbancji",
            "Mapa płytki",
            "Połączone dane",
            "Statystyka",
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

            st.subheader("Heatmapa płytki 96-dołkowej")

            heatmap_figure = create_plate_heatmap(
                plate_results
            )

            st.plotly_chart(
                heatmap_figure,
                use_container_width=True,
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
    with plate_map_tab:
        st.subheader(
            "Automatycznie odczytana mapa płytki"
        )

        try:
            plate_map = extract_sample_plate_map(
                raw_data
            )

            assigned_mask = (
                plate_map["sample_id"]
                .astype(str)
                .str.strip()
                .ne("")
            )

            assigned_wells = assigned_mask.sum()

            unique_samples = (
                plate_map.loc[
                    assigned_mask,
                    "sample_id",
                ]
                .nunique()
            )

            map_column_1, map_column_2, map_column_3 = (
                st.columns(3)
            )

            map_column_1.metric(
                "Wszystkie dołki",
                len(plate_map),
            )

            map_column_2.metric(
                "Opisane dołki",
                assigned_wells,
            )

            map_column_3.metric(
                "Unikalne identyfikatory",
                unique_samples,
            )

            st.subheader("Lista dołków i identyfikatorów")

            st.dataframe(
                plate_map,
                use_container_width=True,
                hide_index=True,
            )

            plate_map_matrix = plate_map.pivot(
                index="plate_row",
                columns="plate_column",
                values="sample_id",
            )

            st.subheader(
                "Układ identyfikatorów na płytce"
            )

            st.dataframe(
                plate_map_matrix,
                use_container_width=True,
            )

            plate_map_csv = plate_map.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Pobierz mapę płytki jako CSV",
                data=plate_map_csv,
                file_name="plate_map.csv",
                mime="text/csv",
            )

        except ValueError as map_error:
            st.warning(
                "W wybranym arkuszu nie znaleziono "
                "mapy próbek."
            )

            st.info(str(map_error))

    with combined_tab:
            st.subheader(
                "Wyniki absorbancji z identyfikatorami"
            )

            try:
                plate_results = extract_absorbance_plate(
                    raw_data
                )

                plate_map = extract_sample_plate_map(
                    raw_data
                )

                combined_data = (
                    combine_results_with_sample_map(
                        plate_results,
                        plate_map,
                    )
                )

                assigned_count = (
                    combined_data["sample_status"]
                    .eq("Assigned")
                    .sum()
                )

                unassigned_count = (
                    combined_data["sample_status"]
                    .eq("Unassigned")
                    .sum()
                )

                unique_sample_count = (
                    combined_data.loc[
                        combined_data["sample_status"]
                        .eq("Assigned"),
                        "sample_id",
                    ]
                        .nunique()
                )

                combined_column_1, combined_column_2, combined_column_3 = (
                    st.columns(3)
                )

                combined_column_1.metric(
              "Opisane wyniki",
                    assigned_count,
                )

                combined_column_2.metric(
              "Nieopisane wyniki",
                    unassigned_count,
                )

                combined_column_3.metric(
              "Unikalne identyfikatory",
                    unique_sample_count,
                )

                st.dataframe(
                    combined_data,
                    use_container_width=True,
                    hide_index=True,
                )

                combined_csv = combined_data.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Pobierz połączone dane jako CSV",
                    data=combined_csv,
                    file_name="combined_plate_results.csv",
                    mime="text/csv",
                )

            except ValueError as combined_error:
                st.warning(
                    "Nie udało się połączyć absorbancji "
                    "z identyfikatorami."
                )

                st.info(str(combined_error))
    with statistics_tab:
        st.subheader(
            "Statystyki według identyfikatora próbki"
        )

        expected_replicates = st.number_input(
            "Oczekiwana liczba powtórzeń",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
        )

        cv_limit = st.number_input(
            "Maksymalne CV [%]",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=1.0,
        )

        try:
            plate_results = extract_absorbance_plate(
                raw_data
            )

            plate_map = extract_sample_plate_map(
                raw_data
            )

            combined_data = (
                combine_results_with_sample_map(
                    plate_results,
                    plate_map,
                )
            )

            sample_statistics = (
                calculate_sample_statistics(
                    combined_data,
                    expected_replicates=(
                        int(expected_replicates)
                    ),
                    cv_limit=cv_limit,
                )
            )

            pass_count = (
                sample_statistics["qc_status"]
                .eq("PASS")
                .sum()
            )

            high_cv_count = (
                sample_statistics["qc_status"]
                .eq("HIGH_CV")
                .sum()
            )

            check_n_count = (
                sample_statistics["qc_status"]
                .eq("CHECK_N")
                .sum()
            )

            (
                statistics_column_1,
                statistics_column_2,
                statistics_column_3,
                statistics_column_4,
            ) = st.columns(4)

            statistics_column_1.metric(
                "Identyfikatory",
                len(sample_statistics),
            )

            statistics_column_2.metric(
                "PASS",
                pass_count,
            )

            statistics_column_3.metric(
                "Wysokie CV",
                high_cv_count,
            )

            statistics_column_4.metric(
                "Nieprawidłowe n",
                check_n_count,
            )

            st.dataframe(
                sample_statistics.style.format(
                    {
                        "mean_absorbance": "{:.4f}",
                        "sd_absorbance": "{:.4f}",
                        "minimum": "{:.4f}",
                        "maximum": "{:.4f}",
                        "cv_pct": "{:.2f}",
                    },
                    na_rep="brak",
                ),
                use_container_width=True,
                hide_index=True,
            )

            statistics_csv = (
                sample_statistics.to_csv(
                    index=False
                ).encode("utf-8")
            )

            st.download_button(
                label="Pobierz statystyki jako CSV",
                data=statistics_csv,
                file_name="sample_statistics.csv",
                mime="text/csv",
            )

        except ValueError as statistics_error:
            st.warning(
                "Nie udało się obliczyć statystyk."
            )

            st.info(str(statistics_error))
except Exception as error:
    st.error("Nie udało się odczytać pliku XLSX.")
    st.exception(error)