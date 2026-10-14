import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scipy.optimize import curve_fit

from database import (
    initialize_database,
    load_experiments,
    load_measurements,
    save_experiment,
)
from viability import show_viability_module
from viability_comparison import (
    show_viability_comparison,
)
from hemolysis_astm import (
show_blood_qualification_module,
)
from hemolysis_assay import (
    show_hemolysis_assay_module,
)
from hemolysis_comparison import (
    show_hemolysis_comparison,
)
initialize_database()


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

def detect_plate_blocks(
    data: pd.DataFrame,
) -> list:
    """
    Wykrywa bloki płytki 96-dołkowej na podstawie:
    - kolumn 1-12,
    - ośmiu kolejnych wierszy A-H.
    """

    detected_blocks = []

    if data.shape[1] < 13:
        return detected_blocks

    for row_index in range(len(data)):
        possible_columns = data.iloc[
            row_index, 1:13
        ].tolist()

        try:
            plate_columns = [
                int(column)
                for column in possible_columns
            ]
        except (TypeError, ValueError):
            continue

        if plate_columns != list(range(1, 13)):
            continue

        if row_index + 8 >= len(data):
            continue

        possible_rows = (
            data.iloc[
                row_index + 1:row_index + 9,
                0,
            ]
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )

        if possible_rows != list("ABCDEFGH"):
            continue

        block_label_value = data.iloc[
            row_index, 0
        ]

        if pd.isna(block_label_value):
            block_label = f"Block_{row_index}"
        else:
            block_label = str(
                block_label_value
            ).strip()

        detected_blocks.append(
            {
                "label": block_label,
                "header_row": row_index,
            }
        )

    return detected_blocks

def get_measurement_config(
    block_label: str,
) -> dict:
    """
    Przypisuje etykiecie bloku rodzaj pomiaru,
    jednostkę oraz nazwę wyświetlaną w aplikacji.
    """

    normalized_label = (
        str(block_label)
        .strip()
        .upper()
    )

    measurement_configs = {
        "ABS": {
            "measurement_type": "Absorbancja",
            "measurement_unit": "AU",
            "value_column": "raw_value",
        },
        "RFU": {
            "measurement_type": "Fluorescencja",
            "measurement_unit": "RFU",
            "value_column": "raw_value",
        },
        "RLU": {
            "measurement_type": "Luminescencja",
            "measurement_unit": "RLU",
            "value_column": "raw_value",
        },
    }

    if normalized_label not in measurement_configs:
        raise ValueError(
            "Nieobsługiwany blok pomiarowy: "
            f"'{block_label}'."
        )

    return measurement_configs[normalized_label]

def get_supported_measurement_blocks(
    detected_blocks: list,
) -> list:
    """
    Zwraca tylko obsługiwane bloki pomiarowe:
    Abs, RFU i RLU.
    Pomija sekcję Sample.
    """

    supported_labels = {
        "ABS",
        "RFU",
        "RLU",
    }

    supported_blocks = []

    for block in detected_blocks:
        normalized_label = (
            str(block["label"])
            .strip()
            .upper()
        )

        if normalized_label in supported_labels:
            supported_blocks.append(block)

    return supported_blocks


def extract_measurement_plate(
    data: pd.DataFrame,
    header_row: int,
) -> pd.DataFrame:
    """
    Pobiera wskazaną macierz A-H x 1-12
    i przekształca ją do formatu:
    jeden dołek = jeden wiersz.
    """

    plate_data = data.iloc[
        header_row + 1:header_row + 9,
        0:13,
    ].copy()

    detected_rows = (
        plate_data.iloc[:, 0]
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    if detected_rows != list("ABCDEFGH"):
        raise ValueError(
            "Wskazany blok nie zawiera "
            "pełnego układu wierszy A-H."
        )

    records = []

    for _, row in plate_data.iterrows():
        plate_row = str(
            row.iloc[0]
        ).strip().upper()

        for plate_column in range(1, 13):
            raw_value = pd.to_numeric(
                row.iloc[plate_column],
                errors="coerce",
            )

            records.append(
                {
                    "well": (
                        f"{plate_row}{plate_column}"
                    ),
                    "plate_row": plate_row,
                    "plate_column": plate_column,
                    "raw_value": raw_value,
                }
            )

    result = pd.DataFrame(records)

    if len(result) != 96:
        raise ValueError(
            "Nie udało się odczytać "
            "96 pozycji płytki."
        )

    return result
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
def create_measurement_heatmap(
    measurement_results: pd.DataFrame,
    measurement_type: str,
    measurement_unit: str,
):
    """
    Tworzy heatmapę dowolnego rodzaju pomiaru:
    absorbancji, fluorescencji lub luminescencji.
    """

    heatmap_data = measurement_results.pivot(
        index="plate_row",
        columns="plate_column",
        values="raw_value",
    )

    heatmap_figure = px.imshow(
        heatmap_data,
        labels={
            "x": "Kolumna płytki",
            "y": "Wiersz płytki",
            "color": measurement_unit,
        },
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale="Viridis",
        text_auto=".4g",
        aspect="auto",
    )

    heatmap_figure.update_layout(
        title=(
            f"Rozkład pomiaru: {measurement_type} "
            f"[{measurement_unit}]"
        ),
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


def combine_measurement_with_sample_map(
    measurement_results: pd.DataFrame,
    plate_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Łączy dowolny rodzaj pomiaru z identyfikatorami
    próbek na podstawie oznaczenia dołka.
    """

    combined_data = measurement_results.merge(
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
def calculate_measurement_statistics(
    combined_data: pd.DataFrame,
    expected_replicates: int = 3,
    cv_limit: float = 15.0,
) -> pd.DataFrame:
    """
    Oblicza statystyki dla absorbancji,
    fluorescencji i luminescencji.
    """

    assigned_data = combined_data.loc[
        combined_data["sample_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    assigned_data["raw_value"] = pd.to_numeric(
        assigned_data["raw_value"],
        errors="coerce",
    )

    assigned_data = assigned_data.dropna(
        subset=["raw_value"]
    )

    if assigned_data.empty:
        raise ValueError(
            "Brak opisanych wyników liczbowych "
            "do obliczenia statystyk."
        )

    summary = (
        assigned_data
        .groupby(
            "sample_id",
            as_index=False,
        )
        .agg(
            n=("raw_value", "count"),
            mean_value=("raw_value", "mean"),
            sd_value=("raw_value", "std"),
            minimum=("raw_value", "min"),
            maximum=("raw_value", "max"),
        )
    )

    summary["cv_pct"] = (
        summary["sd_value"]
        / summary["mean_value"]
        * 100
    )

    summary.loc[
        summary["mean_value"] == 0,
        "cv_pct",
    ] = pd.NA

    summary["qc_status"] = "PASS"

    summary.loc[
        summary["n"] < expected_replicates,
        "qc_status",
    ] = "TOO_FEW"

    summary.loc[
        summary["n"] > expected_replicates,
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

def calculate_database_statistics(
    measurements: pd.DataFrame,
    expected_replicates: int = 3,
    cv_limit: float = 15.0,
) -> pd.DataFrame:
    """
    Oblicza statystyki opisowe dla pomiarów
    pobranych z bazy SQLite.
    """

    assigned_data = measurements.loc[
        measurements["sample_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    assigned_data["raw_value"] = pd.to_numeric(
        assigned_data["raw_value"],
        errors="coerce",
    )

    assigned_data = assigned_data.dropna(
        subset=["raw_value"]
    )

    if assigned_data.empty:
        raise ValueError(
            "Wybrany eksperyment nie zawiera "
            "opisanych wyników liczbowych."
        )

    summary = (
        assigned_data
        .groupby(
            "sample_id",
            as_index=False,
        )
        .agg(
            n=("raw_value", "count"),
            mean_value=("raw_value", "mean"),
            sd_value=("raw_value", "std"),
            minimum=("raw_value", "min"),
            maximum=("raw_value", "max"),
        )
    )

    summary["cv_pct"] = (
        summary["sd_value"]
        / summary["mean_value"]
        * 100
    )

    summary.loc[
        summary["mean_value"] == 0,
        "cv_pct",
    ] = pd.NA

    summary["qc_status"] = "PASS"

    summary.loc[
        summary["n"] < expected_replicates,
        "qc_status",
    ] = "TOO_FEW"

    summary.loc[
        summary["n"] > expected_replicates,
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
def prepare_concentration_chart_data(
    statistics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Przygotowuje kopię danych wyłącznie do wykresu.

    Blank i kontrole pozostają w oryginalnych danych
    oraz mogą być używane do obliczeń, ale nie są
    wyświetlane jako słupki.

    Funkcja rozpoznaje także stężenia zapisane np.
    jako '10', '10 ug', '10 µg/ml' lub 'Sample 10'.
    """

    chart_data = statistics.copy()

    chart_data["sample_id"] = (
        chart_data["sample_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    normalized_sample_id = (
        chart_data["sample_id"]
        .str.lower()
    )

    excluded_mask = (
        normalized_sample_id.str.contains(
            r"blank|ctrl|control|dmso",
            regex=True,
            na=False,
        )
    )

    chart_data = chart_data.loc[
        ~excluded_mask
    ].copy()

    extracted_concentration = (
        chart_data["sample_id"]
        .str.replace(",", ".", regex=False)
        .str.extract(
            r"([-+]?\d*\.?\d+)",
            expand=False,
        )
    )

    chart_data["concentration"] = pd.to_numeric(
        extracted_concentration,
        errors="coerce",
    )

    chart_data = chart_data.dropna(
        subset=["concentration"]
    )

    chart_data = chart_data.sort_values(
        by="concentration",
        ascending=True,
    ).reset_index(drop=True)

    chart_data["concentration_label"] = (
        chart_data["concentration"]
        .map(lambda value: f"{value:g}")
    )

    return chart_data

def show_database_browser() -> None:
    """
    Wyświetla zapisane eksperymenty oraz wykonuje
    analizę wybranego eksperymentu bez uploadu XLSX.
    """

    st.header("Baza eksperymentów")

    saved_experiments = load_experiments()

    if saved_experiments.empty:
        st.info(
            "Baza nie zawiera jeszcze zapisanych "
            "eksperymentów."
        )
        return

    project_options = sorted(
        saved_experiments["project_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_project = st.selectbox(
        "Wybierz projekt",
        options=project_options,
    )

    project_experiments = saved_experiments.loc[
        saved_experiments["project_name"]
        == selected_project
    ].copy()

    experiment_options = project_experiments[
        [
            "experiment_id",
            "project_name",
            "assay_type",
            "experiment_date",
            "measurement_type",
            "measurement_unit",
        ]
    ].to_dict(orient="records")

    selected_experiment = st.selectbox(
        "Wybierz eksperyment",
        options=experiment_options,
        format_func=lambda experiment: (
            f"{experiment['assay_type']} | "
            f"{experiment['experiment_date']} | "
            f"{experiment['experiment_id']}"
        ),
        key="database_experiment_selector",
    )

    selected_experiment_id = selected_experiment[
        "experiment_id"
    ]

    selected_measurements = load_measurements(
        selected_experiment_id
    )

    selected_metadata = project_experiments.loc[
        project_experiments["experiment_id"]
        == selected_experiment_id
        ].copy()

    if selected_measurements.empty:
        st.error(
            "Wybrany eksperyment nie zawiera "
            "żadnych zapisanych pomiarów."
        )

        st.info(
            "Metadane eksperymentu istnieją w bazie, "
            "ale tabela measurements nie zawiera "
            "rekordów dla identyfikatora "
            f"'{selected_experiment_id}'. "
            "Płytkę należy zapisać ponownie."
        )

        st.subheader(
            "Metadane niekompletnego eksperymentu"
        )

        st.dataframe(
            selected_metadata,
            width="stretch",
            hide_index=True,
        )

        return

    (
        overview_tab,
        measurements_tab,
        assay_analysis_tab,
        comparison_tab,
        statistics_tab,
        charts_tab,
    ) = st.tabs(
        [
            "Podsumowanie",
            "Pomiary",
            "Analiza testu",
            "Porównanie",
            "Statystyki ogólne",
            "Wykresy techniczne",
        ]
    )


    with overview_tab:
        st.subheader("Metadane eksperymentu")

        st.dataframe(
            selected_metadata,
            width="stretch",
            hide_index=True,
        )

        assigned_count = (
            selected_measurements["sample_id"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )

        unique_samples = (
            selected_measurements.loc[
                selected_measurements["sample_id"]
                .fillna("")
                .astype(str)
                .str.strip()
                .ne(""),
                "sample_id",
            ]
            .nunique()
        )

        (
            overview_column_1,
            overview_column_2,
            overview_column_3,
        ) = st.columns(3)

        overview_column_1.metric(
            "Wszystkie pomiary",
            len(selected_measurements),
        )

        overview_column_2.metric(
            "Opisane dołki",
            assigned_count,
        )

        overview_column_3.metric(
            "Unikalne identyfikatory",
            unique_samples,
        )

    with measurements_tab:
        st.subheader("Pomiary wybranego eksperymentu")

        st.dataframe(
            selected_measurements,
            width="stretch",
            hide_index=True,
        )

        measurements_csv = (
            selected_measurements.to_csv(
                index=False
            ).encode("utf-8")
        )

        st.download_button(
            label="Pobierz pomiary jako CSV",
            data=measurements_csv,
            file_name=(
                f"{selected_experiment_id}"
                "_measurements.csv"
            ),
            mime="text/csv",
        )
    with assay_analysis_tab:
        st.subheader(
            "Analiza specyficzna dla rodzaju testu"
        )

        selected_assay_type = str(
            selected_experiment.get(
                "assay_type",
                "",
            )
        ).strip()

        selected_measurement_type = str(
            selected_experiment.get(
                "measurement_type",
                "",
            )
        ).strip()

        st.info(
            f"Rodzaj analizy: "
            f"{selected_assay_type} | "
            f"Typ pomiaru: "
            f"{selected_measurement_type}"
        )

        if (
            selected_assay_type.casefold()
            == "żywotność".casefold()
        ):
            show_viability_module(
                selected_measurements
            )

        elif (
            selected_assay_type.casefold()
            == "hemoliza".casefold()
        ):
            stored_sample_ids = set(
                selected_measurements[
                    "sample_id"
                ]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.casefold()
                .tolist()
            )

            has_pfh = (
                "pfh" in stored_sample_ids
            )

            has_tbh = (
                "tbh" in stored_sample_ids
            )

            has_tbhd = (
                "tbhd" in stored_sample_ids
            )

            if (
                has_pfh
                and has_tbh
                and not has_tbhd
            ):
                st.info(
                    "Automatycznie rozpoznano "
                    "płytkę kwalifikacji krwi "
                    "PFH/TBH."
                )

                show_blood_qualification_module(
                    selected_measurements
                )

            elif has_tbhd:
                st.info(
                    "Automatycznie rozpoznano "
                    "właściwy test hemolizy."
                )

                show_hemolysis_assay_module(
                    selected_measurements
                )

            else:
                st.warning(
                    "Nie udało się automatycznie "
                    "rozpoznać rodzaju płytki "
                    "hemolizy. Oczekiwano PFH "
                    "i TBH albo TBHd."
                )

        elif (
            selected_assay_type.casefold()
            == (
                "Fluorescencja - GFP mRNA"
            ).casefold()
        ):
            st.info(
                "Moduł analizy GFP zostanie "
                "dodany w kolejnym etapie."
            )

        elif (
            selected_assay_type.casefold()
            == (
                "Luminescencja - Fluc mRNA"
            ).casefold()
        ):
            st.info(
                "Moduł analizy Fluc zostanie "
                "dodany w kolejnym etapie."
            )

        else:
            st.warning(
                "Dla rodzaju analizy "
                f"'{selected_assay_type}' "
                "nie ma jeszcze dedykowanego "
                "modułu."
            )
    with comparison_tab:
        if (
                selected_assay_type.casefold()
                == "hemoliza".casefold()
        ):
            show_hemolysis_comparison(
                saved_experiments
            )

        elif (
                selected_assay_type.casefold()
                == "żywotność".casefold()
        ):
            show_viability_comparison(
                saved_experiments
            )

        else:
            st.info(
                "Dla wybranego rodzaju testu "
                "nie ma jeszcze modułu porównania."
            )
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
            key="database_expected_replicates",
        )

        cv_limit = st.number_input(
            "Maksymalne CV [%]",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=1.0,
            key="database_cv_limit",
        )

        try:
            database_statistics = (
                calculate_database_statistics(
                    selected_measurements,
                    expected_replicates=int(
                        expected_replicates
                    ),
                    cv_limit=cv_limit,
                )
            )

            pass_count = (
                database_statistics["qc_status"]
                .eq("PASS")
                .sum()
            )

            high_cv_count = (
                database_statistics["qc_status"]
                .eq("HIGH_CV")
                .sum()
            )

            check_n_count = (
                database_statistics["qc_status"]
                .isin(["CHECK_N", "TOO_FEW"])
                .sum()
            )

            (
                statistic_column_1,
                statistic_column_2,
                statistic_column_3,
                statistic_column_4,
            ) = st.columns(4)

            statistic_column_1.metric(
                "Identyfikatory",
                len(database_statistics),
            )

            statistic_column_2.metric(
                "PASS",
                pass_count,
            )

            statistic_column_3.metric(
                "Wysokie CV",
                high_cv_count,
            )

            statistic_column_4.metric(
                "Do sprawdzenia",
                check_n_count,
            )

            st.dataframe(
                database_statistics.style.format(
                    {
                        "mean_value": "{:.4g}",
                        "sd_value": "{:.4g}",
                        "minimum": "{:.4g}",
                        "maximum": "{:.4g}",
                        "cv_pct": "{:.2f}",
                    },
                    na_rep="brak",
                ),
                width="stretch",
                hide_index=True,
            )

            statistics_csv = (
                database_statistics.to_csv(
                    index=False
                ).encode("utf-8")
            )

            st.download_button(
                label="Pobierz statystyki jako CSV",
                data=statistics_csv,
                file_name=(
                    f"{selected_experiment_id}"
                    "_statistics.csv"
                ),
                mime="text/csv",
            )

        except ValueError as statistics_error:
            st.warning(str(statistics_error))

    with charts_tab:
        st.subheader("Heatmapa zapisanej płytki")

        chart_measurements = (
            selected_measurements.copy()
        )

        chart_measurements["raw_value"] = (
            pd.to_numeric(
                chart_measurements["raw_value"],
                errors="coerce",
            )
        )

        heatmap_data = chart_measurements.pivot(
            index="plate_row",
            columns="plate_column",
            values="raw_value",
        )

        measurement_type = selected_experiment[
            "measurement_type"
        ]

        measurement_unit = selected_experiment[
            "measurement_unit"
        ]

        database_heatmap = px.imshow(
            heatmap_data,
            labels={
                "x": "Kolumna płytki",
                "y": "Wiersz płytki",
                "color": measurement_unit,
            },
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale="Viridis",
            text_auto=".4g",
            aspect="auto",
        )

        database_heatmap.update_layout(
            title=(
                f"{measurement_type} "
                f"[{measurement_unit}]"
            )
        )

        database_heatmap.update_xaxes(
            dtick=1,
            side="top",
        )

        st.plotly_chart(
            database_heatmap,
            width="stretch",
        )

        try:
            database_statistics = (
                calculate_database_statistics(
                    selected_measurements,
                    expected_replicates=int(
                        expected_replicates
                    ),
                    cv_limit=cv_limit,
                )
            )

            database_chart_statistics = (
                prepare_concentration_chart_data(
                    database_statistics
                )
            )

            if database_chart_statistics.empty:
                st.warning(
                    "Brak liczbowych stężeń do "
                    "utworzenia wykresu po usunięciu "
                    "blanków i kontroli."
                )

            else:
                st.subheader(
                    "Średni wynik według stężenia"
                )

                concentration_order = (
                    database_chart_statistics[
                        "concentration_label"
                    ].tolist()
                )

                mean_figure = px.bar(
                    database_chart_statistics,
                    x="concentration_label",
                    y="mean_value",
                    error_y="sd_value",
                    color="qc_status",
                    category_orders={
                        "concentration_label": (
                            concentration_order
                        ),
                    },
                    labels={
                        "concentration_label": "Stężenie",
                        "mean_value": (
                            f"Średni wynik "
                            f"[{measurement_unit}]"
                        ),
                        "qc_status": "Status QC",
                    },
                    title=(
                        f"{measurement_type}: "
                        "średnia i odchylenie standardowe"
                    ),
                )

                mean_figure.update_layout(
                    xaxis_title="Stężenie",
                    xaxis_type="category",
                )

                st.plotly_chart(
                    mean_figure,
                    width="stretch",
                )

        except ValueError as chart_error:
            st.warning(str(chart_error))

        except ValueError as chart_error:
            st.warning(str(chart_error))
application_view = st.sidebar.radio(
    "Wybierz moduł",
    options=[
        "Import i analiza pliku",
        "Baza eksperymentów",
    ],
)

if application_view == "Baza eksperymentów":
    show_database_browser()
    st.stop()


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
        measurement_tab,
        plate_map_tab,
        combined_tab,
        statistics_tab,
        viability_tab,
        hemolysis_tab,
        database_tab,
    ) = st.tabs(
        [
            "Arkusz",
            "Pomiar",
            "Mapa",
            "Dane",
            "Statystyka",
            "Żywotność",
            "Hemoliza",
            "Zapis",
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
        detected_blocks = detect_plate_blocks(
            raw_data
        )

        st.subheader(
            "Automatycznie wykryte macierze płytki"
        )

        if detected_blocks:
            detected_blocks_table = pd.DataFrame(
                detected_blocks
            )

            st.dataframe(
                detected_blocks_table,
                width="stretch",
                hide_index=True,
            )

        else:
            st.warning(
                "Nie wykryto macierzy płytki "
                "w układzie A-H x 1-12."
            )

    with measurement_tab:
        st.subheader("Ogólne wyniki pomiaru")

        detected_blocks = detect_plate_blocks(
            raw_data
        )

        supported_blocks = (
            get_supported_measurement_blocks(
                detected_blocks
            )
        )

        if not supported_blocks:
            st.warning(
                "Nie znaleziono obsługiwanego bloku "
                "pomiarowego: Abs, RFU lub RLU."
            )

        else:
            selected_measurement_block = st.selectbox(
                "Wybierz blok pomiarowy",
                options=supported_blocks,
                format_func=lambda block: (
                    str(block["label"])
                ),
                key="measurement_block_selector",
            )

            try:
                measurement_config = (
                    get_measurement_config(
                        selected_measurement_block[
                            "label"
                        ]
                    )
                )

                measurement_results = (
                    extract_measurement_plate(
                        raw_data,
                        selected_measurement_block[
                            "header_row"
                        ],
                    )
                )

                measurement_type = (
                    measurement_config[
                        "measurement_type"
                    ]
                )

                measurement_unit = (
                    measurement_config[
                        "measurement_unit"
                    ]
                )

                measurement_results[
                    "measurement_type"
                ] = measurement_type

                measurement_results[
                    "measurement_unit"
                ] = measurement_unit

                measurement_count = len(
                    measurement_results
                )

                missing_count = (
                    measurement_results["raw_value"]
                    .isna()
                    .sum()
                )

                minimum_value = (
                    measurement_results["raw_value"]
                    .min()
                )

                maximum_value = (
                    measurement_results["raw_value"]
                    .max()
                )

                (
                    measurement_column_1,
                    measurement_column_2,
                    measurement_column_3,
                    measurement_column_4,
                ) = st.columns(4)

                measurement_column_1.metric(
                    "Rodzaj pomiaru",
                    measurement_type,
                )

                measurement_column_2.metric(
                    "Liczba dołków",
                    measurement_count,
                )

                measurement_column_3.metric(
                    "Brakujące wyniki",
                    missing_count,
                )

                measurement_column_4.metric(
                    "Jednostka",
                    measurement_unit,
                )

                st.write(
                    f"Zakres wyników: "
                    f"**{minimum_value:.4g}–"
                    f"{maximum_value:.4g} "
                    f"{measurement_unit}**"
                )

                st.subheader(
                    "Surowe wyniki pomiaru"
                )

                st.dataframe(
                    measurement_results,
                    width="stretch",
                    hide_index=True,
                )

                st.subheader(
                    "Heatmapa płytki 96-dołkowej"
                )

                generic_heatmap = (
                    create_measurement_heatmap(
                        measurement_results,
                        measurement_type,
                        measurement_unit,
                    )
                )

                st.plotly_chart(
                    generic_heatmap,
                    width="stretch",
                )

                measurement_csv = (
                    measurement_results.to_csv(
                        index=False
                    ).encode("utf-8")
                )

                safe_measurement_name = (
                    measurement_type
                    .lower()
                    .replace(" ", "_")
                )

                st.download_button(
                    label=(
                        "Pobierz wyniki pomiaru "
                        "jako CSV"
                    ),
                    data=measurement_csv,
                    file_name=(
                        f"{safe_measurement_name}"
                        "_results.csv"
                    ),
                    mime="text/csv",
                )

            except ValueError as measurement_error:
                st.error(
                    "Nie udało się odczytać "
                    "wybranego bloku pomiarowego."
                )

                st.info(str(measurement_error))


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
            "Wyniki pomiaru z identyfikatorami"
        )

        try:
            combined_detected_blocks = (
                detect_plate_blocks(raw_data)
            )

            combined_supported_blocks = (
                get_supported_measurement_blocks(
                    combined_detected_blocks
                )
            )

            if not combined_supported_blocks:
                raise ValueError(
                    "Nie znaleziono bloku pomiarowego "
                    "Abs, RFU ani RLU."
                )

            selected_combined_block = st.selectbox(
                "Wybierz blok pomiarowy",
                options=combined_supported_blocks,
                format_func=lambda block: str(
                    block["label"]
                ),
                key="combined_block_selector",
            )

            combined_config = get_measurement_config(
                selected_combined_block["label"]
            )

            measurement_results_for_combining = (
                extract_measurement_plate(
                    raw_data,
                    selected_combined_block[
                        "header_row"
                    ],
                )
            )

            plate_map_for_combining = (
                extract_sample_plate_map(raw_data)
            )

            combined_data = (
                combine_measurement_with_sample_map(
                    measurement_results_for_combining,
                    plate_map_for_combining,
                )
            )

            measurement_type = combined_config[
                "measurement_type"
            ]

            measurement_unit = combined_config[
                "measurement_unit"
            ]

            combined_data["measurement_type"] = (
                measurement_type
            )

            combined_data["measurement_unit"] = (
                measurement_unit
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

            (
                combined_column_1,
                combined_column_2,
                combined_column_3,
                combined_column_4,
            ) = st.columns(4)

            combined_column_1.metric(
                "Rodzaj pomiaru",
                measurement_type,
            )

            combined_column_2.metric(
                "Opisane wyniki",
                assigned_count,
            )

            combined_column_3.metric(
                "Nieopisane wyniki",
                unassigned_count,
            )

            combined_column_4.metric(
                "Unikalne identyfikatory",
                unique_sample_count,
            )

            st.dataframe(
                combined_data,
                width="stretch",
                hide_index=True,
            )

            combined_csv = combined_data.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label=(
                    "Pobierz połączone dane jako CSV"
                ),
                data=combined_csv,
                file_name=(
                    "combined_measurement_results.csv"
                ),
                mime="text/csv",
            )

        except ValueError as combined_error:
            st.warning(
                "Nie udało się połączyć wyników "
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
            key="import_expected_replicates",
        )

        cv_limit = st.number_input(
            "Maksymalne CV [%]",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=1.0,
            key="import_cv_limit",
        )

        try:
            statistics_detected_blocks = (
                detect_plate_blocks(raw_data)
            )

            statistics_supported_blocks = (
                get_supported_measurement_blocks(
                    statistics_detected_blocks
                )
            )

            if not statistics_supported_blocks:
                raise ValueError(
                    "Nie znaleziono bloku pomiarowego "
                    "Abs, RFU ani RLU."
                )

            selected_statistics_block = st.selectbox(
                "Wybierz blok pomiarowy do analizy",
                options=statistics_supported_blocks,
                format_func=lambda block: str(
                    block["label"]
                ),
                key="statistics_block_selector",
            )

            statistics_config = (
                get_measurement_config(
                    selected_statistics_block["label"]
                )
            )

            statistics_measurements = (
                extract_measurement_plate(
                    raw_data,
                    selected_statistics_block[
                        "header_row"
                    ],
                )
            )

            statistics_plate_map = (
                extract_sample_plate_map(raw_data)
            )

            statistics_combined_data = (
                combine_measurement_with_sample_map(
                    statistics_measurements,
                    statistics_plate_map,
                )
            )

            sample_statistics = (
                calculate_measurement_statistics(
                    statistics_combined_data,
                    expected_replicates=int(
                        expected_replicates
                    ),
                    cv_limit=cv_limit,
                )
            )

            measurement_type = statistics_config[
                "measurement_type"
            ]

            measurement_unit = statistics_config[
                "measurement_unit"
            ]

            st.info(
                f"Analizowany pomiar: "
                f"{measurement_type} "
                f"[{measurement_unit}]"
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
                .isin(["CHECK_N", "TOO_FEW"])
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
                "Do sprawdzenia",
                check_n_count,
            )

            st.dataframe(
                sample_statistics.style.format(
                    {
                        "mean_value": "{:.4g}",
                        "sd_value": "{:.4g}",
                        "minimum": "{:.4g}",
                        "maximum": "{:.4g}",
                        "cv_pct": "{:.2f}",
                    },
                    na_rep="brak",
                ),
                width="stretch",
                hide_index=True,
            )

            chart_statistics = (
                prepare_concentration_chart_data(
                    sample_statistics
                )
            )

            if chart_statistics.empty:
                st.warning(
                    "Brak liczbowych stężeń do "
                    "utworzenia wykresu po usunięciu "
                    "blanków i kontroli."
                )

            else:
                st.subheader(
                    "Średni wynik według stężenia"
                )

                concentration_order = (
                    chart_statistics[
                        "concentration_label"
                    ].tolist()
                )

                statistics_figure = px.bar(
                    chart_statistics,
                    x="concentration_label",
                    y="mean_value",
                    error_y="sd_value",
                    color="qc_status",
                    category_orders={
                        "concentration_label": (
                            concentration_order
                        ),
                    },
                    labels={
                        "concentration_label": "Stężenie",
                        "mean_value": (
                            f"Średni wynik "
                            f"[{measurement_unit}]"
                        ),
                        "qc_status": "Status QC",
                    },
                    title=(
                        f"{measurement_type}: "
                        "średnia i odchylenie standardowe"
                    ),
                )

                statistics_figure.update_layout(
                    xaxis_title="Stężenie",
                    xaxis_type="category",
                )

                st.plotly_chart(
                    statistics_figure,
                    width="stretch",
                )

            statistics_csv = (
                sample_statistics.to_csv(
                    index=False
                ).encode("utf-8")
            )

            st.download_button(
                label="Pobierz statystyki jako CSV",
                data=statistics_csv,
                file_name=(
                    "measurement_statistics.csv"
                ),
                mime="text/csv",
            )

        except ValueError as statistics_error:
            st.warning(
                "Nie udało się obliczyć statystyk."
            )

            st.info(str(statistics_error))
    with viability_tab:
        st.subheader("Moduł analizy żywotności")

        try:
            viability_detected_blocks = (
                detect_plate_blocks(raw_data)
            )

            viability_supported_blocks = (
                get_supported_measurement_blocks(
                    viability_detected_blocks
                )
            )

            if not viability_supported_blocks:
                raise ValueError(
                    "Nie znaleziono bloku pomiarowego "
                    "Abs, RFU ani RLU."
                )

            selected_viability_block = st.selectbox(
                "Wybierz blok pomiarowy",
                options=viability_supported_blocks,
                format_func=lambda block: str(
                    block["label"]
                ),
                key="viability_block_selector",
            )

            viability_measurements = (
                extract_measurement_plate(
                    raw_data,
                    selected_viability_block[
                        "header_row"
                    ],
                )
            )

            viability_plate_map = (
                extract_sample_plate_map(
                    raw_data
                )
            )

            viability_combined_data = (
                combine_measurement_with_sample_map(
                    viability_measurements,
                    viability_plate_map,
                )
            )

            show_viability_module(
                viability_combined_data
            )

        except ValueError as viability_error:
            st.warning(
                "Nie udało się wykonać "
                "analizy żywotności."
            )

            st.info(str(viability_error))
    with hemolysis_tab:
        st.subheader(
            "Moduł analizy hemolizy"
        )

        try:
            hemolysis_detected_blocks = (
                detect_plate_blocks(raw_data)
            )

            hemolysis_supported_blocks = (
                get_supported_measurement_blocks(
                    hemolysis_detected_blocks
                )
            )

            if not hemolysis_supported_blocks:
                raise ValueError(
                    "Nie znaleziono bloku pomiarowego "
                    "Abs, RFU ani RLU."
                )

            absorbance_blocks = [
                block
                for block in hemolysis_supported_blocks
                if (
                        str(block["label"])
                        .strip()
                        .upper()
                        == "ABS"
                )
            ]

            if not absorbance_blocks:
                raise ValueError(
                    "Nie znaleziono bloku Abs "
                    "w wybranym arkuszu."
                )

            selected_hemolysis_block = (
                absorbance_blocks[0]
            )

            st.caption(
                "Automatycznie wykorzystano "
                "blok absorbancji Abs."
            )

            hemolysis_measurements = (
                extract_measurement_plate(
                    raw_data,
                    selected_hemolysis_block[
                        "header_row"
                    ],
                )
            )

            hemolysis_plate_map = (
                extract_sample_plate_map(
                    raw_data
                )
            )

            hemolysis_combined_data = (
                combine_measurement_with_sample_map(
                    hemolysis_measurements,
                    hemolysis_plate_map,
                )
            )

            hemolysis_plate_type = st.radio(
                "Wybierz rodzaj płytki hemolizy",
                options=[
                    "Kwalifikacja krwi: PFH i TBH",
                    "Właściwy test hemolizy",
                ],
                horizontal=True,
                key="hemolysis_plate_type_selector",
            )

            if (
                    hemolysis_plate_type
                    == "Kwalifikacja krwi: PFH i TBH"
            ):
                show_blood_qualification_module(
                    hemolysis_combined_data
                )

            else:
                show_hemolysis_assay_module(
                    hemolysis_combined_data
                )

        except ValueError as hemolysis_error:
            st.warning(
                "Nie udało się wykonać "
                "analizy hemolizy."
            )

            st.info(str(hemolysis_error))
    with database_tab:
        st.subheader("Zapis eksperymentu w bazie")

        st.write(
            "Uzupełnij metadane, sprawdź wybrany plik "
            "oraz arkusz, a następnie zapisz eksperyment."
        )

        st.info(
            "Formularz obsługuje absorbancję, "
            "fluorescencję i luminescencję. "
            "Przed zapisem sprawdź blok pomiarowy "
            "oraz mapę próbek."
        )
        database_detected_blocks = detect_plate_blocks(
            raw_data
        )

        database_supported_blocks = (
            get_supported_measurement_blocks(
                database_detected_blocks
            )
        )
        with st.form("save_experiment_form"):
            selected_database_block = st.selectbox(
                "Blok pomiarowy do zapisania *",
                options=database_supported_blocks,
                format_func=lambda block: str(
                    block["label"]
                ),
                disabled=(
                    not database_supported_blocks
                ),
            )
            project_name = st.text_input(
                "Nazwa projektu *",
                placeholder="Np. Project SIRT6",
            )

            experiment_id = st.text_input(
                "Unikalny identyfikator eksperymentu *",
                placeholder="Np. HEM_2026_09_26_01",
                help=(
                    "Każdy eksperyment musi mieć inny "
                    "identyfikator."
                ),
            )

            assay_type = st.selectbox(
                "Rodzaj analizy *",
                options=[
                    "Hemoliza",
                    "Żywotność",
                    "Stężenie glukozy",
                    "Luminescencja - Fluc mRNA",
                    "Fluorescencja - GFP mRNA",
                    "Inne badanie absorbancji",
                    "Inne badanie luminescencji",
                    "Inne badanie fluorescencji",
                ],
            )

            experiment_date = st.date_input(
                "Data eksperymentu *",
            )

            operator_name = st.text_input(
                "Operator",
                placeholder="Pole opcjonalne",
            )

            timepoint = st.text_input(
                "Punkt czasowy",
                placeholder="Np. 6 h, 24 h lub 48 h",
            )

            notes = st.text_area(
                "Uwagi",
                placeholder=(
                    "Np. linia komórkowa, numer materiału, "
                    "warunki inkubacji lub obserwacje."
                ),
            )

            st.write(
                f"**Plik źródłowy:** {uploaded_file.name}"
            )

            st.write(
                f"**Wybrany arkusz:** {selected_sheet}"
            )

            save_submitted = st.form_submit_button(
                "Zapisz eksperyment w bazie"
            )

        if save_submitted:
            missing_fields = []

            if not project_name.strip():
                missing_fields.append("nazwa projektu")

            if not experiment_id.strip():
                missing_fields.append(
                    "identyfikator eksperymentu"
                )

            if missing_fields:
                st.error(
                    "Uzupełnij wymagane pola: "
                    + ", ".join(missing_fields)
                    + "."
                )

            else:
                try:
                    if not database_supported_blocks:
                        raise ValueError(
                            "Nie znaleziono obsługiwanego "
                            "bloku pomiarowego Abs, RFU "
                            "ani RLU."
                        )

                    measurement_config_to_save = (
                        get_measurement_config(
                            selected_database_block[
                                "label"
                            ]
                        )
                    )

                    measurement_results_to_save = (
                        extract_measurement_plate(
                            raw_data,
                            selected_database_block[
                                "header_row"
                            ],
                        )
                    )

                    plate_map_to_save = (
                        extract_sample_plate_map(
                            raw_data
                        )
                    )

                    combined_data_to_save = (
                        combine_measurement_with_sample_map(
                            measurement_results_to_save,
                            plate_map_to_save,
                        )
                    )

                    assigned_count = (
                        combined_data_to_save[
                            "sample_id"
                        ]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .ne("")
                        .sum()
                    )

                    if assigned_count == 0:
                        raise ValueError(
                            "Nie zapisano eksperymentu: "
                            "wszystkie dołki mają pusty "
                            "identyfikator próbki."
                        )

                    measurement_type_to_save = (
                        measurement_config_to_save[
                            "measurement_type"
                        ]
                    )

                    measurement_unit_to_save = (
                        measurement_config_to_save[
                            "measurement_unit"
                        ]
                    )

                    metadata_to_save = {
                        "experiment_id": (
                            experiment_id.strip()
                        ),
                        "project_name": (
                            project_name.strip()
                        ),
                        "assay_type": assay_type,
                        "experiment_date": str(
                            experiment_date
                        ),
                        "measurement_type": (measurement_type_to_save),
                        "measurement_unit": (measurement_unit_to_save),
                        "source_file": (
                            uploaded_file.name
                        ),
                        "sheet_name": selected_sheet,
                        "operator": (
                            operator_name.strip()
                        ),
                        "timepoint": timepoint.strip(),
                        "notes": notes.strip(),
                    }

                    required_save_columns = {
                        "well",
                        "plate_row",
                        "plate_column",
                        "raw_value",
                        "sample_id",
                        "sample_status",
                    }

                    missing_save_columns = (
                            required_save_columns
                            - set(
                        combined_data_to_save.columns
                    )
                    )

                    if missing_save_columns:
                        raise ValueError(
                            "Nie można zapisać eksperymentu. "
                            "Brakuje wymaganych kolumn: "
                            + ", ".join(
                                sorted(
                                    missing_save_columns
                                )
                            )
                            + "."
                        )

                    measurement_count_to_save = len(
                        combined_data_to_save
                    )

                    if measurement_count_to_save != 96:
                        raise ValueError(
                            "Nie można zapisać eksperymentu. "
                            "Oczekiwano 96 pomiarów, ale "
                            f"otrzymano "
                            f"{measurement_count_to_save}."
                        )

                    assigned_count_to_save = (
                        combined_data_to_save[
                            "sample_id"
                        ]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .ne("")
                        .sum()
                    )

                    if assigned_count_to_save == 0:
                        raise ValueError(
                            "Nie można zapisać eksperymentu: "
                            "żaden dołek nie ma przypisanego "
                            "identyfikatora sample_id. "
                            "Sprawdź zakładkę Mapa."
                        )

                    st.write(
                        "**Kontrola danych przed zapisem:**"
                    )

                    st.write(
                        f"Liczba pomiarów: "
                        f"**{measurement_count_to_save}** | "
                        f"Opisane dołki: "
                        f"**{assigned_count_to_save}**"
                    )

                    save_experiment(
                        metadata_to_save,
                        combined_data_to_save,
                    )

                    saved_measurements_check = (
                        load_measurements(
                            experiment_id.strip()
                        )
                    )

                    saved_measurement_count = len(
                        saved_measurements_check
                    )

                    if saved_measurement_count != 96:
                        raise RuntimeError(
                            "Zapis eksperymentu nie został "
                            "ukończony prawidłowo. "
                            "W bazie znaleziono "
                            f"{saved_measurement_count} "
                            "pomiarów zamiast 96."
                        )

                    if (
                            "sample_id"
                            not in saved_measurements_check.columns
                    ):
                        raise RuntimeError(
                            "Pomiary zostały zapisane, ale "
                            "odczytane dane nie zawierają "
                            "kolumny sample_id."
                        )

                    saved_assigned_count = (
                        saved_measurements_check[
                            "sample_id"
                        ]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .ne("")
                        .sum()
                    )

                    if saved_assigned_count == 0:
                        raise RuntimeError(
                            "Pomiary zostały zapisane, ale "
                            "wszystkie wartości sample_id "
                            "w bazie są puste."
                        )

                    st.success(
                        "Eksperyment został zapisany "
                        "i zweryfikowany. "
                        f"Dodano {saved_measurement_count} "
                        "pomiarów, w tym "
                        f"{saved_assigned_count} opisanych "
                        "dołków."
                    )
                except ValueError as save_error:
                    st.error(str(save_error))

                except Exception as save_error:
                    st.error(
                        "Nie udało się zapisać "
                        "eksperymentu w bazie."
                    )

                    st.exception(save_error)

        st.divider()

        st.subheader("Eksperymenty zapisane w bazie")

        saved_experiments = load_experiments()

        if saved_experiments.empty:
            st.info(
                "Baza nie zawiera jeszcze "
                "zapisanych eksperymentów."
            )

        else:
            st.dataframe(
                saved_experiments,
                width="stretch",
                hide_index=True,
            )

            saved_experiment_id = st.selectbox(
                "Wybierz zapisany eksperyment",
                options=(
                    saved_experiments[
                        "experiment_id"
                    ].tolist()
                ),
                key="saved_experiment_selector",
            )

            saved_measurements = load_measurements(
                saved_experiment_id
            )

            st.write(
                "Liczba pomiarów w wybranym "
                f"eksperymencie: "
                f"**{len(saved_measurements)}**"
            )

            st.dataframe(
                saved_measurements,
                width="stretch",
                hide_index=True,
            )
except Exception as error:
    st.error("Nie udało się odczytać pliku XLSX.")
    st.exception(error)