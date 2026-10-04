import pandas as pd
import plotly.express as px
import streamlit as st


def extract_numeric_concentration(
    sample_ids: pd.Series,
) -> pd.Series:
    load_experiments,
    """
    Wyciąga liczbowe stężenie z identyfikatorów:
    100
    CTRL 100
    Sample 100
    100 ug/ml
    """

    concentration_text = (
        sample_ids
        .fillna("")
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.extract(
            r"([-+]?\d*\.?\d+)",
            expand=False,
        )
    )

    return pd.to_numeric(
        concentration_text,
        errors="coerce",
    )


def calculate_paired_control_viability(
    combined_data: pd.DataFrame,
    blank_sample_id: str,
    control_prefix: str = "CTRL",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    float,
]:
    """
    Oblicza żywotność względem kontroli
    odpowiadającej temu samemu stężeniu.

    Przykład:
    100 porównywane jest z CTRL 100.

    Żywotność [%] =
    (próbka - blank)
    / (kontrola tego samego stężenia - blank)
    * 100
    """

    required_columns = {
        "sample_id",
        "raw_value",
    }

    missing_columns = (
        required_columns
        - set(combined_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Brakuje wymaganych kolumn: "
            + ", ".join(sorted(missing_columns))
        )

    data = combined_data.copy()

    data["sample_id"] = (
        data["sample_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    data["raw_value"] = pd.to_numeric(
        data["raw_value"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["raw_value"]
    )

    blank_values = data.loc[
        data["sample_id"] == blank_sample_id,
        "raw_value",
    ]

    if blank_values.empty:
        raise ValueError(
            "Nie znaleziono wyników dla "
            "wybranego blanku."
        )

    blank_mean = float(blank_values.mean())

    normalized_sample_id = (
        data["sample_id"]
        .str.upper()
        .str.strip()
    )

    normalized_control_prefix = (
        control_prefix
        .upper()
        .strip()
    )

    data["is_control"] = (
        normalized_sample_id
        .str.startswith(
            normalized_control_prefix
        )
    )

    data["is_blank"] = (
        data["sample_id"]
        == blank_sample_id
    )

    data["concentration"] = (
        extract_numeric_concentration(
            data["sample_id"]
        )
    )

    controls = data.loc[
        data["is_control"]
        & data["concentration"].notna()
    ].copy()

    samples = data.loc[
        ~data["is_control"]
        & ~data["is_blank"]
        & data["concentration"].notna()
    ].copy()

    if controls.empty:
        raise ValueError(
            "Nie znaleziono kontroli oznaczonych "
            f"prefiksem '{control_prefix}'."
        )

    if samples.empty:
        raise ValueError(
            "Nie znaleziono próbek z liczbowymi "
            "stężeniami."
        )

    control_summary = (
        controls
        .groupby(
            "concentration",
            as_index=False,
        )
        .agg(
            control_n=("raw_value", "count"),
            control_mean=("raw_value", "mean"),
            control_sd=("raw_value", "std"),
        )
    )

    control_summary["control_cv_pct"] = (
        control_summary["control_sd"]
        / control_summary["control_mean"]
        * 100
    )

    viability_data = samples.merge(
        control_summary,
        on="concentration",
        how="left",
        validate="many_to_one",
    )

    missing_controls = (
        viability_data.loc[
            viability_data["control_mean"].isna(),
            "concentration",
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if missing_controls:
        missing_text = ", ".join(
            f"{value:g}"
            for value in missing_controls
        )

        raise ValueError(
            "Nie znaleziono odpowiadającej kontroli "
            "dla stężeń: "
            + missing_text
        )

    denominator = (
        viability_data["control_mean"]
        - blank_mean
    )

    invalid_denominator = denominator <= 0

    if invalid_denominator.any():
        invalid_concentrations = (
            viability_data.loc[
                invalid_denominator,
                "concentration",
            ]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        invalid_text = ", ".join(
            f"{value:g}"
            for value in invalid_concentrations
        )

        raise ValueError(
            "Po korekcji blanku kontrola ma "
            "wartość mniejszą lub równą zero "
            "dla stężeń: "
            + invalid_text
        )

    viability_data["blank_corrected_value"] = (
        viability_data["raw_value"]
        - blank_mean
    )

    viability_data[
        "control_blank_corrected"
    ] = denominator

    viability_data["viability_pct"] = (
        viability_data[
            "blank_corrected_value"
        ]
        / viability_data[
            "control_blank_corrected"
        ]
        * 100
    )

    viability_summary = (
        viability_data
        .groupby(
            [
                "sample_id",
                "concentration",
            ],
            as_index=False,
        )
        .agg(
            n=("viability_pct", "count"),
            mean_raw_value=(
                "raw_value",
                "mean",
            ),
            sd_raw_value=(
                "raw_value",
                "std",
            ),
            mean_control_value=(
                "control_mean",
                "first",
            ),
            control_n=(
                "control_n",
                "first",
            ),
            control_cv_pct=(
                "control_cv_pct",
                "first",
            ),
            mean_viability_pct=(
                "viability_pct",
                "mean",
            ),
            sd_viability_pct=(
                "viability_pct",
                "std",
            ),
        )
        .sort_values(
            "concentration",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    viability_summary["sample_cv_pct"] = (
        viability_summary["sd_raw_value"]
        / viability_summary["mean_raw_value"]
        * 100
    )

    viability_summary[
        "viability_cv_pct"
    ] = (
        viability_summary[
            "sd_viability_pct"
        ]
        / viability_summary[
            "mean_viability_pct"
        ]
        * 100
    )

    return (
        viability_data,
        viability_summary,
        control_summary,
        blank_mean,
    )


def create_viability_chart(
    viability_data: pd.DataFrame,
    viability_summary: pd.DataFrame,
):
    """
    Tworzy wykres punktów pojedynczych,
    średniej i SD żywotności.
    """

    concentration_order = (
        viability_summary[
            "concentration"
        ]
        .sort_values()
        .tolist()
    )

    concentration_labels = [
        f"{value:g}"
        for value in concentration_order
    ]

    chart_data = viability_data.copy()

    chart_data[
        "concentration_label"
    ] = (
        chart_data["concentration"]
        .map(lambda value: f"{value:g}")
    )

    summary_for_chart = (
        viability_summary.copy()
    )

    summary_for_chart[
        "concentration_label"
    ] = (
        summary_for_chart["concentration"]
        .map(lambda value: f"{value:g}")
    )

    figure = px.strip(
        chart_data,
        x="concentration_label",
        y="viability_pct",
        category_orders={
            "concentration_label": (
                concentration_labels
            ),
        },
        labels={
            "concentration_label": "Stężenie",
            "viability_pct": "Żywotność [%]",
        },
        title=(
            "Żywotność względem kontroli "
            "dopasowanej do stężenia"
        ),
    )

    figure.add_scatter(
        x=summary_for_chart[
            "concentration_label"
        ],
        y=summary_for_chart[
            "mean_viability_pct"
        ],
        error_y={
            "type": "data",
            "array": summary_for_chart[
                "sd_viability_pct"
            ],
            "visible": True,
        },
        mode="markers",
        name="Średnia ± SD",
        marker={
            "size": 11,
            "color": "black",
            "symbol": "diamond",
        },
    )

    figure.add_hline(
        y=100,
        line_dash="dash",
        line_color="gray",
        annotation_text="Kontrola = 100%",
    )

    figure.update_layout(
        xaxis_title="Stężenie",
        yaxis_title="Żywotność [%]",
    )

    return figure


def show_viability_module(
        combined_data: pd.DataFrame,
) -> None:
    """
    Wyświetla moduł analizy żywotności.
    """

    st.subheader("Analiza żywotności")

    st.write(
        "Każde stężenie próbki jest porównywane "
        "z kontrolą o tym samym stężeniu, np. "
        "100 z CTRL 100."
    )

    available_sample_ids = sorted(
        combined_data["sample_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .loc[
            lambda values: values.ne("")
        ]
        .unique()
        .tolist()
    )

    if not available_sample_ids:
        st.warning(
            "Nie znaleziono opisanych próbek."
        )
        return

    blank_candidates = [
        sample_id
        for sample_id in available_sample_ids
        if "blank" in sample_id.lower()
    ]

    if blank_candidates:
        default_blank_index = (
            available_sample_ids.index(
                blank_candidates[0]
            )
        )
    else:
        default_blank_index = 0

    blank_sample_id = st.selectbox(
        "Wybierz blank",
        options=available_sample_ids,
        index=default_blank_index,
        key="viability_blank_selector",
    )

    control_prefix = st.text_input(
        "Prefiks kontroli dopasowanej do stężenia",
        value="CTRL",
        help=(
            "Dla identyfikatorów CTRL 1000, "
            "CTRL 800 itd. wpisz CTRL."
        ),
    )

    try:
        (
            viability_data,
            viability_summary,
            control_summary,
            blank_mean,
        ) = calculate_paired_control_viability(
            combined_data,
            blank_sample_id,
            control_prefix,
        )

        (
            metric_column_1,
            metric_column_2,
            metric_column_3,
        ) = st.columns(3)

        metric_column_1.metric(
            "Średni sygnał blanku",
            f"{blank_mean:.4g}",
        )

        metric_column_2.metric(
            "Liczba stężeń",
            len(viability_summary),
        )

        metric_column_3.metric(
            "Liczba poziomów kontroli",
            len(control_summary),
        )

        st.subheader(
            "Kontrole dopasowane do stężeń"
        )

        st.dataframe(
            control_summary.style.format(
                {
                    "concentration": "{:.4g}",
                    "control_mean": "{:.4g}",
                    "control_sd": "{:.4g}",
                    "control_cv_pct": "{:.2f}",
                },
                na_rep="brak",
            ),
            width="stretch",
            hide_index=True,
        )

        st.subheader("Wyniki żywotności")

        st.dataframe(
            viability_summary.style.format(
                {
                    "concentration": "{:.4g}",
                    "mean_raw_value": "{:.4g}",
                    "sd_raw_value": "{:.4g}",
                    "sample_cv_pct": "{:.2f}",
                    "mean_control_value": "{:.4g}",
                    "control_cv_pct": "{:.2f}",
                    "mean_viability_pct": "{:.2f}",
                    "sd_viability_pct": "{:.2f}",
                    "viability_cv_pct": "{:.2f}",
                },
                na_rep="brak",
            ),
            width="stretch",
            hide_index=True,
        )

        viability_figure = create_viability_chart(
            viability_data,
            viability_summary,
        )

        st.plotly_chart(
            viability_figure,
            width="stretch",
        )

        results_csv = viability_summary.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Pobierz wyniki żywotności jako CSV",
            data=results_csv,
            file_name="viability_results.csv",
            mime="text/csv",
        )

    except ValueError as error:
        st.error(str(error))