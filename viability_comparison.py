import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database import load_measurements
from viability import (
    calculate_paired_control_viability,
    fit_viability_4pl,
    four_parameter_logistic,
)


def find_blank_sample_id(
    measurements: pd.DataFrame,
) -> str:
    """
    Automatycznie znajduje identyfikator blanku.
    """

    sample_ids = (
        measurements["sample_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .loc[
            lambda values: values.ne("")
        ]
        .unique()
        .tolist()
    )

    blank_candidates = [
        sample_id
        for sample_id in sample_ids
        if "blank" in sample_id.casefold()
    ]

    if not blank_candidates:
        raise ValueError(
            "Nie znaleziono identyfikatora blanku."
        )

    if len(blank_candidates) > 1:
        raise ValueError(
            "Znaleziono więcej niż jeden możliwy "
            "identyfikator blanku: "
            + ", ".join(blank_candidates)
            + "."
        )

    return blank_candidates[0]


def analyze_stored_viability_experiment(
    experiment_id: str,
    control_prefix: str = "CTRL",
) -> dict:
    """
    Pobiera eksperyment żywotności z bazy,
    oblicza żywotność i dopasowuje model 4PL.
    """

    measurements = load_measurements(
        experiment_id
    )

    if measurements.empty:
        raise ValueError(
            "Eksperyment nie zawiera pomiarów."
        )

    if len(measurements) != 96:
        raise ValueError(
            "Eksperyment zawiera "
            f"{len(measurements)} pomiarów "
            "zamiast 96."
        )

    required_columns = {
        "sample_id",
        "raw_value",
    }

    missing_columns = (
        required_columns
        - set(measurements.columns)
    )

    if missing_columns:
        raise ValueError(
            "Brakuje wymaganych kolumn: "
            + ", ".join(
                sorted(missing_columns)
            )
            + "."
        )

    assigned_count = (
        measurements["sample_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    if assigned_count == 0:
        raise ValueError(
            "Eksperyment nie zawiera opisanych "
            "dołków."
        )

    blank_sample_id = find_blank_sample_id(
        measurements
    )

    (
        viability_data,
        viability_summary,
        control_summary,
        blank_mean,
    ) = calculate_paired_control_viability(
        measurements,
        blank_sample_id,
        control_prefix,
    )

    fit_result = fit_viability_4pl(
        viability_summary
    )

    fit_reliable = (
        fit_result["crosses_50_percent"]
        and fit_result["ic50_in_range"]
        and not np.isnan(
            fit_result["r_squared"]
        )
        and fit_result["r_squared"] >= 0.8
    )

    viability_summary = (
        viability_summary.copy()
    )

    viability_summary["experiment_id"] = (
        experiment_id
    )

    viability_data = viability_data.copy()

    viability_data["experiment_id"] = (
        experiment_id
    )

    return {
        "experiment_id": experiment_id,
        "blank_sample_id": blank_sample_id,
        "blank_mean": blank_mean,
        "viability_data": viability_data,
        "viability_summary": (
            viability_summary
        ),
        "control_summary": control_summary,
        "fit_result": fit_result,
        "fit_reliable": fit_reliable,
    }


def find_viability_experiments(
    saved_experiments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Zwraca eksperymenty zapisane jako żywotność.
    """

    if saved_experiments.empty:
        return saved_experiments.copy()

    required_columns = {
        "experiment_id",
        "assay_type",
    }

    if not required_columns.issubset(
        saved_experiments.columns
    ):
        return pd.DataFrame()

    viability_experiments = (
        saved_experiments.loc[
            saved_experiments["assay_type"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq("żywotność".casefold())
        ]
        .copy()
    )

    return viability_experiments


def create_viability_comparison_figure(
    analyses: list,
):
    """
    Tworzy wspólny wykres punktów i krzywych 4PL.
    """

    figure = go.Figure()

    for analysis in analyses:
        experiment_id = analysis[
            "experiment_id"
        ]

        fit_result = analysis["fit_result"]
        fit_data = fit_result["fit_data"]

        figure.add_trace(
            go.Scatter(
                x=fit_data["concentration"],
                y=fit_data[
                    "mean_viability_pct"
                ],
                error_y={
                    "type": "data",
                    "array": fit_data[
                        "sd_viability_pct"
                    ],
                    "visible": True,
                },
                mode="markers",
                name=(
                    f"{experiment_id}: "
                    "średnia ± SD"
                ),
                legendgroup=experiment_id,
                marker={
                    "size": 9,
                },
                hovertemplate=(
                    f"<b>{experiment_id}</b>"
                    "<br>Stężenie: %{x:.4g}"
                    "<br>Żywotność: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

        figure.add_trace(
            go.Scatter(
                x=fit_result["curve_x"],
                y=fit_result["curve_y"],
                mode="lines",
                name=(
                    f"{experiment_id}: 4PL"
                ),
                legendgroup=experiment_id,
                line={
                    "width": 3,
                },
                hovertemplate=(
                    f"<b>{experiment_id}</b>"
                    "<br>Stężenie: %{x:.4g}"
                    "<br>4PL: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

    figure.add_hline(
        y=50,
        line_dash="dot",
        line_color="gray",
        annotation_text="50% żywotności",
    )

    all_minimum_concentrations = [
        analysis["fit_result"][
            "minimum_concentration"
        ]
        for analysis in analyses
    ]

    all_maximum_concentrations = [
        analysis["fit_result"][
            "maximum_concentration"
        ]
        for analysis in analyses
    ]

    minimum_concentration = min(
        all_minimum_concentrations
    )

    maximum_concentration = max(
        all_maximum_concentrations
    )

    x_axis_minimum = np.log10(
        minimum_concentration * 0.8
    )

    x_axis_maximum = np.log10(
        maximum_concentration * 1.2
    )

    all_y_values = []

    for analysis in analyses:
        fit_result = analysis["fit_result"]
        fit_data = fit_result["fit_data"]

        all_y_values.extend(
            fit_data["mean_viability_pct"]
            .dropna()
            .tolist()
        )

        all_y_values.extend(
            fit_result["curve_y"].tolist()
        )

    maximum_y_value = max(all_y_values)

    y_axis_maximum = max(
        110.0,
        maximum_y_value * 1.1,
    )

    figure.update_layout(
        title=(
            "Porównanie krzywych "
            "dawka–odpowiedź"
        ),
        xaxis={
            "title": "Stężenie",
            "type": "log",
            "range": [
                x_axis_minimum,
                x_axis_maximum,
            ],
            "dtick": 1,
        },
        yaxis={
            "title": "Żywotność [%]",
            "range": [
                0,
                y_axis_maximum,
            ],
        },
        legend={
            "title": "Eksperyment",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        margin={
            "l": 70,
            "r": 40,
            "t": 120,
            "b": 70,
        },
        hovermode="closest",
    )

    return figure


def create_ic50_comparison_figure(
    parameter_table: pd.DataFrame,
):
    """
    Tworzy wykres porównujący IC50.
    """

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=parameter_table[
                "experiment_id"
            ],
            y=parameter_table["ic50"],
            error_y={
                "type": "data",
                "array": (
                    parameter_table[
                        "ic50_ci_upper"
                    ]
                    - parameter_table["ic50"]
                ),
                "arrayminus": (
                    parameter_table["ic50"]
                    - parameter_table[
                        "ic50_ci_lower"
                    ]
                ),
                "visible": True,
            },
            marker_color=(
                parameter_table[
                    "fit_status"
                ].map(
                    {
                        "PASS": "seagreen",
                        "CHECK": "darkorange",
                    }
                )
            ),
            hovertemplate=(
                "<b>%{x}</b>"
                "<br>IC50: %{y:.4g}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title="Porównanie IC50",
        xaxis_title="Eksperyment",
        yaxis_title="IC50",
        xaxis_tickangle=-45,
    )

    return figure


def create_viability_parameter_table(
    analyses: list,
) -> pd.DataFrame:
    """
    Tworzy tabelę parametrów 4PL.
    """

    rows = []

    for analysis in analyses:
        fit_result = analysis["fit_result"]

        rows.append(
            {
                "experiment_id": analysis[
                    "experiment_id"
                ],
                "blank_sample_id": analysis[
                    "blank_sample_id"
                ],
                "blank_mean": analysis[
                    "blank_mean"
                ],
                "ic50": fit_result["ic50"],
                "ic50_ci_lower": fit_result[
                    "ic50_ci_lower"
                ],
                "ic50_ci_upper": fit_result[
                    "ic50_ci_upper"
                ],
                "top": fit_result["top"],
                "bottom": fit_result["bottom"],
                "hill_slope": fit_result[
                    "hill_slope"
                ],
                "r_squared": fit_result[
                    "r_squared"
                ],
                "ic50_in_range": fit_result[
                    "ic50_in_range"
                ],
                "crosses_50_percent": (
                    fit_result[
                        "crosses_50_percent"
                    ]
                ),
                "fit_status": (
                    "PASS"
                    if analysis["fit_reliable"]
                    else "CHECK"
                ),
            }
        )

    return pd.DataFrame(rows)


def show_viability_comparison(
    saved_experiments: pd.DataFrame,
) -> None:
    """
    Wyświetla porównanie wybranych eksperymentów
    żywotności zapisanych w SQLite.
    """

    st.subheader(
        "Porównanie eksperymentów żywotności"
    )

    st.write(
        "Każdy eksperyment jest ponownie "
        "normalizowany względem blanku oraz "
        "kontroli CTRL odpowiadającej danemu "
        "stężeniu."
    )

    viability_experiments = (
        find_viability_experiments(
            saved_experiments
        )
    )

    if viability_experiments.empty:
        st.info(
            "W bazie nie znaleziono eksperymentów "
            "zapisanych jako Żywotność."
        )
        return

    experiment_records = (
        viability_experiments.to_dict(
            orient="records"
        )
    )

    selected_experiments = st.multiselect(
        "Wybierz eksperymenty do porównania",
        options=experiment_records,
        format_func=lambda experiment: (
            f"{experiment['experiment_id']} | "
            f"{experiment.get('experiment_date', '')}"
        ),
        key="viability_comparison_selector",
    )

    if len(selected_experiments) < 2:
        st.info(
            "Wybierz co najmniej dwa "
            "eksperymenty żywotności."
        )
        return

    control_prefix = st.text_input(
        "Prefiks kontroli dopasowanej "
        "do stężenia",
        value="CTRL",
        key="viability_comparison_control_prefix",
    )

    include_unreliable = st.checkbox(
        "Uwzględnij dopasowania wymagające "
        "sprawdzenia",
        value=False,
        key="include_unreliable_viability",
    )

    analyses = []
    skipped_experiments = []

    for experiment in selected_experiments:
        experiment_id = experiment[
            "experiment_id"
        ]

        try:
            analysis = (
                analyze_stored_viability_experiment(
                    experiment_id,
                    control_prefix,
                )
            )

            if (
                not include_unreliable
                and not analysis["fit_reliable"]
            ):
                skipped_experiments.append(
                    f"{experiment_id}: "
                    "dopasowanie 4PL wymaga "
                    "sprawdzenia"
                )
                continue

            analyses.append(analysis)

        except (
            ValueError,
            RuntimeError,
            TypeError,
        ) as error:
            skipped_experiments.append(
                f"{experiment_id}: {error}"
            )

    if skipped_experiments:
        st.warning(
            "Pominięte eksperymenty: "
            + "; ".join(skipped_experiments)
        )

    if len(analyses) < 2:
        st.warning(
            "Po ocenie danych pozostał mniej niż "
            "dwa eksperymenty. Zaznacz opcję "
            "uwzględnienia dopasowań wymagających "
            "sprawdzenia albo wybierz inne dane."
        )
        return

    parameter_table = (
        create_viability_parameter_table(
            analyses
        )
    )

    st.subheader("Parametry dopasowania 4PL")

    st.dataframe(
        parameter_table.style.format(
            {
                "blank_mean": "{:.4g}",
                "ic50": "{:.4g}",
                "ic50_ci_lower": "{:.4g}",
                "ic50_ci_upper": "{:.4g}",
                "top": "{:.2f}",
                "bottom": "{:.2f}",
                "hill_slope": "{:.4g}",
                "r_squared": "{:.4f}",
            },
            na_rep="brak",
        ),
        width="stretch",
        hide_index=True,
    )

    comparison_figure = (
        create_viability_comparison_figure(
            analyses
        )
    )

    st.plotly_chart(
        comparison_figure,
        width="stretch",
    )

    ic50_figure = (
        create_ic50_comparison_figure(
            parameter_table
        )
    )

    st.plotly_chart(
        ic50_figure,
        width="stretch",
    )

    summary_frames = [
        analysis["viability_summary"]
        for analysis in analyses
    ]

    combined_summary = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    st.subheader(
        "Zbiorcza tabela żywotności"
    )

    st.dataframe(
        combined_summary.style.format(
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

    parameter_csv = parameter_table.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Pobierz porównanie parametrów 4PL",
        data=parameter_csv,
        file_name=(
            "viability_4pl_comparison.csv"
        ),
        mime="text/csv",
    )

    results_csv = combined_summary.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Pobierz zbiorcze wyniki żywotności",
        data=results_csv,
        file_name=(
            "viability_experiment_comparison.csv"
        ),
        mime="text/csv",
    )