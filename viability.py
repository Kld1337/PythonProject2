import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scipy.optimize import curve_fit


def extract_numeric_concentration(
    sample_ids: pd.Series,
) -> pd.Series:
    """
    Wyciąga liczbowe stężenie z identyfikatorów takich jak:
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

def four_parameter_logistic(
    concentration: np.ndarray,
    bottom: float,
    top: float,
    ic50: float,
    hill_slope: float,
) -> np.ndarray:
    """
    Czteroparametrowy model logistyczny 4PL
    dla malejącej odpowiedzi zależnej od dawki.
    """

    return bottom + (
        (top - bottom)
        / (
            1
            + (
                concentration / ic50
            ) ** hill_slope
        )
    )


def fit_viability_4pl(
    viability_summary: pd.DataFrame,
) -> dict:
    """
    Dopasowuje model 4PL do średniej żywotności
    w zależności od stężenia.
    """

    fit_data = viability_summary[
        [
            "concentration",
            "mean_viability_pct",
            "sd_viability_pct",
        ]
    ].copy()

    fit_data["concentration"] = pd.to_numeric(
        fit_data["concentration"],
        errors="coerce",
    )

    fit_data["mean_viability_pct"] = pd.to_numeric(
        fit_data["mean_viability_pct"],
        errors="coerce",
    )

    fit_data["sd_viability_pct"] = pd.to_numeric(
        fit_data["sd_viability_pct"],
        errors="coerce",
    )

    fit_data = fit_data.dropna(
        subset=[
            "concentration",
            "mean_viability_pct",
        ]
    )

    fit_data = fit_data.loc[
        fit_data["concentration"] > 0
    ].copy()

    fit_data = fit_data.sort_values(
        "concentration"
    ).reset_index(drop=True)

    if fit_data["concentration"].nunique() < 4:
        raise ValueError(
            "Do dopasowania modelu 4PL potrzebne są "
            "co najmniej cztery różne dodatnie stężenia."
        )

    minimum_response = float(
        fit_data["mean_viability_pct"].min()
    )

    maximum_response = float(
        fit_data["mean_viability_pct"].max()
    )

    crosses_50_percent = (
        minimum_response <= 50
        and maximum_response >= 50
    )

    x_values = fit_data[
        "concentration"
    ].to_numpy(dtype=float)

    y_values = fit_data[
        "mean_viability_pct"
    ].to_numpy(dtype=float)

    initial_bottom = max(
        -10.0,
        minimum_response,
    )

    initial_top = min(
        150.0,
        maximum_response,
    )

    closest_to_50_index = int(
        np.argmin(
            np.abs(y_values - 50)
        )
    )

    initial_ic50 = float(
        x_values[closest_to_50_index]
    )

    fitted_parameters, covariance_matrix = curve_fit(
        four_parameter_logistic,
        x_values,
        y_values,
        p0=[
            initial_bottom,
            initial_top,
            initial_ic50,
            1.0,
        ],
        bounds=(
            [
                -50.0,
                20.0,
                float(x_values.min()) / 100.0,
                0.05,
            ],
            [
                100.0,
                200.0,
                float(x_values.max()) * 100.0,
                10.0,
            ],
        ),
        maxfev=50000,
    )

    (
        bottom,
        top,
        ic50,
        hill_slope,
    ) = fitted_parameters

    predicted_values = four_parameter_logistic(
        x_values,
        bottom,
        top,
        ic50,
        hill_slope,
    )

    residual_sum_of_squares = float(
        np.sum(
            (
                y_values
                - predicted_values
            ) ** 2
        )
    )

    total_sum_of_squares = float(
        np.sum(
            (
                y_values
                - np.mean(y_values)
            ) ** 2
        )
    )

    if total_sum_of_squares == 0:
        r_squared = np.nan
    else:
        r_squared = (
            1
            - residual_sum_of_squares
            / total_sum_of_squares
        )

    parameter_errors = np.sqrt(
        np.diag(covariance_matrix)
    )

    ic50_standard_error = float(
        parameter_errors[2]
    )

    ic50_ci_lower = max(
        0.0,
        float(
            ic50
            - 1.96 * ic50_standard_error
        ),
    )

    ic50_ci_upper = float(
        ic50
        + 1.96 * ic50_standard_error
    )

    minimum_concentration = float(
        x_values.min()
    )

    maximum_concentration = float(
        x_values.max()
    )

    ic50_in_range = (
        minimum_concentration
        <= ic50
        <= maximum_concentration
    )

    curve_x = np.logspace(
        np.log10(minimum_concentration),
        np.log10(maximum_concentration),
        300,
    )

    curve_y = four_parameter_logistic(
        curve_x,
        bottom,
        top,
        ic50,
        hill_slope,
    )

    return {
        "fit_data": fit_data,
        "bottom": float(bottom),
        "top": float(top),
        "ic50": float(ic50),
        "hill_slope": float(hill_slope),
        "r_squared": float(r_squared),
        "ic50_ci_lower": ic50_ci_lower,
        "ic50_ci_upper": ic50_ci_upper,
        "ic50_in_range": ic50_in_range,
        "crosses_50_percent": crosses_50_percent,
        "minimum_concentration": (
            minimum_concentration
        ),
        "maximum_concentration": (
            maximum_concentration
        ),
        "curve_x": curve_x,
        "curve_y": curve_y,
    }


def create_4pl_viability_figure(
    fit_result: dict,
):
    """
    Tworzy wykres krzywej 4PL z punktami
    średniej żywotności i odchyleniem standardowym.
    """

    fit_data = fit_result["fit_data"]

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=fit_data["concentration"],
            y=fit_data["mean_viability_pct"],
            error_y={
                "type": "data",
                "array": (
                    fit_data[
                        "sd_viability_pct"
                    ]
                ),
                "visible": True,
            },
            mode="markers",
            name="Średnia ± SD",
            marker={
                "size": 10,
                "color": "black",
            },
        )
    )

    figure.add_trace(
        go.Scatter(
            x=fit_result["curve_x"],
            y=fit_result["curve_y"],
            mode="lines",
            name="Dopasowanie 4PL",
            line={
                "width": 3,
            },
        )
    )

    figure.add_hline(
        y=50,
        line_dash="dot",
        line_color="gray",
        annotation_text="50% żywotności",
    )

    if fit_result["ic50_in_range"]:
        ic50_value = fit_result["ic50"]

        visible_y_values = np.concatenate(
            [
                fit_result["curve_y"],
                fit_data[
                    "mean_viability_pct"
                ].to_numpy(dtype=float),
            ]
        )

        ic50_line_min = float(
            np.nanmin(visible_y_values)
        )

        ic50_line_max = float(
            np.nanmax(visible_y_values)
        )

        figure.add_trace(
            go.Scatter(
                x=[
                    ic50_value,
                    ic50_value,
                ],
                y=[
                    ic50_line_min,
                    ic50_line_max,
                ],
                mode="lines",
                name=(
                    f"IC50 = {ic50_value:.4g}"
                ),
                line={
                    "color": "red",
                    "dash": "dash",
                    "width": 2,
                },
                hovertemplate=(
                    f"IC50 = {ic50_value:.4g}"
                    "<extra></extra>"
                ),
            )
        )
    minimum_concentration = fit_result[
        "minimum_concentration"
    ]

    maximum_concentration = fit_result[
        "maximum_concentration"
    ]

    x_axis_minimum = np.log10(
        minimum_concentration * 0.8
    )

    x_axis_maximum = np.log10(
        maximum_concentration * 1.2
    )

    figure.update_layout(
        title="Krzywa dawka–odpowiedź 4PL",
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
            "rangemode": "tozero",
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        margin={
            "l": 70,
            "r": 40,
            "t": 100,
            "b": 70,
        },
    )

    return figure
def create_viability_summary(
    viability_summary: pd.DataFrame,
    fit_result: dict | None = None,
    cv_limit: float = 15.0,
) -> list:
    """
    Tworzy krótkie, automatyczne podsumowanie
    wyników analizy żywotności.
    """

    if viability_summary.empty:
        return [
            "Brak wyników do utworzenia podsumowania."
        ]

    summary_data = viability_summary.copy()

    summary_data["concentration"] = pd.to_numeric(
        summary_data["concentration"],
        errors="coerce",
    )

    summary_data["mean_viability_pct"] = pd.to_numeric(
        summary_data["mean_viability_pct"],
        errors="coerce",
    )

    summary_data["viability_cv_pct"] = pd.to_numeric(
        summary_data["viability_cv_pct"],
        errors="coerce",
    )

    summary_data = summary_data.dropna(
        subset=[
            "concentration",
            "mean_viability_pct",
        ]
    )

    if summary_data.empty:
        return [
            "Brak liczbowych wyników do utworzenia "
            "podsumowania."
        ]

    summary_data = summary_data.sort_values(
        "concentration"
    ).reset_index(drop=True)

    conclusions = []

    concentration_count = (
        summary_data["concentration"].nunique()
    )

    minimum_concentration = float(
        summary_data["concentration"].min()
    )

    maximum_concentration = float(
        summary_data["concentration"].max()
    )

    conclusions.append(
        "Analiza obejmowała "
        f"{concentration_count} poziomów stężenia "
        f"w zakresie od {minimum_concentration:g} "
        f"do {maximum_concentration:g}."
    )

    minimum_row = summary_data.loc[
        summary_data["mean_viability_pct"].idxmin()
    ]

    minimum_viability = float(
        minimum_row["mean_viability_pct"]
    )

    minimum_viability_concentration = float(
        minimum_row["concentration"]
    )

    conclusions.append(
        "Najniższą średnią żywotność "
        f"({minimum_viability:.2f}%) "
        "zaobserwowano dla stężenia "
        f"{minimum_viability_concentration:g}."
    )

    below_50 = summary_data.loc[
        summary_data["mean_viability_pct"] < 50
    ]

    if below_50.empty:
        conclusions.append(
            "W badanym zakresie stężeń średnia "
            "żywotność nie spadła poniżej 50%."
        )
    else:
        first_below_50 = (
            below_50
            .sort_values("concentration")
            .iloc[0]
        )

        conclusions.append(
            "Pierwsze badane stężenie, dla którego "
            "średnia żywotność była niższa niż 50%, "
            "wynosiło "
            f"{float(first_below_50['concentration']):g} "
            "i dawało "
            f"{float(first_below_50['mean_viability_pct']):.2f}% "
            "żywotności."
        )

    high_cv_mask = (
        summary_data["viability_cv_pct"]
        .gt(cv_limit)
        .fillna(False)
    )

    high_cv_count = int(high_cv_mask.sum())

    if high_cv_count == 0:
        conclusions.append(
            "Żaden poziom stężenia nie przekroczył "
            f"przyjętego limitu CV = {cv_limit:.1f}% "
            "dla żywotności."
        )
    else:
        high_cv_concentrations = (
            summary_data.loc[
                high_cv_mask,
                "concentration",
            ]
            .sort_values()
            .map(lambda value: f"{value:g}")
            .tolist()
        )

        conclusions.append(
            f"{high_cv_count} poziomów stężenia "
            f"przekroczyło limit CV = {cv_limit:.1f}%: "
            + ", ".join(high_cv_concentrations)
            + ". Wyniki te wymagają sprawdzenia."
        )

    if fit_result is not None:
        ic50 = float(fit_result["ic50"])
        r_squared = float(
            fit_result["r_squared"]
        )

        conclusions.append(
            "Model 4PL oszacował IC50 na "
            f"{ic50:.4g}, przy R² = "
            f"{r_squared:.4f}."
        )

        fit_is_reliable = (
            fit_result["crosses_50_percent"]
            and fit_result["ic50_in_range"]
            and not np.isnan(r_squared)
            and r_squared >= 0.8
        )

        if fit_is_reliable:
            conclusions.append(
                "Dane obejmują wartości żywotności "
                "po obu stronach 50%, IC50 znajduje się "
                "w badanym zakresie, a dopasowanie "
                "spełnia przyjęte kryterium R² ≥ 0,80."
            )
        else:
            warning_reasons = []

            if not fit_result["crosses_50_percent"]:
                warning_reasons.append(
                    "brak wyników po obu stronach 50%"
                )

            if not fit_result["ic50_in_range"]:
                warning_reasons.append(
                    "IC50 poza badanym zakresem"
                )

            if (
                np.isnan(r_squared)
                or r_squared < 0.8
            ):
                warning_reasons.append(
                    "R² poniżej 0,80"
                )

            conclusions.append(
                "Oszacowanie IC50 należy interpretować "
                "ostrożnie: "
                + ", ".join(warning_reasons)
                + "."
            )

    return conclusions

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
        st.divider()

        st.subheader(
            "Dopasowanie 4PL i oszacowanie IC50"
        )
        fit_result_for_summary = None
        enable_4pl = st.checkbox(
           "Wykonaj dopasowanie 4PL",
            value=True,
            key="enable_viability_4pl",
        )

        if enable_4pl:
            try:
                fit_result = fit_viability_4pl(
                    viability_summary
                )
                fit_result_for_summary = fit_result
                (
                    fit_column_1,
                    fit_column_2,
                    fit_column_3,
                    fit_column_4,
                ) = st.columns(4)

                fit_column_1.metric(
                    "IC50",
                    f"{fit_result['ic50']:.4g}",
                )

                fit_column_2.metric(
                    "Top",
                    f"{fit_result['top']:.2f}%",
                )

                fit_column_3.metric(
                    "Bottom",
                    f"{fit_result['bottom']:.2f}%",
                )

                fit_column_4.metric(
                    "R²",
                    f"{fit_result['r_squared']:.4f}",
                )

                st.write(
                    "**Hill slope:** "
                    f"{fit_result['hill_slope']:.4g}"
                )

                st.write(
                    "**Przybliżony 95% CI dla IC50:** "
                    f"{fit_result['ic50_ci_lower']:.4g}"
                    "–"
                    f"{fit_result['ic50_ci_upper']:.4g}"
                )

                if not fit_result["crosses_50_percent"]:
                    st.warning(
                        "Średnia żywotność nie obejmuje "
                        "wartości po obu stronach 50%. "
                        "Oszacowanie IC50 może być "
                        "niewiarygodne."
                    )

                if not fit_result["ic50_in_range"]:
                    st.warning(
                        "Oszacowane IC50 znajduje się "
                        "poza zakresem badanych stężeń. "
                        "Nie należy raportować tej "
                        "wartości jako wiarygodnego IC50."
                    )

                if (
                    not np.isnan(
                        fit_result["r_squared"]
                    )
                    and fit_result["r_squared"] < 0.8
                ):
                    st.warning(
                        "Dopasowanie 4PL ma R² poniżej "
                        "0,80. Sprawdź zakres stężeń "
                        "i przebieg krzywej."
                    )

                four_pl_figure = (
                    create_4pl_viability_figure(
                        fit_result
                    )
                )

                st.plotly_chart(
                    four_pl_figure,
                    width="stretch",
                )

                fit_parameters = pd.DataFrame(
                    [
                        {
                            "parameter": "Bottom",
                            "value": fit_result["bottom"],
                        },
                        {
                            "parameter": "Top",
                            "value": fit_result["top"],
                        },
                        {
                            "parameter": "IC50",
                            "value": fit_result["ic50"],
                        },
                        {
                            "parameter": "Hill slope",
                            "value": fit_result[
                                "hill_slope"
                            ],
                        },
                        {
                            "parameter": "R squared",
                            "value": fit_result[
                                "r_squared"
                            ],
                        },
                        {
                            "parameter": "IC50 CI lower",
                            "value": fit_result[
                                "ic50_ci_lower"
                            ],
                        },
                        {
                            "parameter": "IC50 CI upper",
                            "value": fit_result[
                                "ic50_ci_upper"
                            ],
                        },
                    ]
                )

                fit_csv = fit_parameters.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label=(
                        "Pobierz parametry 4PL jako CSV"
                    ),
                    data=fit_csv,
                    file_name=(
                        "viability_4pl_parameters.csv"
                    ),
                    mime="text/csv",
                )

            except (
                RuntimeError,
                ValueError,
                TypeError,
            ) as fit_error:
                st.warning(
                    "Nie udało się uzyskać "
                    "wiarygodnego dopasowania 4PL."
                )

                st.info(str(fit_error))
        st.divider()

        st.subheader(
            "Podsumowanie i krótkie wnioski"
        )

        summary_cv_limit = st.number_input(
            "Limit CV używany w podsumowaniu [%]",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=1.0,
            key="viability_summary_cv_limit",
        )

        viability_conclusions = (
            create_viability_summary(
                viability_summary,
                fit_result=fit_result_for_summary,
                cv_limit=summary_cv_limit,
            )
        )

        for conclusion in viability_conclusions:
            st.markdown(
                f"- {conclusion}"
            )

        st.caption(
            "Podsumowanie ma charakter opisowy. "
            "Interpretację biologiczną i kryteria "
            "akceptacji należy zweryfikować zgodnie "
            "z protokołem eksperymentu."
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