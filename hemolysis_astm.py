import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


CALIBRATOR_CONCENTRATIONS = {
    "Cal1": 0.8,
    "Cal2": 0.4,
    "Cal3": 0.2,
    "Cal4": 0.1,
    "Cal5": 0.05,
    "Cal6": 0.025,
}

QC_CONCENTRATIONS = {
    "QC1": 0.625,
    "QC2": 0.125,
    "QC3": 0.0625,
}


def summarize_absorbance(
    combined_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Oblicza n, średnią, SD i CV absorbancji
    dla każdego identyfikatora.
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

    data = data.loc[
        data["sample_id"].ne("")
    ].dropna(
        subset=["raw_value"]
    )

    if data.empty:
        raise ValueError(
            "Brak opisanych wyników liczbowych."
        )

    summary = (
        data
        .groupby(
            "sample_id",
            as_index=False,
        )
        .agg(
            n=("raw_value", "count"),
            mean_absorbance=("raw_value", "mean"),
            sd_absorbance=("raw_value", "std"),
        )
    )

    summary["cv_pct"] = (
        summary["sd_absorbance"]
        / summary["mean_absorbance"].abs()
        * 100
    )

    summary.loc[
        summary["mean_absorbance"] == 0,
        "cv_pct",
    ] = pd.NA

    return summary


def fit_linear_calibration(
    absorbance_summary: pd.DataFrame,
) -> dict:
    """
    Dopasowuje regresję liniową do Cal1-Cal6:

    absorbancja = slope * concentration + intercept
    """

    calibrator_rows = []

    for calibrator_id, theoretical_value in (
        CALIBRATOR_CONCENTRATIONS.items()
    ):
        selected = absorbance_summary.loc[
            absorbance_summary["sample_id"]
            .str.casefold()
            .eq(calibrator_id.casefold())
        ]

        if selected.empty:
            raise ValueError(
                "Brakuje kalibratora "
                f"{calibrator_id}."
            )

        row = selected.iloc[0].to_dict()
        row["sample_id"] = calibrator_id
        row["theoretical_concentration"] = (
            theoretical_value
        )
        calibrator_rows.append(row)

    calibrators = pd.DataFrame(
        calibrator_rows
    )

    x_values = calibrators[
        "theoretical_concentration"
    ].to_numpy(dtype=float)

    y_values = calibrators[
        "mean_absorbance"
    ].to_numpy(dtype=float)

    slope, intercept = np.polyfit(
        x_values,
        y_values,
        1,
    )

    if slope <= 0:
        raise ValueError(
            "Nachylenie krzywej kalibracyjnej "
            "musi być dodatnie."
        )

    predicted_absorbance = (
        slope * x_values + intercept
    )

    residual_sum_of_squares = float(
        np.sum(
            (
                y_values
                - predicted_absorbance
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

    calibrators[
        "calculated_concentration"
    ] = (
        calibrators["mean_absorbance"]
        - intercept
    ) / slope

    calibrators["pdft_pct"] = (
        (
            calibrators[
                "calculated_concentration"
            ]
            - calibrators[
                "theoretical_concentration"
            ]
        )
        / calibrators[
            "theoretical_concentration"
        ]
        * 100
    )

    calibrators["cv_limit_pct"] = 20.0

    calibrators.loc[
        calibrators["sample_id"].eq("Cal6"),
        "cv_limit_pct",
    ] = 30.0

    calibrators["cv_pass"] = (
        calibrators["cv_pct"]
        <= calibrators["cv_limit_pct"]
    )

    calibrators["pdft_pass"] = (
        calibrators["pdft_pct"].abs()
        <= 20.0
    )

    calibrators["status"] = np.where(
        calibrators["cv_pass"]
        & calibrators["pdft_pass"],
        "PASS",
        "FAIL",
    )

    return {
        "calibrators": calibrators,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
    }


def calculate_qc_results(
    absorbance_summary: pd.DataFrame,
    slope: float,
    intercept: float,
) -> pd.DataFrame:
    """
    Odczytuje stężenia QC z gotowej krzywej.
    QC nie wpływają na dopasowanie regresji.
    """

    qc_rows = []

    for qc_id, theoretical_value in (
        QC_CONCENTRATIONS.items()
    ):
        selected = absorbance_summary.loc[
            absorbance_summary["sample_id"]
            .str.casefold()
            .eq(qc_id.casefold())
        ]

        if selected.empty:
            raise ValueError(
                f"Brakuje kontroli jakości {qc_id}."
            )

        row = selected.iloc[0].to_dict()
        row["sample_id"] = qc_id
        row["theoretical_concentration"] = (
            theoretical_value
        )
        qc_rows.append(row)

    qc_results = pd.DataFrame(qc_rows)

    qc_results[
        "calculated_concentration"
    ] = (
        qc_results["mean_absorbance"]
        - intercept
    ) / slope

    qc_results["pdft_pct"] = (
        (
            qc_results[
                "calculated_concentration"
            ]
            - qc_results[
                "theoretical_concentration"
            ]
        )
        / qc_results[
            "theoretical_concentration"
        ]
        * 100
    )

    qc_results["cv_pass"] = (
        qc_results["cv_pct"] <= 20.0
    )

    qc_results["pdft_pass"] = (
        qc_results["pdft_pct"].abs()
        <= 20.0
    )

    qc_results["status"] = np.where(
        qc_results["cv_pass"]
        & qc_results["pdft_pass"],
        "PASS",
        "FAIL",
    )

    return qc_results


def calculate_blood_qualification(
    absorbance_summary: pd.DataFrame,
    slope: float,
    intercept: float,
) -> pd.DataFrame:
    """
    Oblicza PFH i TBH z uwzględnieniem
    współczynników rozcieńczenia.
    """

    parameters = [
        {
            "sample_id": "PFH",
            "dilution_factor": 2.0,
        },
        {
            "sample_id": "TBH",
            "dilution_factor": 251.0,
        },
    ]

    result_rows = []

    for parameter in parameters:
        sample_id = parameter["sample_id"]

        selected = absorbance_summary.loc[
            absorbance_summary["sample_id"]
            .str.casefold()
            .eq(sample_id.casefold())
        ]

        if selected.empty:
            raise ValueError(
                f"Brakuje wyników dla {sample_id}."
            )

        row = selected.iloc[0].to_dict()

        concentration_from_curve = (
            row["mean_absorbance"]
            - intercept
        ) / slope

        final_concentration = (
            concentration_from_curve
            * parameter["dilution_factor"]
        )

        result_rows.append(
            {
                "sample_id": sample_id,
                "n": int(row["n"]),
                "mean_absorbance": (
                    row["mean_absorbance"]
                ),
                "sd_absorbance": (
                    row["sd_absorbance"]
                ),
                "cv_pct": row["cv_pct"],
                "concentration_from_curve": (
                    concentration_from_curve
                ),
                "dilution_factor": (
                    parameter["dilution_factor"]
                ),
                "final_concentration": (
                    final_concentration
                ),
            }
        )

    qualification = pd.DataFrame(
        result_rows
    )

    qualification["status"] = "INFO"

    qualification.loc[
        qualification["sample_id"].eq("PFH"),
        "status",
    ] = np.where(
        qualification.loc[
            qualification["sample_id"].eq("PFH"),
            "final_concentration",
        ]
        <= 1.0,
        "PASS",
        "FAIL",
    )

    return qualification


def create_calibration_figure(
    calibration_result: dict,
    qc_results: pd.DataFrame,
):
    """
    Tworzy wykres krzywej kalibracyjnej
    z punktami QC.
    """

    calibrators = calibration_result[
        "calibrators"
    ]

    slope = calibration_result["slope"]
    intercept = calibration_result["intercept"]
    r_squared = calibration_result["r_squared"]

    x_min = 0.0
    x_max = max(
        calibrators[
            "theoretical_concentration"
        ].max(),
        qc_results[
            "theoretical_concentration"
        ].max(),
    ) * 1.08

    line_x = np.linspace(
        x_min,
        x_max,
        200,
    )

    line_y = slope * line_x + intercept

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=calibrators[
                "theoretical_concentration"
            ],
            y=calibrators["mean_absorbance"],
            error_y={
                "type": "data",
                "array": calibrators[
                    "sd_absorbance"
                ],
                "visible": True,
            },
            mode="markers+text",
            text=calibrators["sample_id"],
            textposition="top center",
            name="Kalibratory",
            marker={
                "size": 10,
            },
        )
    )

    figure.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name="Regresja liniowa",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=qc_results[
                "theoretical_concentration"
            ],
            y=qc_results["mean_absorbance"],
            error_y={
                "type": "data",
                "array": qc_results[
                    "sd_absorbance"
                ],
                "visible": True,
            },
            mode="markers+text",
            text=qc_results["sample_id"],
            textposition="bottom center",
            name="Kontrole jakości",
            marker={
                "size": 11,
                "symbol": "diamond",
            },
        )
    )

    equation_text = (
        f"y = {slope:.4g}x "
        f"{intercept:+.4g}"
        f"<br>R² = {r_squared:.4f}"
    )

    figure.add_annotation(
        x=0.03 * x_max,
        y=max(line_y) * 0.9,
        text=equation_text,
        showarrow=False,
        align="left",
        bgcolor="white",
        bordercolor="gray",
    )

    figure.update_layout(
        title="Krzywa kalibracyjna hemoglobiny",
        xaxis_title=(
            "Stężenie teoretyczne [mg/mL]"
        ),
        yaxis_title="Średnia absorbancja",
    )

    return figure


def create_qualification_summary(
    calibration_result: dict,
    qc_results: pd.DataFrame,
    qualification: pd.DataFrame,
) -> list:
    """
    Tworzy automatyczne podsumowanie
    kwalifikacji krwi.
    """

    conclusions = []

    failed_calibrators = (
        calibration_result["calibrators"]
        .loc[
            lambda data: data["status"].eq("FAIL"),
            "sample_id",
        ]
        .tolist()
    )

    failed_qc = (
        qc_results.loc[
            qc_results["status"].eq("FAIL"),
            "sample_id",
        ]
        .tolist()
    )

    pfh_row = qualification.loc[
        qualification["sample_id"].eq("PFH")
    ].iloc[0]

    tbh_row = qualification.loc[
        qualification["sample_id"].eq("TBH")
    ].iloc[0]

    r_squared = calibration_result["r_squared"]

    conclusions.append(
        "Krzywa kalibracyjna została opisana "
        f"równaniem y = "
        f"{calibration_result['slope']:.4g}x "
        f"{calibration_result['intercept']:+.4g}, "
        f"przy R² = {r_squared:.4f}."
    )

    if not failed_calibrators and not failed_qc:
        conclusions.append(
            "Wszystkie kalibratory i kontrole "
            "jakości spełniły kryteria PDFT i CV."
        )
    else:
        failed_items = (
            failed_calibrators + failed_qc
        )

        conclusions.append(
            "Kryteriów PDFT lub CV nie spełniły: "
            + ", ".join(failed_items)
            + "."
        )

    pfh_value = float(
        pfh_row["final_concentration"]
    )

    if pfh_value <= 1.0:
        pfh_text = "mieściło się"
    else:
        pfh_text = "nie mieściło się"

    conclusions.append(
        "Stężenie PFH po zastosowaniu "
        "współczynnika rozcieńczenia 2 wynosiło "
        f"{pfh_value:.4g} mg/mL i {pfh_text} "
        "w limicie ≤ 1 mg/mL."
    )

    conclusions.append(
        "Stężenie TBH po zastosowaniu "
        "współczynnika rozcieńczenia 251 "
        f"wynosiło "
        f"{float(tbh_row['final_concentration']):.4g} "
        "mg/mL."
    )

    return conclusions


def show_blood_qualification_module(
    combined_data: pd.DataFrame,
) -> None:
    """
    Wyświetla kwalifikację krwi:
    Cal1-Cal6, QC1-QC3, PFH i TBH.
    """

    st.subheader(
        "Kwalifikacja krwi: PFH i TBH"
    )

    st.write(
        "Krzywa jest wyznaczana wyłącznie "
        "z Cal1–Cal6. QC1–QC3 służą do "
        "niezależnego sprawdzenia krzywej."
    )

    try:
        absorbance_summary = summarize_absorbance(
            combined_data
        )

        calibration_result = (
            fit_linear_calibration(
                absorbance_summary
            )
        )

        qc_results = calculate_qc_results(
            absorbance_summary,
            calibration_result["slope"],
            calibration_result["intercept"],
        )

        qualification = (
            calculate_blood_qualification(
                absorbance_summary,
                calibration_result["slope"],
                calibration_result["intercept"],
            )
        )

        (
            metric_1,
            metric_2,
            metric_3,
        ) = st.columns(3)

        metric_1.metric(
            "Nachylenie",
            f"{calibration_result['slope']:.4g}",
        )

        metric_2.metric(
            "Wyraz wolny",
            (
                f"{calibration_result['intercept']:.4g}"
            ),
        )

        metric_3.metric(
            "R²",
            (
                f"{calibration_result['r_squared']:.4f}"
            ),
        )

        calibration_figure = (
            create_calibration_figure(
                calibration_result,
                qc_results,
            )
        )

        st.plotly_chart(
            calibration_figure,
            width="stretch",
        )

        st.subheader("Standardy kalibracyjne")

        st.dataframe(
            calibration_result[
                "calibrators"
            ].style.format(
                {
                    "theoretical_concentration": (
                        "{:.4g}"
                    ),
                    "mean_absorbance": "{:.4g}",
                    "sd_absorbance": "{:.4g}",
                    "cv_pct": "{:.2f}",
                    "calculated_concentration": (
                        "{:.4g}"
                    ),
                    "pdft_pct": "{:.2f}",
                    "cv_limit_pct": "{:.1f}",
                },
                na_rep="brak",
            ),
            width="stretch",
            hide_index=True,
        )

        st.subheader("Kontrola jakości")

        st.dataframe(
            qc_results.style.format(
                {
                    "theoretical_concentration": (
                        "{:.4g}"
                    ),
                    "mean_absorbance": "{:.4g}",
                    "sd_absorbance": "{:.4g}",
                    "cv_pct": "{:.2f}",
                    "calculated_concentration": (
                        "{:.4g}"
                    ),
                    "pdft_pct": "{:.2f}",
                },
                na_rep="brak",
            ),
            width="stretch",
            hide_index=True,
        )

        st.subheader("PFH i TBH")

        st.dataframe(
            qualification.style.format(
                {
                    "mean_absorbance": "{:.4g}",
                    "sd_absorbance": "{:.4g}",
                    "cv_pct": "{:.2f}",
                    "concentration_from_curve": (
                        "{:.4g}"
                    ),
                    "dilution_factor": "{:.4g}",
                    "final_concentration": "{:.4g}",
                },
                na_rep="brak",
            ),
            width="stretch",
            hide_index=True,
        )

        st.subheader(
            "Podsumowanie i krótkie wnioski"
        )

        conclusions = create_qualification_summary(
            calibration_result,
            qc_results,
            qualification,
        )

        for conclusion in conclusions:
            st.markdown(
                f"- {conclusion}"
            )

        calibrators_csv = calibration_result[
            "calibrators"
        ].to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label=(
                "Pobierz wyniki kalibratorów jako CSV"
            ),
            data=calibrators_csv,
            file_name=(
                "blood_qualification_calibrators.csv"
            ),
            mime="text/csv",
        )

        qc_csv = qc_results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Pobierz wyniki QC jako CSV",
            data=qc_csv,
            file_name="blood_qualification_qc.csv",
            mime="text/csv",
        )

        qualification_csv = qualification.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Pobierz PFH i TBH jako CSV",
            data=qualification_csv,
            file_name="blood_qualification_results.csv",
            mime="text/csv",
        )

    except ValueError as error:
        st.error(str(error))