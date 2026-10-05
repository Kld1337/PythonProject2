import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from hemolysis_astm import (
    calculate_qc_results,
    create_calibration_figure,
    fit_linear_calibration,
    summarize_absorbance,
)


SAMPLE_DILUTION_FACTOR = 18.0
TBHD_DILUTION_FACTOR = 13.5


def calculate_concentration(
    mean_absorbance: float,
    slope: float,
    intercept: float,
) -> float:
    """
    Odczytuje stężenie z równania:
    absorbancja = slope * stężenie + intercept.
    """

    if slope <= 0:
        raise ValueError(
            "Nachylenie krzywej musi być dodatnie."
        )

    return (
        mean_absorbance - intercept
    ) / slope


def classify_hemolysis_ids(
    absorbance_summary: pd.DataFrame,
) -> dict:
    """
    Rozdziela identyfikatory na kalibratory, QC,
    TBHd, NC, PC, kontrole interferencji i próbki.
    """

    sample_ids = (
        absorbance_summary["sample_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    calibrators = [
        sample_id
        for sample_id in sample_ids
        if sample_id.casefold()
        in {
            "cal1",
            "cal2",
            "cal3",
            "cal4",
            "cal5",
            "cal6",
        }
    ]

    qc_ids = [
        sample_id
        for sample_id in sample_ids
        if sample_id.casefold()
        in {
            "qc1",
            "qc2",
            "qc3",
        }
    ]

    tbhd_ids = [
        sample_id
        for sample_id in sample_ids
        if sample_id.casefold() == "tbhd"
    ]

    negative_control_ids = [
        sample_id
        for sample_id in sample_ids
        if sample_id.casefold()
        in {
            "nc",
            "negative control",
            "negative",
        }
    ]

    positive_control_ids = [
        sample_id
        for sample_id in sample_ids
        if sample_id.casefold()
        in {
            "pc",
            "positive control",
            "positive",
        }
    ]

    blank_ids = [
        sample_id
        for sample_id in sample_ids
        if "blank" in sample_id.casefold()
    ]

    interference_ids = [
        sample_id
        for sample_id in sample_ids
        if sample_id.casefold().endswith(" ic")
    ]

    excluded_ids = set(
        calibrators
        + qc_ids
        + tbhd_ids
        + negative_control_ids
        + positive_control_ids
        + blank_ids
        + interference_ids
    )

    sample_ids_proper = [
        sample_id
        for sample_id in sample_ids
        if sample_id not in excluded_ids
    ]

    return {
        "calibrators": calibrators,
        "qc": qc_ids,
        "tbhd": tbhd_ids,
        "negative_control": negative_control_ids,
        "positive_control": positive_control_ids,
        "blank": blank_ids,
        "interference_controls": interference_ids,
        "samples": sample_ids_proper,
    }


def get_single_row(
    absorbance_summary: pd.DataFrame,
    sample_id: str,
) -> pd.Series:
    """
    Zwraca podsumowanie wskazanego identyfikatora.
    """

    selected = absorbance_summary.loc[
        absorbance_summary["sample_id"]
        .str.casefold()
        .eq(sample_id.casefold())
    ]

    if selected.empty:
        raise ValueError(
            f"Nie znaleziono wyników dla '{sample_id}'."
        )

    return selected.iloc[0]


def calculate_tbhd(
    absorbance_summary: pd.DataFrame,
    slope: float,
    intercept: float,
) -> dict:
    """
    Oblicza TBHd z krzywej i stosuje
    współczynnik rozcieńczenia 13,5.
    """

    tbhd_row = get_single_row(
        absorbance_summary,
        "TBHd",
    )

    concentration_from_curve = (
        calculate_concentration(
            float(tbhd_row["mean_absorbance"]),
            slope,
            intercept,
        )
    )

    final_tbhd = (
        concentration_from_curve
        * TBHD_DILUTION_FACTOR
    )

    return {
        "sample_id": "TBHd",
        "n": int(tbhd_row["n"]),
        "mean_absorbance": float(
            tbhd_row["mean_absorbance"]
        ),
        "sd_absorbance": float(
            tbhd_row["sd_absorbance"]
        ),
        "cv_pct": float(
            tbhd_row["cv_pct"]
        ),
        "concentration_from_curve": (
            concentration_from_curve
        ),
        "dilution_factor": (
            TBHD_DILUTION_FACTOR
        ),
        "final_concentration": final_tbhd,
        "status": (
            "PASS"
            if 9.0 <= final_tbhd <= 11.0
            else "FAIL"
        ),
    }


def calculate_control_result(
    absorbance_summary: pd.DataFrame,
    sample_id: str,
    slope: float,
    intercept: float,
    tbhd_concentration: float,
) -> dict:
    """
    Oblicza wynik NC albo PC.
    """

    row = get_single_row(
        absorbance_summary,
        sample_id,
    )

    concentration_from_curve = (
        calculate_concentration(
            float(row["mean_absorbance"]),
            slope,
            intercept,
        )
    )

    diluted_concentration = (
        concentration_from_curve
        * SAMPLE_DILUTION_FACTOR
    )

    hemolysis_pct = (
        diluted_concentration
        / tbhd_concentration
        * 100
    )

    return {
        "sample_id": sample_id,
        "role": (
            "Negative Control"
            if sample_id.casefold() in {
                "nc",
                "negative control",
                "negative",
            }
            else "Positive Control"
        ),
        "n": int(row["n"]),
        "mean_absorbance": float(
            row["mean_absorbance"]
        ),
        "sd_absorbance": float(
            row["sd_absorbance"]
        ),
        "cv_pct": float(row["cv_pct"]),
        "concentration_from_curve": (
            concentration_from_curve
        ),
        "dilution_factor": (
            SAMPLE_DILUTION_FACTOR
        ),
        "corrected_concentration": (
            diluted_concentration
        ),
        "hemolysis_pct": hemolysis_pct,
    }


def calculate_sample_results(
    absorbance_summary: pd.DataFrame,
    sample_ids: list,
    slope: float,
    intercept: float,
    tbhd_concentration: float,
) -> tuple[pd.DataFrame, list]:
    """
    Oblicza wyniki właściwych próbek po odjęciu
    odpowiadających kontroli interferencji.
    """

    result_rows = []
    missing_interference_controls = []

    for sample_id in sample_ids:
        interference_id = f"{sample_id} IC"

        sample_row = get_single_row(
            absorbance_summary,
            sample_id,
        )

        interference_selected = (
            absorbance_summary.loc[
                absorbance_summary["sample_id"]
                .str.casefold()
                .eq(interference_id.casefold())
            ]
        )

        if interference_selected.empty:
            missing_interference_controls.append(
                sample_id
            )
            continue

        interference_row = (
            interference_selected.iloc[0]
        )

        sample_from_curve = (
            calculate_concentration(
                float(
                    sample_row["mean_absorbance"]
                ),
                slope,
                intercept,
            )
        )

        interference_from_curve = (
            calculate_concentration(
                float(
                    interference_row[
                        "mean_absorbance"
                    ]
                ),
                slope,
                intercept,
            )
        )

        sample_concentration = (
            sample_from_curve
            * SAMPLE_DILUTION_FACTOR
        )

        interference_concentration = (
            interference_from_curve
            * SAMPLE_DILUTION_FACTOR
        )

        corrected_pfh = (
            sample_concentration
            - interference_concentration
        )

        hemolysis_pct = (
            corrected_pfh
            / tbhd_concentration
            * 100
        )

        result_rows.append(
            {
                "sample_id": sample_id,
                "role": "Sample",
                "interference_control_id": (
                    interference_id
                ),
                "sample_n": int(
                    sample_row["n"]
                ),
                "sample_mean_absorbance": float(
                    sample_row[
                        "mean_absorbance"
                    ]
                ),
                "sample_sd_absorbance": float(
                    sample_row[
                        "sd_absorbance"
                    ]
                ),
                "sample_cv_pct": float(
                    sample_row["cv_pct"]
                ),
                "ic_n": int(
                    interference_row["n"]
                ),
                "ic_mean_absorbance": float(
                    interference_row[
                        "mean_absorbance"
                    ]
                ),
                "ic_sd_absorbance": float(
                    interference_row[
                        "sd_absorbance"
                    ]
                ),
                "ic_cv_pct": float(
                    interference_row["cv_pct"]
                ),
                "sample_concentration": (
                    sample_concentration
                ),
                "interference_concentration": (
                    interference_concentration
                ),
                "corrected_pfh_concentration": (
                    corrected_pfh
                ),
                "tbhd_concentration": (
                    tbhd_concentration
                ),
                "hemolysis_pct": hemolysis_pct,
                "hemolysis_status": (
                    "PASS"
                    if hemolysis_pct < 5.0
                    else "FAIL"
                ),
            }
        )

    return (
        pd.DataFrame(result_rows),
        missing_interference_controls,
    )


def create_final_hemolysis_chart(
    sample_results: pd.DataFrame,
    control_results: pd.DataFrame,
):
    """
    Tworzy wykres zawierający tylko próbki,
    Negative Control i Positive Control.
    """

    sample_chart = pd.DataFrame()

    if not sample_results.empty:
        sample_chart = sample_results[
            [
                "sample_id",
                "role",
                "hemolysis_pct",
            ]
        ].copy()

    control_chart = control_results[
        [
            "sample_id",
            "role",
            "hemolysis_pct",
        ]
    ].copy()

    chart_data = pd.concat(
        [
            sample_chart,
            control_chart,
        ],
        ignore_index=True,
    )

    chart_data["display_label"] = (
        chart_data["sample_id"]
    )

    figure = px.bar(
        chart_data,
        x="display_label",
        y="hemolysis_pct",
        color="role",
        labels={
            "display_label": "Próbka",
            "hemolysis_pct": "Hemoliza [%]",
            "role": "Rola",
        },
        title=(
            "Hemoliza próbek po korekcji "
            "kontroli interferencji"
        ),
    )

    figure.add_hline(
        y=5,
        line_dash="dash",
        line_color="orange",
        annotation_text="Próg 5%",
    )

    figure.add_hline(
        y=0,
        line_dash="dot",
        line_color="gray",
        annotation_text="0%",
    )

    figure.add_hline(
        y=100,
        line_dash="dot",
        line_color="red",
        annotation_text="100%",
    )

    return figure


def create_assay_conclusions(
    calibration_result: dict,
    qc_results: pd.DataFrame,
    tbhd_result: dict,
    sample_results: pd.DataFrame,
    control_results: pd.DataFrame,
    missing_interference_controls: list,
    cv_limit: float,
) -> list:
    """
    Tworzy podsumowanie właściwego testu hemolizy.
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

    qualification_passed = (
        not failed_calibrators
        and not failed_qc
        and tbhd_result["status"] == "PASS"
        and not missing_interference_controls
    )

    if qualification_passed:
        conclusions.append(
            "Badanie spełniło podstawowe kryteria "
            "kwalifikacyjne krzywej, QC i TBHd."
        )
    else:
        conclusions.append(
            "Badanie nie spełniło wszystkich "
            "kryteriów kwalifikacyjnych."
        )

    conclusions.append(
        "Krzywa kalibracyjna została opisana "
        f"równaniem y = "
        f"{calibration_result['slope']:.4g}x "
        f"{calibration_result['intercept']:+.4g}, "
        f"przy R² = "
        f"{calibration_result['r_squared']:.4f}."
    )

    conclusions.append(
        "TBHd po zastosowaniu współczynnika "
        "rozcieńczenia 13,5 wynosiło "
        f"{tbhd_result['final_concentration']:.4g} "
        "mg/mL i uzyskało status "
        f"{tbhd_result['status']}."
    )

    if failed_calibrators:
        conclusions.append(
            "Kryteriów nie spełniły kalibratory: "
            + ", ".join(failed_calibrators)
            + "."
        )

    if failed_qc:
        conclusions.append(
            "Kryteriów nie spełniły QC: "
            + ", ".join(failed_qc)
            + "."
        )

    if missing_interference_controls:
        conclusions.append(
            "Brak kontroli interferencji dla próbek: "
            + ", ".join(
                missing_interference_controls
            )
            + "."
        )

    if not sample_results.empty:
        for row in sample_results.itertuples(
            index=False
        ):
            comparison = (
                "niższa"
                if row.hemolysis_pct < 5.0
                else "równa lub wyższa"
            )

            conclusions.append(
                f"Hemoliza próbki "
                f"{row.sample_id} wynosiła "
                f"{row.hemolysis_pct:.2f}% "
                f"i była {comparison} od progu 5%."
            )

    for row in control_results.itertuples(
        index=False
    ):
        conclusions.append(
            f"{row.role} ({row.sample_id}) "
            f"wyniosła {row.hemolysis_pct:.2f}%."
        )

    high_cv_samples = []

    if not sample_results.empty:
        high_cv_samples = (
            sample_results.loc[
                (
                    sample_results[
                        "sample_cv_pct"
                    ] > cv_limit
                )
                | (
                    sample_results[
                        "ic_cv_pct"
                    ] > cv_limit
                ),
                "sample_id",
            ]
            .tolist()
        )

    if high_cv_samples:
        conclusions.append(
            f"Limit CV = {cv_limit:.1f}% "
            "przekroczyły próbki lub ich IC: "
            + ", ".join(high_cv_samples)
            + "."
        )

    return conclusions


def show_hemolysis_assay_module(
    combined_data: pd.DataFrame,
) -> None:
    """
    Wyświetla właściwą analizę hemolizy:
    Cal, QC, TBHd, próbki, IC, NC i PC.
    """

    st.subheader(
        "Właściwy test hemolizy"
    )

    cv_limit = st.number_input(
        "Limit CV dla próbek i IC [%]",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
        key="astm_assay_cv_limit",
    )

    try:
        absorbance_summary = summarize_absorbance(
            combined_data
        )

        classification = classify_hemolysis_ids(
            absorbance_summary
        )

        if len(classification["tbhd"]) != 1:
            raise ValueError(
                "Płytka musi zawierać dokładnie "
                "jeden identyfikator TBHd."
            )

        if (
            len(
                classification[
                    "negative_control"
                ]
            )
            != 1
        ):
            raise ValueError(
                "Płytka musi zawierać dokładnie "
                "jedną kontrolę NC."
            )

        if (
            len(
                classification[
                    "positive_control"
                ]
            )
            != 1
        ):
            raise ValueError(
                "Płytka musi zawierać dokładnie "
                "jedną kontrolę PC."
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

        tbhd_result = calculate_tbhd(
            absorbance_summary,
            calibration_result["slope"],
            calibration_result["intercept"],
        )

        negative_result = (
            calculate_control_result(
                absorbance_summary,
                classification[
                    "negative_control"
                ][0],
                calibration_result["slope"],
                calibration_result[
                    "intercept"
                ],
                tbhd_result[
                    "final_concentration"
                ],
            )
        )

        positive_result = (
            calculate_control_result(
                absorbance_summary,
                classification[
                    "positive_control"
                ][0],
                calibration_result["slope"],
                calibration_result[
                    "intercept"
                ],
                tbhd_result[
                    "final_concentration"
                ],
            )
        )

        control_results = pd.DataFrame(
            [
                negative_result,
                positive_result,
            ]
        )

        (
            sample_results,
            missing_interference_controls,
        ) = calculate_sample_results(
            absorbance_summary,
            classification["samples"],
            calibration_result["slope"],
            calibration_result["intercept"],
            tbhd_result["final_concentration"],
        )

        st.subheader("Krzywa kalibracyjna")

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

        st.subheader("Kalibratory")

        st.dataframe(
            calibration_result[
                "calibrators"
            ],
            width="stretch",
            hide_index=True,
        )

        st.subheader("Kontrola jakości")

        st.dataframe(
            qc_results,
            width="stretch",
            hide_index=True,
        )

        st.subheader("TBHd")

        st.dataframe(
            pd.DataFrame([tbhd_result]),
            width="stretch",
            hide_index=True,
        )

        st.subheader(
            "Próbki po korekcji interferencji"
        )

        if sample_results.empty:
            st.warning(
                "Nie obliczono żadnej próbki "
                "z przypisaną kontrolą IC."
            )
        else:
            st.dataframe(
                sample_results.style.format(
                    {
                        "sample_mean_absorbance": (
                            "{:.4g}"
                        ),
                        "sample_sd_absorbance": (
                            "{:.4g}"
                        ),
                        "sample_cv_pct": "{:.2f}",
                        "ic_mean_absorbance": (
                            "{:.4g}"
                        ),
                        "ic_sd_absorbance": (
                            "{:.4g}"
                        ),
                        "ic_cv_pct": "{:.2f}",
                        "sample_concentration": (
                            "{:.4g}"
                        ),
                        "interference_concentration": (
                            "{:.4g}"
                        ),
                        "corrected_pfh_concentration": (
                            "{:.4g}"
                        ),
                        "tbhd_concentration": (
                            "{:.4g}"
                        ),
                        "hemolysis_pct": "{:.2f}",
                    },
                    na_rep="brak",
                ),
                width="stretch",
                hide_index=True,
            )

        st.subheader("Kontrole NC i PC")

        st.dataframe(
            control_results.style.format(
                {
                    "mean_absorbance": "{:.4g}",
                    "sd_absorbance": "{:.4g}",
                    "cv_pct": "{:.2f}",
                    "concentration_from_curve": (
                        "{:.4g}"
                    ),
                    "corrected_concentration": (
                        "{:.4g}"
                    ),
                    "hemolysis_pct": "{:.2f}",
                },
                na_rep="brak",
            ),
            width="stretch",
            hide_index=True,
        )

        final_figure = (
            create_final_hemolysis_chart(
                sample_results,
                control_results,
            )
        )

        st.plotly_chart(
            final_figure,
            width="stretch",
        )

        st.subheader(
            "Podsumowanie i krótkie wnioski"
        )

        conclusions = create_assay_conclusions(
            calibration_result,
            qc_results,
            tbhd_result,
            sample_results,
            control_results,
            missing_interference_controls,
            cv_limit,
        )

        for conclusion in conclusions:
            st.markdown(
                f"- {conclusion}"
            )

        if not sample_results.empty:
            results_csv = sample_results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label=(
                    "Pobierz wyniki hemolizy "
                    "jako CSV"
                ),
                data=results_csv,
                file_name=(
                    "hemolysis_assay_results.csv"
                ),
                mime="text/csv",
            )

    except ValueError as error:
        st.error(str(error))