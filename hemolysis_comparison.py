import pandas as pd
import plotly.express as px
import streamlit as st

from database import load_measurements
from hemolysis_astm import (
    calculate_qc_results,
    fit_linear_calibration,
    summarize_absorbance,
)
from hemolysis_assay import (
    calculate_control_result,
    calculate_sample_results,
    calculate_tbhd,
    classify_hemolysis_ids,
)


def analyze_stored_hemolysis_experiment(
    experiment_id: str,
) -> dict:
    """
    Pobiera właściwy eksperyment hemolizy z bazy
    i ponownie wykonuje pełne obliczenia.
    """

    measurements = load_measurements(
        experiment_id
    )

    if measurements.empty:
        raise ValueError(
            f"Eksperyment {experiment_id} "
            "nie zawiera pomiarów."
        )

    absorbance_summary = summarize_absorbance(
        measurements
    )

    classification = classify_hemolysis_ids(
        absorbance_summary
    )

    if len(classification["tbhd"]) != 1:
        raise ValueError(
            "Eksperyment nie jest właściwą płytką "
            "hemolizy: brak dokładnie jednego TBHd."
        )

    if (
        len(classification["negative_control"])
        != 1
    ):
        raise ValueError(
            "Nie znaleziono dokładnie jednej "
            "kontroli negatywnej."
        )

    if (
        len(classification["positive_control"])
        != 1
    ):
        raise ValueError(
            "Nie znaleziono dokładnie jednej "
            "kontroli pozytywnej."
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

    negative_result = calculate_control_result(
        absorbance_summary,
        classification[
            "negative_control"
        ][0],
        calibration_result["slope"],
        calibration_result["intercept"],
        tbhd_result["final_concentration"],
    )

    positive_result = calculate_control_result(
        absorbance_summary,
        classification[
            "positive_control"
        ][0],
        calibration_result["slope"],
        calibration_result["intercept"],
        tbhd_result["final_concentration"],
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

    if sample_results.empty:
        raise ValueError(
            "Nie znaleziono próbek z prawidłowo "
            "dopasowanymi kontrolami IC."
        )

    sample_results = sample_results.copy()
    sample_results["experiment_id"] = (
        experiment_id
    )

    control_results = control_results.copy()
    control_results["experiment_id"] = (
        experiment_id
    )

    calibrator_pass = bool(
        calibration_result["calibrators"][
            "status"
        ]
        .eq("PASS")
        .all()
    )

    qc_pass = bool(
        qc_results["status"]
        .eq("PASS")
        .all()
    )

    tbhd_pass = (
        tbhd_result["status"] == "PASS"
    )

    interference_pass = (
        len(missing_interference_controls)
        == 0
    )

    overall_pass = (
        calibrator_pass
        and qc_pass
        and tbhd_pass
        and interference_pass
    )

    return {
        "experiment_id": experiment_id,
        "sample_results": sample_results,
        "control_results": control_results,
        "calibration_result": (
            calibration_result
        ),
        "qc_results": qc_results,
        "tbhd_result": tbhd_result,
        "missing_interference_controls": (
            missing_interference_controls
        ),
        "calibrator_pass": calibrator_pass,
        "qc_pass": qc_pass,
        "tbhd_pass": tbhd_pass,
        "interference_pass": (
            interference_pass
        ),
        "overall_pass": overall_pass,
    }


def find_proper_hemolysis_experiments(
    saved_experiments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Zwraca zapisane eksperymenty hemolizy,
    które zawierają TBHd, czyli właściwe testy.
    """

    if saved_experiments.empty:
        return saved_experiments.copy()

    if "assay_type" not in saved_experiments:
        return pd.DataFrame()

    hemolysis_experiments = (
        saved_experiments.loc[
            saved_experiments["assay_type"]
            .fillna("")
            .astype(str)
            .str.casefold()
            .eq("hemoliza")
        ]
        .copy()
    )

    proper_ids = []

    for experiment_id in (
        hemolysis_experiments[
            "experiment_id"
        ].tolist()
    ):
        measurements = load_measurements(
            experiment_id
        )

        stored_ids = set(
            measurements["sample_id"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
            .tolist()
        )

        if "tbhd" in stored_ids:
            proper_ids.append(experiment_id)

    return hemolysis_experiments.loc[
        hemolysis_experiments[
            "experiment_id"
        ].isin(proper_ids)
    ].copy()


def create_hemolysis_comparison_chart(
    comparison_data: pd.DataFrame,
    chart_type: str,
):
    """
    Tworzy wspólny wykres hemolizy
    dla wybranych eksperymentów.
    """

    if chart_type == "Punkty i linie":
        figure = px.line(
            comparison_data,
            x="sample_id",
            y="hemolysis_pct",
            color="experiment_label",
            markers=True,
            labels={
                "sample_id": "Próbka",
                "hemolysis_pct": (
                    "Hemoliza [%]"
                ),
                "experiment_label": (
                    "Eksperyment"
                ),
            },
            title=(
                "Porównanie hemolizy "
                "między eksperymentami"
            ),
        )

    else:
        figure = px.bar(
            comparison_data,
            x="sample_id",
            y="hemolysis_pct",
            color="experiment_label",
            barmode="group",
            labels={
                "sample_id": "Próbka",
                "hemolysis_pct": (
                    "Hemoliza [%]"
                ),
                "experiment_label": (
                    "Eksperyment"
                ),
            },
            title=(
                "Porównanie hemolizy "
                "między eksperymentami"
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
    )

    figure.update_layout(
        xaxis_tickangle=-45,
        legend={
            "title": "Eksperyment",
        },
    )

    return figure


def show_hemolysis_comparison(
    saved_experiments: pd.DataFrame,
) -> None:
    """
    Wyświetla porównanie wybranych właściwych
    eksperymentów hemolizy zapisanych w SQLite.
    """

    st.subheader(
        "Porównanie eksperymentów hemolizy"
    )

    st.write(
        "Każdy eksperyment jest ponownie "
        "przeliczany na podstawie zapisanych "
        "surowych pomiarów, kalibratorów, QC, "
        "TBHd oraz kontroli interferencji."
    )

    proper_experiments = (
        find_proper_hemolysis_experiments(
            saved_experiments
        )
    )

    if proper_experiments.empty:
        st.info(
            "W bazie nie znaleziono właściwych "
            "testów hemolizy zawierających TBHd."
        )
        return

    experiment_records = (
        proper_experiments.to_dict(
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
        key="hemolysis_comparison_selector",
    )

    if len(selected_experiments) < 2:
        st.info(
            "Wybierz co najmniej dwa "
            "eksperymenty."
        )
        return

    include_failed = st.checkbox(
        "Uwzględnij eksperymenty niespełniające QC",
        value=False,
        key="include_failed_hemolysis",
    )

    show_controls = st.checkbox(
        "Dodaj NC i PC do tabeli porównawczej",
        value=True,
        key="show_hemolysis_controls",
    )

    chart_type = st.radio(
        "Rodzaj wykresu",
        options=[
            "Punkty i linie",
            "Słupki grupowane",
        ],
        horizontal=True,
        key="hemolysis_comparison_chart_type",
    )

    comparison_rows = []
    control_rows = []
    quality_rows = []
    skipped_experiments = []

    for experiment in selected_experiments:
        experiment_id = experiment[
            "experiment_id"
        ]

        try:
            analysis = (
                analyze_stored_hemolysis_experiment(
                    experiment_id
                )
            )

            experiment_label = (
                experiment_id
            )

            quality_rows.append(
                {
                    "experiment_id": (
                        experiment_id
                    ),
                    "calibrators": (
                        "PASS"
                        if analysis[
                            "calibrator_pass"
                        ]
                        else "FAIL"
                    ),
                    "qc": (
                        "PASS"
                        if analysis["qc_pass"]
                        else "FAIL"
                    ),
                    "tbhd": (
                        "PASS"
                        if analysis["tbhd_pass"]
                        else "FAIL"
                    ),
                    "interference_controls": (
                        "PASS"
                        if analysis[
                            "interference_pass"
                        ]
                        else "FAIL"
                    ),
                    "overall_status": (
                        "PASS"
                        if analysis["overall_pass"]
                        else "FAIL"
                    ),
                    "r_squared": (
                        analysis[
                            "calibration_result"
                        ]["r_squared"]
                    ),
                    "tbhd_concentration": (
                        analysis["tbhd_result"][
                            "final_concentration"
                        ]
                    ),
                }
            )

            if (
                not include_failed
                and not analysis["overall_pass"]
            ):
                skipped_experiments.append(
                    experiment_id
                )
                continue

            sample_data = analysis[
                "sample_results"
            ].copy()

            sample_data["experiment_label"] = (
                experiment_label
            )

            comparison_rows.append(
                sample_data
            )

            if show_controls:
                controls = analysis[
                    "control_results"
                ].copy()

                controls[
                    "experiment_label"
                ] = experiment_label

                control_rows.append(controls)

        except ValueError as error:
            skipped_experiments.append(
                f"{experiment_id}: {error}"
            )

    st.subheader(
        "Kontrola jakości porównywanych eksperymentów"
    )

    quality_table = pd.DataFrame(
        quality_rows
    )

    if not quality_table.empty:
        st.dataframe(
            quality_table.style.format(
                {
                    "r_squared": "{:.4f}",
                    "tbhd_concentration": (
                        "{:.4g}"
                    ),
                },
                na_rep="brak",
            ),
            width="stretch",
            hide_index=True,
        )

    if skipped_experiments:
        st.warning(
            "Pominięte eksperymenty: "
            + "; ".join(skipped_experiments)
        )

    if not comparison_rows:
        st.warning(
            "Po kontroli jakości nie pozostały "
            "eksperymenty do porównania."
        )
        return

    comparison_data = pd.concat(
        comparison_rows,
        ignore_index=True,
    )

    unique_experiments = (
        comparison_data[
            "experiment_label"
        ].nunique()
    )

    if unique_experiments < 2:
        st.warning(
            "Do wykresu pozostał mniej niż "
            "dwa eksperymenty. Włącz wyniki "
            "niespełniające QC albo wybierz "
            "inne eksperymenty."
        )
        return

    st.subheader(
        "Końcowe wyniki próbek"
    )

    st.dataframe(
        comparison_data.style.format(
            {
                "sample_cv_pct": "{:.2f}",
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

    comparison_figure = (
        create_hemolysis_comparison_chart(
            comparison_data,
            chart_type,
        )
    )

    st.plotly_chart(
        comparison_figure,
        width="stretch",
    )

    if show_controls and control_rows:
        controls_data = pd.concat(
            control_rows,
            ignore_index=True,
        )

        st.subheader(
            "Kontrole NC i PC"
        )

        st.dataframe(
            controls_data.style.format(
                {
                    "cv_pct": "{:.2f}",
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

    comparison_csv = (
        comparison_data.to_csv(
            index=False
        ).encode("utf-8")
    )

    st.download_button(
        label=(
            "Pobierz porównanie hemolizy jako CSV"
        ),
        data=comparison_csv,
        file_name=(
            "hemolysis_experiment_comparison.csv"
        ),
        mime="text/csv",
    )