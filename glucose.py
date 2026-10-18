import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DEFAULT_STANDARD_CONCENTRATIONS = {
    "Std1": 18.18,
    "Std2": 36.36,
    "Std3": 54.54,
    "Std4": 72.72,
}


def split_glucose_sample_id(
    sample_id: str,
) -> tuple[str, int]:
    """
    Rozdziela identyfikator próbki i numer
    niezależnego przygotowania.

    Cztery końcowe cyfry są traktowane jako
    numer przygotowania dodany przez Varioskan.

    Przykłady:
    QC/1487/20260001 -> QC/1487/2026, 1
    QC/1487/20260002 -> QC/1487/2026, 2
    Std10001         -> Std1, 1
    QC0001           -> QC, 1
    Blank1           -> Blank, 1
    """

    cleaned_id = str(sample_id).strip()

    if not cleaned_id:
        return "", 0

    suffix_match = re.search(
        r"(\d{4})$",
        cleaned_id,
    )

    if suffix_match is not None:
        preparation_text = suffix_match.group(1)

        base_sample_id = cleaned_id[:-4].strip()

        preparation_number = int(
            preparation_text
        )

        if not base_sample_id:
            base_sample_id = cleaned_id

        return (
            base_sample_id,
            preparation_number,
        )

    if cleaned_id.casefold() == "blank1":
        return "Blank", 1

    return cleaned_id, 1


def classify_glucose_sample(
    sample_id: str,
) -> str:
    """
    Klasyfikuje oczyszczony identyfikator jako:
    blank, standard, curve_qc albo sample.
    """

    normalized_id = (
        str(sample_id)
        .strip()
        .casefold()
    )

    if normalized_id.startswith("blank"):
        return "blank"

    if normalized_id in {
        "std1",
        "std2",
        "std3",
        "std4",
    }:
        return "standard"

    if normalized_id == "qc":
        return "curve_qc"

    return "sample"


def prepare_glucose_raw_data(
    combined_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Przygotowuje surowe dane do analizy glukozy.

    Wymagane kolumny:
    sample_id
    raw_value
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
            + ", ".join(
                sorted(missing_columns)
            )
            + "."
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
            "Brak opisanych wyników liczbowych "
            "do analizy glukozy."
        )

    parsed_identifiers = (
        data["sample_id"]
        .map(split_glucose_sample_id)
    )

    data["base_sample_id"] = (
        parsed_identifiers.map(
            lambda result: result[0]
        )
    )

    data["preparation_number"] = (
        parsed_identifiers.map(
            lambda result: result[1]
        )
    )

    data["sample_role"] = (
        data["base_sample_id"]
        .map(classify_glucose_sample)
    )

    return data


def summarize_glucose_preparations(
    prepared_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Oblicza statystyki techniczne osobno dla
    każdego niezależnego przygotowania.

    Przykład:
    QC/1487/20260001 -> przygotowanie 1
    QC/1487/20260002 -> przygotowanie 2
    QC/1487/20260003 -> przygotowanie 3
    """

    preparation_summary = (
        prepared_data
        .groupby(
            [
                "base_sample_id",
                "preparation_number",
                "sample_role",
            ],
            as_index=False,
        )
        .agg(
            technical_n=(
                "raw_value",
                "count",
            ),
            mean_absorbance=(
                "raw_value",
                "mean",
            ),
            sd_absorbance=(
                "raw_value",
                "std",
            ),
            minimum_absorbance=(
                "raw_value",
                "min",
            ),
            maximum_absorbance=(
                "raw_value",
                "max",
            ),
        )
    )

    preparation_summary[
        "technical_cv_pct"
    ] = (
        preparation_summary[
            "sd_absorbance"
        ]
        / preparation_summary[
            "mean_absorbance"
        ].abs()
        * 100
    )

    preparation_summary.loc[
        preparation_summary[
            "mean_absorbance"
        ].eq(0),
        "technical_cv_pct",
    ] = pd.NA

    return preparation_summary


def assign_standard_concentrations(
    preparation_summary: pd.DataFrame,
    standard_concentrations: dict | None = None,
) -> pd.DataFrame:
    """
    Przypisuje standardom wartości nominalne
    wyrażone jako glukoza w ug/mL.

    Domyślne wartości:
    Std1 = 18.18
    Std2 = 36.36
    Std3 = 54.54
    Std4 = 72.72
    """

    if standard_concentrations is None:
        standard_concentrations = (
            DEFAULT_STANDARD_CONCENTRATIONS
        )

    summary = preparation_summary.copy()

    summary[
        "nominal_concentration_ug_ml"
    ] = summary["base_sample_id"].map(
        standard_concentrations
    )

    blank_mask = summary[
        "sample_role"
    ].eq("blank")

    summary.loc[
        blank_mask,
        "nominal_concentration_ug_ml",
    ] = 0.0

    return summary


def validate_glucose_plate_structure(
    preparation_summary: pd.DataFrame,
) -> None:
    """
    Sprawdza obecność blanku i standardów Std1-Std4.
    """

    available_ids = set(
        preparation_summary[
            "base_sample_id"
        ]
        .astype(str)
        .str.strip()
        .tolist()
    )

    required_ids = {
        "Blank",
        "Std1",
        "Std2",
        "Std3",
        "Std4",
    }

    missing_ids = (
        required_ids - available_ids
    )

    if missing_ids:
        raise ValueError(
            "Brakuje wymaganych identyfikatorów: "
            + ", ".join(
                sorted(missing_ids)
            )
            + "."
        )


def calculate_glucose_calibration(
    preparation_summary: pd.DataFrame,
    agreement_lower_pct: float = 98.0,
    agreement_upper_pct: float = 102.0,
    cv_limit_pct: float = 15.0,
) -> dict:
    """
    Odejmuje średni blank i wykonuje liniową
    regresję dla Std1-Std4 oraz punktu 0.

    Kryterium zgodności:
    98-102% wartości obliczonej względem nominalnej.
    """

    validate_glucose_plate_structure(
        preparation_summary
    )

    blank_rows = preparation_summary.loc[
        preparation_summary[
            "sample_role"
        ].eq("blank")
    ]

    blank_mean = float(
        blank_rows[
            "mean_absorbance"
        ].mean()
    )

    standards = preparation_summary.loc[
        preparation_summary[
            "sample_role"
        ].eq("standard")
    ].copy()

    standards[
        "corrected_absorbance"
    ] = (
        standards["mean_absorbance"]
        - blank_mean
    )

    blank_point = pd.DataFrame(
        [
            {
                "base_sample_id": "Blank",
                "preparation_number": 1,
                "sample_role": "blank",
                "technical_n": int(
                    blank_rows[
                        "technical_n"
                    ].sum()
                ),
                "mean_absorbance": blank_mean,
                "sd_absorbance": float(
                    blank_rows[
                        "sd_absorbance"
                    ].mean()
                ),
                "minimum_absorbance": float(
                    blank_rows[
                        "minimum_absorbance"
                    ].min()
                ),
                "maximum_absorbance": float(
                    blank_rows[
                        "maximum_absorbance"
                    ].max()
                ),
                "technical_cv_pct": float(
                    blank_rows[
                        "technical_cv_pct"
                    ].mean()
                ),
                "nominal_concentration_ug_ml": 0.0,
                "corrected_absorbance": 0.0,
            }
        ]
    )

    calibration_data = pd.concat(
        [
            standards,
            blank_point,
        ],
        ignore_index=True,
    )

    calibration_data = (
        calibration_data
        .sort_values(
            "nominal_concentration_ug_ml"
        )
        .reset_index(drop=True)
    )

    x_values = calibration_data[
        "nominal_concentration_ug_ml"
    ].to_numpy(dtype=float)

    y_values = calibration_data[
        "corrected_absorbance"
    ].to_numpy(dtype=float)

    slope, intercept = np.polyfit(
        x_values,
        y_values,
        1,
    )

    if slope <= 0:
        raise ValueError(
            "Nachylenie krzywej glukozy "
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

    calibration_data[
        "calculated_concentration_ug_ml"
    ] = (
        calibration_data[
            "corrected_absorbance"
        ]
        - intercept
    ) / slope

    calibration_data[
        "agreement_pct"
    ] = pd.NA

    nonzero_standard_mask = (
        calibration_data[
            "nominal_concentration_ug_ml"
        ] > 0
    )

    calibration_data.loc[
        nonzero_standard_mask,
        "agreement_pct",
    ] = (
        calibration_data.loc[
            nonzero_standard_mask,
            "calculated_concentration_ug_ml",
        ]
        / calibration_data.loc[
            nonzero_standard_mask,
            "nominal_concentration_ug_ml",
        ]
        * 100
    )

    calibration_data[
        "agreement_status"
    ] = "NOT_APPLICABLE"

    calibration_data.loc[
        nonzero_standard_mask,
        "agreement_status",
    ] = np.where(
        (
            calibration_data.loc[
                nonzero_standard_mask,
                "agreement_pct",
            ]
            >= agreement_lower_pct
        )
        & (
            calibration_data.loc[
                nonzero_standard_mask,
                "agreement_pct",
            ]
            <= agreement_upper_pct
        ),
        "PASS",
        "FAIL",
    )

    calibration_data["cv_status"] = np.where(
        calibration_data[
            "technical_cv_pct"
        ].fillna(np.inf)
        <= cv_limit_pct,
        "PASS",
        "FAIL",
    )

    calibration_data[
        "overall_status"
    ] = np.where(
        (
            calibration_data[
                "agreement_status"
            ].isin(
                [
                    "PASS",
                    "NOT_APPLICABLE",
                ]
            )
        )
        & calibration_data[
            "cv_status"
        ].eq("PASS"),
        "PASS",
        "FAIL",
    )

    curve_x = np.linspace(
        0,
        float(
            calibration_data[
                "nominal_concentration_ug_ml"
            ].max()
        ),
        300,
    )

    curve_y = (
        slope * curve_x
        + intercept
    )

    return {
        "calibration_data": calibration_data,
        "blank_mean": blank_mean,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "curve_x": curve_x,
        "curve_y": curve_y,
        "agreement_lower_pct": (
            agreement_lower_pct
        ),
        "agreement_upper_pct": (
            agreement_upper_pct
        ),
        "cv_limit_pct": cv_limit_pct,
    }
def calculate_curve_qc(
    preparation_summary: pd.DataFrame,
    calibration_result: dict,
    nominal_qc_concentration_ug_ml: float = 50.0,
    agreement_lower_pct: float = 98.0,
    agreement_upper_pct: float = 102.0,
    cv_limit_pct: float = 15.0,
) -> pd.DataFrame:
    """
    Oblicza wynik kontroli jakości krzywej.

    QC nie uczestniczy w dopasowaniu krzywej.
    Jest odczytywane z gotowego równania regresji.

    Zgodność [%] =
    stężenie obliczone / stężenie nominalne * 100
    """

    if nominal_qc_concentration_ug_ml <= 0:
        raise ValueError(
            "Nominalne stężenie QC musi być "
            "większe od zera."
        )

    qc_results = preparation_summary.loc[
        preparation_summary[
            "sample_role"
        ].eq("curve_qc")
    ].copy()

    if qc_results.empty:
        return pd.DataFrame()

    slope = calibration_result["slope"]
    intercept = calibration_result["intercept"]
    blank_mean = calibration_result["blank_mean"]

    qc_results[
        "corrected_absorbance"
    ] = (
        qc_results["mean_absorbance"]
        - blank_mean
    )

    qc_results[
        "nominal_concentration_ug_ml"
    ] = float(
        nominal_qc_concentration_ug_ml
    )

    qc_results[
        "calculated_concentration_ug_ml"
    ] = (
        qc_results[
            "corrected_absorbance"
        ]
        - intercept
    ) / slope

    qc_results["agreement_pct"] = (
        qc_results[
            "calculated_concentration_ug_ml"
        ]
        / qc_results[
            "nominal_concentration_ug_ml"
        ]
        * 100
    )

    qc_results["agreement_status"] = np.where(
        (
            qc_results["agreement_pct"]
            >= agreement_lower_pct
        )
        & (
            qc_results["agreement_pct"]
            <= agreement_upper_pct
        ),
        "PASS",
        "FAIL",
    )

    qc_results["cv_status"] = np.where(
        qc_results[
            "technical_cv_pct"
        ].fillna(np.inf)
        <= cv_limit_pct,
        "PASS",
        "FAIL",
    )

    qc_results["overall_status"] = np.where(
        (
            qc_results[
                "agreement_status"
            ].eq("PASS")
        )
        & (
            qc_results[
                "cv_status"
            ].eq("PASS")
        ),
        "PASS",
        "FAIL",
    )

    return qc_results.reset_index(
        drop=True
    )


def calculate_glucose_preparation_results(
    preparation_summary: pd.DataFrame,
    calibration_result: dict,
    dilution_factor: float,
    cv_limit_pct: float = 15.0,
) -> pd.DataFrame:
    """
    Oblicza stężenie glukozy osobno dla każdego
    niezależnego przygotowania próbki.

    Wynik końcowy [g/L] =
    stężenie z krzywej [ug/mL]
    * współczynnik rozcieńczenia
    / 1000
    """

    if dilution_factor <= 0:
        raise ValueError(
            "Współczynnik rozcieńczenia musi być "
            "większy od zera."
        )

    samples = preparation_summary.loc[
        preparation_summary[
            "sample_role"
        ].eq("sample")
    ].copy()

    if samples.empty:
        raise ValueError(
            "Nie znaleziono właściwych próbek "
            "do obliczenia stężenia glukozy."
        )

    slope = calibration_result["slope"]
    intercept = calibration_result["intercept"]
    blank_mean = calibration_result["blank_mean"]

    samples[
        "corrected_absorbance"
    ] = (
        samples["mean_absorbance"]
        - blank_mean
    )

    samples[
        "concentration_from_curve_ug_ml"
    ] = (
        samples[
            "corrected_absorbance"
        ]
        - intercept
    ) / slope

    samples["dilution_factor"] = float(
        dilution_factor
    )

    samples[
        "concentration_after_dilution_ug_ml"
    ] = (
        samples[
            "concentration_from_curve_ug_ml"
        ]
        * samples["dilution_factor"]
    )

    samples[
        "final_concentration_g_l"
    ] = (
        samples[
            "concentration_after_dilution_ug_ml"
        ]
        / 1000
    )

    samples[
        "concentration_sd_from_absorbance_g_l"
    ] = (
        samples["sd_absorbance"]
        / abs(slope)
        * samples["dilution_factor"]
        / 1000
    )

    minimum_calibration_absorbance = float(
        calibration_result[
            "calibration_data"
        ]["corrected_absorbance"].min()
    )

    maximum_calibration_absorbance = float(
        calibration_result[
            "calibration_data"
        ]["corrected_absorbance"].max()
    )

    samples["range_status"] = "IN_RANGE"

    samples.loc[
        samples["corrected_absorbance"]
        < minimum_calibration_absorbance,
        "range_status",
    ] = "BELOW_RANGE"

    samples.loc[
        samples["corrected_absorbance"]
        > maximum_calibration_absorbance,
        "range_status",
    ] = "ABOVE_RANGE"

    samples["technical_cv_status"] = np.where(
        samples[
            "technical_cv_pct"
        ].fillna(np.inf)
        <= cv_limit_pct,
        "PASS",
        "FAIL",
    )

    samples[
        "preparation_status"
    ] = np.where(
        (
            samples[
                "technical_cv_status"
            ].eq("PASS")
        )
        & (
            samples[
                "range_status"
            ].eq("IN_RANGE")
        ),
        "PASS",
        "CHECK",
    )

    selected_columns = [
        "base_sample_id",
        "preparation_number",
        "technical_n",
        "mean_absorbance",
        "sd_absorbance",
        "technical_cv_pct",
        "technical_cv_status",
        "corrected_absorbance",
        "concentration_from_curve_ug_ml",
        "dilution_factor",
        "concentration_after_dilution_ug_ml",
        "final_concentration_g_l",
        "concentration_sd_from_absorbance_g_l",
        "range_status",
        "preparation_status",
    ]

    return (
        samples[selected_columns]
        .sort_values(
            [
                "base_sample_id",
                "preparation_number",
            ]
        )
        .reset_index(drop=True)
    )


def calculate_curve_qc(
    preparation_summary: pd.DataFrame,
    calibration_result: dict,
    nominal_qc_concentration_ug_ml: float = 50.0,
    agreement_lower_pct: float = 98.0,
    agreement_upper_pct: float = 102.0,
    cv_limit_pct: float = 15.0,
) -> pd.DataFrame:
    """
    Oblicza wynik kontroli jakości krzywej.

    QC nie uczestniczy w dopasowaniu krzywej.
    Jest odczytywane z gotowego równania regresji.

    Zgodność [%] =
    stężenie obliczone / stężenie nominalne * 100
    """

    if nominal_qc_concentration_ug_ml <= 0:
        raise ValueError(
            "Nominalne stężenie QC musi być "
            "większe od zera."
        )

    qc_results = preparation_summary.loc[
        preparation_summary[
            "sample_role"
        ].eq("curve_qc")
    ].copy()

    if qc_results.empty:
        return pd.DataFrame()

    slope = calibration_result["slope"]
    intercept = calibration_result["intercept"]
    blank_mean = calibration_result["blank_mean"]

    qc_results[
        "corrected_absorbance"
    ] = (
        qc_results["mean_absorbance"]
        - blank_mean
    )

    qc_results[
        "nominal_concentration_ug_ml"
    ] = float(
        nominal_qc_concentration_ug_ml
    )

    qc_results[
        "calculated_concentration_ug_ml"
    ] = (
        qc_results[
            "corrected_absorbance"
        ]
        - intercept
    ) / slope

    qc_results["agreement_pct"] = (
        qc_results[
            "calculated_concentration_ug_ml"
        ]
        / qc_results[
            "nominal_concentration_ug_ml"
        ]
        * 100
    )

    qc_results["agreement_status"] = np.where(
        (
            qc_results["agreement_pct"]
            >= agreement_lower_pct
        )
        & (
            qc_results["agreement_pct"]
            <= agreement_upper_pct
        ),
        "PASS",
        "FAIL",
    )

    qc_results["cv_status"] = np.where(
        qc_results[
            "technical_cv_pct"
        ].fillna(np.inf)
        <= cv_limit_pct,
        "PASS",
        "FAIL",
    )

    qc_results["overall_status"] = np.where(
        (
            qc_results[
                "agreement_status"
            ].eq("PASS")
        )
        & (
            qc_results[
                "cv_status"
            ].eq("PASS")
        ),
        "PASS",
        "FAIL",
    )

    return qc_results.reset_index(
        drop=True
    )


def calculate_glucose_preparation_results(
    preparation_summary: pd.DataFrame,
    calibration_result: dict,
    dilution_factor: float,
    cv_limit_pct: float = 15.0,
) -> pd.DataFrame:
    """
    Oblicza stężenie glukozy osobno dla każdego
    niezależnego przygotowania próbki.

    Wynik końcowy [g/L] =
    stężenie z krzywej [ug/mL]
    * współczynnik rozcieńczenia
    / 1000
    """

    if dilution_factor <= 0:
        raise ValueError(
            "Współczynnik rozcieńczenia musi być "
            "większy od zera."
        )

    samples = preparation_summary.loc[
        preparation_summary[
            "sample_role"
        ].eq("sample")
    ].copy()

    if samples.empty:
        raise ValueError(
            "Nie znaleziono właściwych próbek "
            "do obliczenia stężenia glukozy."
        )

    slope = calibration_result["slope"]
    intercept = calibration_result["intercept"]
    blank_mean = calibration_result["blank_mean"]

    samples[
        "corrected_absorbance"
    ] = (
        samples["mean_absorbance"]
        - blank_mean
    )

    samples[
        "concentration_from_curve_ug_ml"
    ] = (
        samples[
            "corrected_absorbance"
        ]
        - intercept
    ) / slope

    samples["dilution_factor"] = float(
        dilution_factor
    )

    samples[
        "concentration_after_dilution_ug_ml"
    ] = (
        samples[
            "concentration_from_curve_ug_ml"
        ]
        * samples["dilution_factor"]
    )

    samples[
        "final_concentration_g_l"
    ] = (
        samples[
            "concentration_after_dilution_ug_ml"
        ]
        / 1000
    )

    samples[
        "concentration_sd_from_absorbance_g_l"
    ] = (
        samples["sd_absorbance"]
        / abs(slope)
        * samples["dilution_factor"]
        / 1000
    )

    minimum_calibration_absorbance = float(
        calibration_result[
            "calibration_data"
        ]["corrected_absorbance"].min()
    )

    maximum_calibration_absorbance = float(
        calibration_result[
            "calibration_data"
        ]["corrected_absorbance"].max()
    )

    samples["range_status"] = "IN_RANGE"

    samples.loc[
        samples["corrected_absorbance"]
        < minimum_calibration_absorbance,
        "range_status",
    ] = "BELOW_RANGE"

    samples.loc[
        samples["corrected_absorbance"]
        > maximum_calibration_absorbance,
        "range_status",
    ] = "ABOVE_RANGE"

    samples["technical_cv_status"] = np.where(
        samples[
            "technical_cv_pct"
        ].fillna(np.inf)
        <= cv_limit_pct,
        "PASS",
        "FAIL",
    )

    samples[
        "preparation_status"
    ] = np.where(
        (
            samples[
                "technical_cv_status"
            ].eq("PASS")
        )
        & (
            samples[
                "range_status"
            ].eq("IN_RANGE")
        ),
        "PASS",
        "CHECK",
    )

    selected_columns = [
        "base_sample_id",
        "preparation_number",
        "technical_n",
        "mean_absorbance",
        "sd_absorbance",
        "technical_cv_pct",
        "technical_cv_status",
        "corrected_absorbance",
        "concentration_from_curve_ug_ml",
        "dilution_factor",
        "concentration_after_dilution_ug_ml",
        "final_concentration_g_l",
        "concentration_sd_from_absorbance_g_l",
        "range_status",
        "preparation_status",
    ]

    return (
        samples[selected_columns]
        .sort_values(
            [
                "base_sample_id",
                "preparation_number",
            ]
        )
        .reset_index(drop=True)
    )


def summarize_glucose_samples(
    preparation_results: pd.DataFrame,
    between_preparation_cv_limit_pct: float = 15.0,
) -> pd.DataFrame:
    """
    Tworzy końcowy wynik próbki na podstawie
    wyników niezależnych przygotowań.

    Przykład:
    przygotowanie 1 -> średni wynik z 3 dołków
    przygotowanie 2 -> średni wynik z 3 dołków
    przygotowanie 3 -> średni wynik z 3 dołków

    Następnie obliczana jest średnia, SD i CV
    z trzech wyników przygotowań.
    """

    if preparation_results.empty:
        raise ValueError(
            "Brak wyników przygotowań "
            "do utworzenia podsumowania."
        )

    sample_summary = (
        preparation_results
        .groupby(
            "base_sample_id",
            as_index=False,
        )
        .agg(
            preparation_n=(
                "preparation_number",
                "nunique",
            ),
            mean_concentration_g_l=(
                "final_concentration_g_l",
                "mean",
            ),
            sd_between_preparations_g_l=(
                "final_concentration_g_l",
                "std",
            ),
            minimum_concentration_g_l=(
                "final_concentration_g_l",
                "min",
            ),
            maximum_concentration_g_l=(
                "final_concentration_g_l",
                "max",
            ),
            minimum_technical_cv_pct=(
                "technical_cv_pct",
                "min",
            ),
            maximum_technical_cv_pct=(
                "technical_cv_pct",
                "max",
            ),
            all_preparations_pass=(
                "preparation_status",
                lambda values: all(
                    value == "PASS"
                    for value in values
                ),
            ),
        )
    )

    sample_summary[
        "cv_between_preparations_pct"
    ] = (
        sample_summary[
            "sd_between_preparations_g_l"
        ]
        / sample_summary[
            "mean_concentration_g_l"
        ].abs()
        * 100
    )

    sample_summary.loc[
        sample_summary[
            "mean_concentration_g_l"
        ].eq(0),
        "cv_between_preparations_pct",
    ] = pd.NA

    sample_summary[
        "between_preparation_cv_status"
    ] = "NOT_APPLICABLE"

    multiple_preparation_mask = (
        sample_summary[
            "preparation_n"
        ] > 1
    )

    sample_summary.loc[
        multiple_preparation_mask,
        "between_preparation_cv_status",
    ] = np.where(
        sample_summary.loc[
            multiple_preparation_mask,
            "cv_between_preparations_pct",
        ].fillna(np.inf)
        <= between_preparation_cv_limit_pct,
        "PASS",
        "FAIL",
    )

    sample_summary[
        "overall_status"
    ] = np.where(
        (
            sample_summary[
                "all_preparations_pass"
            ]
        )
        & (
            sample_summary[
                "between_preparation_cv_status"
            ].isin(
                [
                    "PASS",
                    "NOT_APPLICABLE",
                ]
            )
        ),
        "PASS",
        "CHECK",
    )

    return sample_summary.sort_values(
        "base_sample_id"
    ).reset_index(drop=True)


def create_glucose_analysis(
    combined_data: pd.DataFrame,
    standard_concentrations: dict | None = None,
    nominal_qc_concentration_ug_ml: float = 50.0,
    dilution_factor: float = 1.0,
    agreement_lower_pct: float = 98.0,
    agreement_upper_pct: float = 102.0,
    technical_cv_limit_pct: float = 15.0,
    between_preparation_cv_limit_pct: float = 15.0,
) -> dict:
    """
    Wykonuje pełną dotychczasową analizę:

    1. rozpoznanie identyfikatorów,
    2. statystyki przygotowań,
    3. krzywa standardowa,
    4. ocena QC,
    5. stężenia każdego przygotowania,
    6. końcowe wyniki próbek.
    """

    prepared_data = prepare_glucose_raw_data(
        combined_data
    )

    preparation_summary = (
        summarize_glucose_preparations(
            prepared_data
        )
    )

    preparation_summary = (
        assign_standard_concentrations(
            preparation_summary,
            standard_concentrations,
        )
    )

    calibration_result = (
        calculate_glucose_calibration(
            preparation_summary,
            agreement_lower_pct=(
                agreement_lower_pct
            ),
            agreement_upper_pct=(
                agreement_upper_pct
            ),
            cv_limit_pct=(
                technical_cv_limit_pct
            ),
        )
    )

    qc_results = calculate_curve_qc(
        preparation_summary,
        calibration_result,
        nominal_qc_concentration_ug_ml=(
            nominal_qc_concentration_ug_ml
        ),
        agreement_lower_pct=(
            agreement_lower_pct
        ),
        agreement_upper_pct=(
            agreement_upper_pct
        ),
        cv_limit_pct=(
            technical_cv_limit_pct
        ),
    )

    preparation_results = (
        calculate_glucose_preparation_results(
            preparation_summary,
            calibration_result,
            dilution_factor=(
                dilution_factor
            ),
            cv_limit_pct=(
                technical_cv_limit_pct
            ),
        )
    )

    sample_summary = (
        summarize_glucose_samples(
            preparation_results,
            between_preparation_cv_limit_pct=(
                between_preparation_cv_limit_pct
            ),
        )
    )

    standards_pass = bool(
        calibration_result[
            "calibration_data"
        ]["overall_status"]
        .eq("PASS")
        .all()
    )

    qc_pass = True

    if not qc_results.empty:
        qc_pass = bool(
            qc_results[
                "overall_status"
            ]
            .eq("PASS")
            .all()
        )

    samples_pass = bool(
        sample_summary[
            "overall_status"
        ]
        .eq("PASS")
        .all()
    )

    return {
        "prepared_data": prepared_data,
        "preparation_summary": (
            preparation_summary
        ),
        "calibration_result": (
            calibration_result
        ),
        "qc_results": qc_results,
        "preparation_results": (
            preparation_results
        ),
        "sample_summary": sample_summary,
        "standards_pass": standards_pass,
        "qc_pass": qc_pass,
        "samples_pass": samples_pass,
        "overall_pass": (
            standards_pass
            and qc_pass
            and samples_pass
        ),
    }
def create_glucose_sample_figure(
    sample_summary: pd.DataFrame,
    chart_type: str,
    specification_lower_g_l: float,
    specification_upper_g_l: float,
    y_axis_maximum_g_l: float,
):
    """
    Tworzy wykres końcowego stężenia glukozy
    z zaznaczonym zakresem specyfikacji.
    """

    if (
        specification_lower_g_l
        >= specification_upper_g_l
    ):
        raise ValueError(
            "Dolna granica specyfikacji musi być "
            "mniejsza od górnej granicy."
        )

    if (
        y_axis_maximum_g_l
        <= specification_upper_g_l
    ):
        raise ValueError(
            "Maksimum osi Y musi być większe "
            "od górnej granicy specyfikacji."
        )

    chart_data = sample_summary.copy()

    if chart_type == "Punkty":
        figure = px.scatter(
            chart_data,
            x="base_sample_id",
            y="mean_concentration_g_l",
            error_y=(
                "sd_between_preparations_g_l"
            ),
            color="overall_status",
            labels={
                "base_sample_id": "Próbka",
                "mean_concentration_g_l": (
                    "Stężenie glukozy [g/L]"
                ),
                "overall_status": "Status",
            },
            title=(
                "Końcowe stężenie glukozy "
                "w próbkach"
            ),
        )

        figure.update_traces(
            marker={
                "size": 11,
            }
        )

    else:
        figure = px.bar(
            chart_data,
            x="base_sample_id",
            y="mean_concentration_g_l",
            error_y=(
                "sd_between_preparations_g_l"
            ),
            color="overall_status",
            labels={
                "base_sample_id": "Próbka",
                "mean_concentration_g_l": (
                    "Stężenie glukozy [g/L]"
                ),
                "overall_status": "Status",
            },
            title=(
                "Końcowe stężenie glukozy "
                "w próbkach"
            ),
        )

    figure.add_hrect(
        y0=specification_lower_g_l,
        y1=specification_upper_g_l,
        fillcolor="green",
        opacity=0.15,
        line_width=0,
        annotation_text=(
            "Zakres specyfikacji: "
            f"{specification_lower_g_l:.2f}–"
            f"{specification_upper_g_l:.2f} g/L"
        ),
        annotation_position="top left",
    )

    figure.add_hline(
        y=specification_lower_g_l,
        line_dash="dash",
        line_color="green",
    )

    figure.add_hline(
        y=specification_upper_g_l,
        line_dash="dash",
        line_color="green",
    )

    figure.update_layout(
        xaxis_tickangle=-45,
        yaxis={
            "title": "Stężenie glukozy [g/L]",
            "range": [
                0,
                y_axis_maximum_g_l,
            ],
        },
        margin={
            "l": 70,
            "r": 40,
            "t": 100,
            "b": 120,
        },
    )

    return figure

def create_glucose_calibration_figure(
    calibration_result: dict,
):
    """
    Tworzy liniową krzywą standardową glukozy
    i opcjonalnie nanosi kontrolę jakości QC.
    """

    calibration_data = (
        calibration_result[
            "calibration_data"
        ]
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=calibration_data[
                "nominal_concentration_ug_ml"
            ],
            y=calibration_data[
                "corrected_absorbance"
            ],
            error_y={
                "type": "data",
                "array": calibration_data[
                    "sd_absorbance"
                ],
                "visible": True,
            },
            mode="markers+text",
            text=calibration_data[
                "base_sample_id"
            ],
            textposition="top center",
            name="Standardy",
            marker={
                "size": 10,
                "color": "black",
            },
        )
    )

    figure.add_trace(
        go.Scatter(
            x=calibration_result["curve_x"],
            y=calibration_result["curve_y"],
            mode="lines",
            name="Regresja liniowa",
            line={
                "width": 3,
            },
        )
    )

    if not qc_results.empty:
        figure.add_trace(
            go.Scatter(
                x=qc_results[
                    "nominal_concentration_ug_ml"
                ],
                y=qc_results[
                    "corrected_absorbance"
                ],
                error_y={
                    "type": "data",
                    "array": qc_results[
                        "sd_absorbance"
                    ],
                    "visible": True,
                },
                mode="markers+text",
                text=qc_results[
                    "base_sample_id"
                ],
                textposition="bottom center",
                name="QC",
                marker={
                    "size": 12,
                    "symbol": "diamond",
                    "color": "darkorange",
                },
            )
        )

    equation_text = (
        f"y = "
        f"{calibration_result['slope']:.6g}x "
        f"{calibration_result['intercept']:+.6g}"
        f"<br>R² = "
        f"{calibration_result['r_squared']:.4f}"
    )

    maximum_y = float(
        np.nanmax(
            calibration_result["curve_y"]
        )
    )

    figure.add_annotation(
        x=4,
        y=maximum_y * 0.90,
        text=equation_text,
        showarrow=False,
        align="left",
        bgcolor="white",
        bordercolor="gray",
    )

    figure.update_layout(
        title="Krzywa standardowa glukozy",
        xaxis_title=(
            "Stężenie nominalne glukozy [µg/mL]"
        ),
        yaxis_title=(
            "Absorbancja po korekcji blanku"
        ),
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
            "t": 110,
            "b": 70,
        },
    )

    return figure
def create_glucose_conclusions(
    analysis_result: dict,
) -> list:
    """
    Tworzy krótkie podsumowanie wyników
    oznaczenia glukozy.
    """

    calibration_result = analysis_result[
        "calibration_result"
    ]

    qc_results = analysis_result[
        "qc_results"
    ]

    sample_summary = analysis_result[
        "sample_summary"
    ]

    conclusions = []

    conclusions.append(
        "Krzywa standardowa została opisana "
        f"równaniem y = "
        f"{calibration_result['slope']:.6g}x "
        f"{calibration_result['intercept']:+.6g}, "
        f"przy R² = "
        f"{calibration_result['r_squared']:.4f}."
    )

    if analysis_result["standards_pass"]:
        conclusions.append(
            "Standardy spełniły kryteria "
            "zgodności 98–102% oraz CV."
        )
    else:
        failed_standards = (
            calibration_result[
                "calibration_data"
            ]
            .loc[
                lambda data: (
                    data["overall_status"]
                    .eq("FAIL")
                ),
                "base_sample_id",
            ]
            .astype(str)
            .tolist()
        )

        conclusions.append(
            "Kryteriów standardów nie spełniły: "
            + ", ".join(failed_standards)
            + "."
        )

    if qc_results.empty:
        conclusions.append(
            "Na płytce nie znaleziono kontroli "
            "jakości krzywej QC."
        )
    else:
        qc_row = qc_results.iloc[0]

        conclusions.append(
            "Kontrola jakości o stężeniu "
            f"nominalnym "
            f"{float(qc_row['nominal_concentration_ug_ml']):.2f} "
            "µg/mL osiągnęła stężenie obliczone "
            f"{float(qc_row['calculated_concentration_ug_ml']):.2f} "
            "µg/mL, zgodność "
            f"{float(qc_row['agreement_pct']):.2f}% "
            "oraz CV "
            f"{float(qc_row['technical_cv_pct']):.2f}%."
        )

        conclusions.append(
            "Kontrola jakości uzyskała status "
            f"{qc_row['overall_status']}."
        )

    for row in sample_summary.itertuples(
        index=False
    ):
        cv_value = (
            row.cv_between_preparations_pct
        )

        if pd.isna(cv_value):
            cv_text = (
                "CV między przygotowaniami "
                "nie dotyczy jednego przygotowania"
            )
        else:
            cv_text = (
                "CV między przygotowaniami "
                f"wyniosło {cv_value:.2f}%"
            )

        conclusions.append(
            f"Dla próbki {row.base_sample_id} "
            f"średnie stężenie glukozy wyniosło "
            f"{row.mean_concentration_g_l:.4g} g/L; "
            f"{cv_text}."
        )

    return conclusions

def show_glucose_module(
    combined_data: pd.DataFrame,
) -> None:
    """
    Wyświetla analizę stężenia glukozy.
    """

    st.subheader(
        "Oznaczanie stężenia glukozy"
    )

    st.write(
        "Analiza uwzględnia niezależne "
        "przygotowania oznaczone końcówkami "
        "0001–0003 oraz ich powtórzenia "
        "techniczne."
    )

    st.subheader(
        "Stężenia standardów"
    )

    (
        standard_column_1,
        standard_column_2,
        standard_column_3,
        standard_column_4,
    ) = st.columns(4)

    std1 = standard_column_1.number_input(
        "Std1 [µg/mL]",
        min_value=0.0,
        value=18.18,
        step=0.01,
        key="glucose_std1",
    )

    std2 = standard_column_2.number_input(
        "Std2 [µg/mL]",
        min_value=0.0,
        value=36.36,
        step=0.01,
        key="glucose_std2",
    )

    std3 = standard_column_3.number_input(
        "Std3 [µg/mL]",
        min_value=0.0,
        value=54.54,
        step=0.01,
        key="glucose_std3",
    )

    std4 = standard_column_4.number_input(
        "Std4 [µg/mL]",
        min_value=0.0,
        value=72.72,
        step=0.01,
        key="glucose_std4",
    )

    standard_concentrations = {
        "Std1": std1,
        "Std2": std2,
        "Std3": std3,
        "Std4": std4,
    }

    settings_column_1, (
        settings_column_2
    ), settings_column_3 = st.columns(3)

    nominal_qc = (
        settings_column_1.number_input(
            "Nominalne stężenie QC [µg/mL]",
            min_value=0.01,
            value=50.0,
            step=1.0,
            key="glucose_nominal_qc",
        )
    )

    dilution_factor = (
        settings_column_2.number_input(
            "Współczynnik rozcieńczenia próbek",
            min_value=0.01,
            value=1.0,
            step=1.0,
            key="glucose_dilution_factor",
        )
    )

    cv_limit = (
        settings_column_3.number_input(
            "Limit CV [%]",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=1.0,
            key="glucose_cv_limit",
        )
    )
    st.subheader(
        "Zakres specyfikacji i ustawienia wykresu"
    )

    (
        specification_column_1,
        specification_column_2,
        specification_column_3,
    ) = st.columns(3)

    specification_lower_g_l = (
        specification_column_1.number_input(
            "Dolna granica specyfikacji [g/L]",
            min_value=0.0,
            max_value=1000.0,
            value=1.71,
            step=0.01,
            format="%.2f",
            key="glucose_specification_lower",
        )
    )

    specification_upper_g_l = (
        specification_column_2.number_input(
            "Górna granica specyfikacji [g/L]",
            min_value=0.0,
            max_value=1000.0,
            value=1.89,
            step=0.01,
            format="%.2f",
            key="glucose_specification_upper",
        )
    )

    y_axis_maximum_g_l = (
        specification_column_3.number_input(
            "Maksimum osi Y [g/L]",
            min_value=0.1,
            max_value=1000.0,
            value=2.0,
            step=0.1,
            format="%.2f",
            key="glucose_y_axis_maximum",
        )
    )
    chart_type = st.radio(
        "Rodzaj wykresu próbek",
        options=[
            "Punkty",
            "Słupki",
        ],
        horizontal=True,
        key="glucose_chart_type",
    )

    try:
        analysis_result = (
            create_glucose_analysis(
                combined_data,
                standard_concentrations=(
                    standard_concentrations
                ),
                nominal_qc_concentration_ug_ml=(
                    nominal_qc
                ),
                dilution_factor=(
                    dilution_factor
                ),
                agreement_lower_pct=98.0,
                agreement_upper_pct=102.0,
                technical_cv_limit_pct=(
                    cv_limit
                ),
                between_preparation_cv_limit_pct=(
                    cv_limit
                ),
            )
        )

        calibration_result = analysis_result[
            "calibration_result"
        ]

        qc_results = analysis_result[
            "qc_results"
        ]

        preparation_results = (
            analysis_result[
                "preparation_results"
            ]
        )

        sample_summary = analysis_result[
            "sample_summary"
        ]

        (
            metric_1,
            metric_2,
            metric_3,
            metric_4,
        ) = st.columns(4)

        metric_1.metric(
            "Średni blank",
            (
                f"{calibration_result['blank_mean']:.4f}"
            ),
        )

        metric_2.metric(
            "Nachylenie",
            (
                f"{calibration_result['slope']:.6g}"
            ),
        )

        metric_3.metric(
            "Wyraz wolny",
            (
                f"{calibration_result['intercept']:.6g}"
            ),
        )

        metric_4.metric(
            "R²",
            (
                f"{calibration_result['r_squared']:.4f}"
            ),
        )

        calibration_figure = (
            create_glucose_calibration_figure(
                calibration_result,
                qc_results,
            )
        )

        st.plotly_chart(
            calibration_figure,
            width="stretch",
        )

        st.subheader(
            "Standardy"
        )

        st.dataframe(
            calibration_result[
                "calibration_data"
            ].style.format(
                {
                    "mean_absorbance": "{:.4f}",
                    "sd_absorbance": "{:.4f}",
                    "technical_cv_pct": "{:.2f}",
                    "nominal_concentration_ug_ml": (
                        "{:.2f}"
                    ),
                    "corrected_absorbance": (
                        "{:.4f}"
                    ),
                    "calculated_concentration_ug_ml": (
                        "{:.2f}"
                    ),
                    "agreement_pct": "{:.2f}",
                },
                na_rep="brak",
            ),
            width="stretch",
            hide_index=True,
        )

        st.subheader(
            "Kontrola jakości krzywej"
        )

        if qc_results.empty:
            st.info(
                "Na płytce nie znaleziono "
                "kontroli QC."
            )
        else:
            st.dataframe(
                qc_results.style.format(
                    {
                        "mean_absorbance": "{:.4f}",
                        "sd_absorbance": "{:.4f}",
                        "technical_cv_pct": "{:.2f}",
                        "nominal_concentration_ug_ml": (
                            "{:.2f}"
                        ),
                        "corrected_absorbance": (
                            "{:.4f}"
                        ),
                        "calculated_concentration_ug_ml": (
                            "{:.2f}"
                        ),
                        "agreement_pct": "{:.2f}",
                    },
                    na_rep="brak",
                ),
                width="stretch",
                hide_index=True,
            )

        st.subheader(
            "Wyniki niezależnych przygotowań"
        )

        st.dataframe(
            preparation_results.style.format(
                {
                    "mean_absorbance": "{:.4f}",
                    "sd_absorbance": "{:.4f}",
                    "technical_cv_pct": "{:.2f}",
                    "corrected_absorbance": (
                        "{:.4f}"
                    ),
                    "concentration_from_curve_ug_ml": (
                        "{:.2f}"
                    ),
                    "dilution_factor": "{:.2f}",
                    "concentration_after_dilution_ug_ml": (
                        "{:.2f}"
                    ),
                    "final_concentration_g_l": (
                        "{:.4f}"
                    ),
                    "concentration_sd_from_absorbance_g_l": (
                        "{:.4f}"
                    ),
                },
                na_rep="brak",
            ),
            width="stretch",
            hide_index=True,
        )

        st.subheader(
            "Końcowe wyniki próbek"
        )

        st.dataframe(
            sample_summary.style.format(
                {
                    "mean_concentration_g_l": (
                        "{:.4f}"
                    ),
                    "sd_between_preparations_g_l": (
                        "{:.4f}"
                    ),
                    "minimum_concentration_g_l": (
                        "{:.4f}"
                    ),
                    "maximum_concentration_g_l": (
                        "{:.4f}"
                    ),
                    "minimum_technical_cv_pct": (
                        "{:.2f}"
                    ),
                    "maximum_technical_cv_pct": (
                        "{:.2f}"
                    ),
                    "cv_between_preparations_pct": (
                        "{:.2f}"
                    ),
                },
                na_rep="brak",
            ),
            width="stretch",
            hide_index=True,
        )

        sample_figure = (
            create_glucose_sample_figure(
                sample_summary,
                chart_type,
                specification_lower_g_l,
                specification_upper_g_l,
                y_axis_maximum_g_l,
            )
        )

        st.plotly_chart(
            sample_figure,
            width="stretch",
        )

        st.subheader(
            "Podsumowanie i krótkie wnioski"
        )

        conclusions = (
            create_glucose_conclusions(
                analysis_result
            )
        )

        for conclusion in conclusions:
            st.markdown(
                f"- {conclusion}"
            )

        if analysis_result["overall_pass"]:
            st.success(
                "Analiza spełniła przyjęte "
                "kryteria standardów, QC i CV."
            )
        else:
            st.warning(
                "Analiza nie spełniła wszystkich "
                "przyjętych kryteriów."
            )

    except (
        ValueError,
        TypeError,
        ZeroDivisionError,
    ) as error:
        st.error(str(error))
