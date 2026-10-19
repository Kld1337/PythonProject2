import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from excluded_results import show_exclusion_editor


TE1_STANDARD_CONCENTRATIONS = {
    "Std0001": 0.200,
    "Std0002": 0.100,
    "Std0003": 0.040,
    "Std0004": 0.008,
    "Std0005": 0.004,
    "Blank1": 0.000,
}

TX3_STANDARD_CONCENTRATIONS = {
    "Std0006": 1.000,
    "Std0007": 0.500,
    "Std0008": 0.300,
    "Std0009": 0.200,
    "Std0010": 0.100,
    "Blank2": 0.000,
}


def _validate_input(combined_data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "well",
        "plate_row",
        "plate_column",
        "sample_id",
        "raw_value",
    }
    missing = required - set(combined_data.columns)
    if missing:
        raise ValueError(
            "Brakuje wymaganych kolumn: "
            + ", ".join(sorted(missing))
            + "."
        )

    data = combined_data.copy()
    data["plate_row"] = (
        data["plate_row"].fillna("").astype(str).str.strip().str.upper()
    )
    data["plate_column"] = pd.to_numeric(
        data["plate_column"], errors="coerce"
    )
    data["sample_id"] = (
        data["sample_id"].fillna("").astype(str).str.strip()
    )
    data["raw_value"] = pd.to_numeric(data["raw_value"], errors="coerce")
    data = data.dropna(subset=["plate_column", "raw_value"])
    data["plate_column"] = data["plate_column"].astype(int)

    if data.empty:
        raise ValueError("Brak liczbowych wyników RFU do analizy RiboGreen.")
    return data


def classify_ribogreen_wells(combined_data: pd.DataFrame) -> pd.DataFrame:
    """Przypisuje dołki do krzywych TE1/TX3 lub próbek Free/Total."""
    data = _validate_input(combined_data)
    data["analysis_role"] = "unused"
    data["buffer_type"] = ""

    curve_rows = data["plate_row"].isin(["A", "B"])
    sample_rows = data["plate_row"].isin(list("CDEFGH"))

    te1_curve = curve_rows & data["plate_column"].between(1, 6)
    tx3_curve = curve_rows & data["plate_column"].between(7, 12)
    free_samples = sample_rows & (
        data["plate_column"].between(1, 3)
        | data["plate_column"].between(7, 9)
    )
    total_samples = sample_rows & (
        data["plate_column"].between(4, 6)
        | data["plate_column"].between(10, 12)
    )

    data.loc[te1_curve, ["analysis_role", "buffer_type"]] = [
        "calibration",
        "TE1",
    ]
    data.loc[tx3_curve, ["analysis_role", "buffer_type"]] = [
        "calibration",
        "TX3",
    ]
    data.loc[free_samples, ["analysis_role", "buffer_type"]] = [
        "sample_free",
        "TE1",
    ]
    data.loc[total_samples, ["analysis_role", "buffer_type"]] = [
        "sample_total",
        "TX3",
    ]

    return data


def _fit_linear_calibration(
    classified_data: pd.DataFrame,
    buffer_type: str,
) -> dict:
    standards_map = (
        TE1_STANDARD_CONCENTRATIONS
        if buffer_type == "TE1"
        else TX3_STANDARD_CONCENTRATIONS
    )
    blank_id = "Blank1" if buffer_type == "TE1" else "Blank2"

    curve_data = classified_data.loc[
        classified_data["analysis_role"].eq("calibration")
        & classified_data["buffer_type"].eq(buffer_type)
    ].copy()

    available = set(curve_data["sample_id"].tolist())
    missing = set(standards_map) - available
    if missing:
        raise ValueError(
            f"Krzywa {buffer_type}: brakuje identyfikatorów: "
            + ", ".join(sorted(missing))
            + "."
        )

    summary = (
        curve_data.groupby("sample_id", as_index=False)
        .agg(
            n=("raw_value", "count"),
            mean_rfu=("raw_value", "mean"),
            sd_rfu=("raw_value", "std"),
            minimum_rfu=("raw_value", "min"),
            maximum_rfu=("raw_value", "max"),
        )
    )
    summary["cv_pct"] = summary["sd_rfu"] / summary["mean_rfu"].abs() * 100
    summary["nominal_concentration_ug_ml"] = summary["sample_id"].map(
        standards_map
    )

    blank_rows = summary.loc[summary["sample_id"].eq(blank_id)]
    if blank_rows.empty:
        raise ValueError(f"Krzywa {buffer_type}: nie znaleziono {blank_id}.")
    blank_mean = float(blank_rows["mean_rfu"].iloc[0])
    summary["corrected_rfu"] = summary["mean_rfu"] - blank_mean
    summary.loc[summary["sample_id"].eq(blank_id), "corrected_rfu"] = 0.0
    summary = summary.sort_values("nominal_concentration_ug_ml").reset_index(
        drop=True
    )

    x = summary["nominal_concentration_ug_ml"].to_numpy(dtype=float)
    y = summary["corrected_rfu"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    if slope <= 0:
        raise ValueError(f"Krzywa {buffer_type} ma niedodatnie nachylenie.")

    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

    summary["calculated_concentration_ug_ml"] = (
        summary["corrected_rfu"] - intercept
    ) / slope
    summary["agreement_pct"] = np.where(
        summary["nominal_concentration_ug_ml"] > 0,
        summary["calculated_concentration_ug_ml"]
        / summary["nominal_concentration_ug_ml"]
        * 100,
        np.nan,
    )

    curve_x = np.linspace(0, float(x.max()), 300)
    curve_y = slope * curve_x + intercept

    return {
        "buffer_type": buffer_type,
        "summary": summary,
        "blank_mean": blank_mean,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "curve_x": curve_x,
        "curve_y": curve_y,
        "minimum_corrected_rfu": float(summary["corrected_rfu"].min()),
        "maximum_corrected_rfu": float(summary["corrected_rfu"].max()),
    }


def create_ribogreen_calibration_figure(calibration: dict):
    summary = calibration["summary"]
    buffer_type = calibration["buffer_type"]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=summary["nominal_concentration_ug_ml"],
            y=summary["corrected_rfu"],
            error_y={"type": "data", "array": summary["sd_rfu"], "visible": True},
            mode="markers+text",
            text=summary["sample_id"],
            textposition="top center",
            name=f"Standardy {buffer_type}",
            marker={"size": 10},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=calibration["curve_x"],
            y=calibration["curve_y"],
            mode="lines",
            name="Regresja liniowa",
            line={"width": 3},
        )
    )
    equation = (
        f"y = {calibration['slope']:.6g}x "
        f"{calibration['intercept']:+.6g}<br>"
        f"R² = {calibration['r_squared']:.4f}"
    )
    figure.add_annotation(
        x=0.04 * float(summary["nominal_concentration_ug_ml"].max()),
        y=0.90 * float(np.nanmax(calibration["curve_y"])),
        text=equation,
        showarrow=False,
        align="left",
        bgcolor="white",
        bordercolor="gray",
    )
    figure.update_layout(
        title=f"Krzywa kalibracyjna RiboGreen {buffer_type}",
        xaxis_title="Stężenie RNA [µg/mL]",
        yaxis_title="RFU po korekcji blanku",
        legend={"orientation": "h", "y": 1.02, "x": 0},
        margin={"l": 70, "r": 40, "t": 105, "b": 70},
    )
    return figure


def _summarize_sample_signals(
    classified_data: pd.DataFrame,
    calibration: dict,
    role: str,
    measurement_name: str,
) -> pd.DataFrame:
    sample_data = classified_data.loc[
        classified_data["analysis_role"].eq(role)
        & classified_data["sample_id"].ne("")
    ].copy()
    if sample_data.empty:
        raise ValueError(f"Nie znaleziono wyników próbek {measurement_name}.")

    summary = (
        sample_data.groupby("sample_id", as_index=False)
        .agg(
            technical_n=("raw_value", "count"),
            mean_rfu=("raw_value", "mean"),
            sd_rfu=("raw_value", "std"),
            minimum_rfu=("raw_value", "min"),
            maximum_rfu=("raw_value", "max"),
        )
    )
    summary["cv_pct"] = summary["sd_rfu"] / summary["mean_rfu"].abs() * 100
    summary["corrected_rfu"] = summary["mean_rfu"] - calibration["blank_mean"]
    summary["concentration_from_curve_ug_ml"] = (
        summary["corrected_rfu"] - calibration["intercept"]
    ) / calibration["slope"]
    summary["range_status"] = "IN_RANGE"
    summary.loc[
        summary["corrected_rfu"] < calibration["minimum_corrected_rfu"],
        "range_status",
    ] = "BELOW_RANGE"
    summary.loc[
        summary["corrected_rfu"] > calibration["maximum_corrected_rfu"],
        "range_status",
    ] = "ABOVE_RANGE"
    summary["measurement_type"] = measurement_name
    return summary


def create_default_ribogreen_config(sample_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "final_volume_v1_ul": 150.0,
            "sample_volume_ul": 1.0,
            "plate_final_volume_v2_ul": 200.0,
            "plate_sample_volume_v3_ul": 50.0,
            "theoretical_rna_ug_ml": np.nan,
        }
    )


def calculate_ribogreen_results(
    free_summary: pd.DataFrame,
    total_summary: pd.DataFrame,
    sample_config: pd.DataFrame,
) -> pd.DataFrame:
    results = free_summary.merge(
        total_summary,
        on="sample_id",
        how="outer",
        suffixes=("_free", "_total"),
        validate="one_to_one",
    )
    results = results.merge(sample_config, on="sample_id", how="left")

    required_numeric = [
        "final_volume_v1_ul",
        "sample_volume_ul",
        "plate_final_volume_v2_ul",
        "plate_sample_volume_v3_ul",
        "theoretical_rna_ug_ml",
    ]
    for column in required_numeric:
        results[column] = pd.to_numeric(results[column], errors="coerce")

    invalid_denominator = (
        results["sample_volume_ul"].fillna(0).le(0)
        | results["plate_sample_volume_v3_ul"].fillna(0).le(0)
    )
    if invalid_denominator.any():
        invalid_samples = results.loc[invalid_denominator, "sample_id"].tolist()
        raise ValueError(
            "Objętości używane w mianowniku DF muszą być większe od zera: "
            + ", ".join(invalid_samples)
            + "."
        )

    results["dilution_factor"] = (
        results["final_volume_v1_ul"] / results["sample_volume_ul"]
        * results["plate_final_volume_v2_ul"]
        / results["plate_sample_volume_v3_ul"]
    )
    results["free_rna_ug_ml"] = (
        results["concentration_from_curve_ug_ml_free"]
        * results["dilution_factor"]
    )
    results["total_rna_ug_ml"] = (
        results["concentration_from_curve_ug_ml_total"]
        * results["dilution_factor"]
    )
    results["rna_in_lnp_ug_ml"] = (
        results["total_rna_ug_ml"] - results["free_rna_ug_ml"]
    )
    results["ee_pct"] = np.where(
        results["total_rna_ug_ml"] > 0,
        results["rna_in_lnp_ug_ml"] / results["total_rna_ug_ml"] * 100,
        np.nan,
    )
    results["pe_pct"] = np.where(
        results["theoretical_rna_ug_ml"] > 0,
        results["total_rna_ug_ml"] / results["theoretical_rna_ug_ml"] * 100,
        np.nan,
    )
    results["logical_status"] = "OK"
    results.loc[
        results["free_rna_ug_ml"] > results["total_rna_ug_ml"],
        "logical_status",
    ] = "CHECK_FREE_GT_TOTAL"
    results.loc[
        results["ee_pct"].notna()
        & ((results["ee_pct"] < 0) | (results["ee_pct"] > 100)),
        "logical_status",
    ] = "CHECK_EE"
    results.loc[
        results["technical_n_free"].fillna(0).lt(3)
        | results["technical_n_total"].fillna(0).lt(3),
        "logical_status",
    ] = "CHECK_N"
    return results.sort_values("sample_id").reset_index(drop=True)


def create_rna_concentration_figure(results: pd.DataFrame):
    long_data = results.melt(
        id_vars=["sample_id"],
        value_vars=["free_rna_ug_ml", "total_rna_ug_ml", "rna_in_lnp_ug_ml"],
        var_name="rna_type",
        value_name="concentration_ug_ml",
    )
    labels = {
        "free_rna_ug_ml": "Free RNA",
        "total_rna_ug_ml": "Total RNA",
        "rna_in_lnp_ug_ml": "RNA in LNP",
    }
    long_data["rna_type"] = long_data["rna_type"].map(labels)
    figure = go.Figure()
    for rna_type in ["Free RNA", "Total RNA", "RNA in LNP"]:
        subset = long_data.loc[long_data["rna_type"].eq(rna_type)]
        figure.add_trace(
            go.Bar(
                x=subset["sample_id"],
                y=subset["concentration_ug_ml"],
                name=rna_type,
            )
        )
    figure.update_layout(
        title="Stężenie RNA w próbkach LNP",
        xaxis_title="Próbka",
        yaxis_title="Stężenie RNA [µg/mL]",
        barmode="group",
        xaxis_tickangle=-45,
    )
    return figure


def create_efficiency_figure(results: pd.DataFrame):
    figure = go.Figure()
    figure.add_trace(go.Bar(x=results["sample_id"], y=results["ee_pct"], name="EE [%]"))
    figure.add_trace(go.Bar(x=results["sample_id"], y=results["pe_pct"], name="PE [%]"))
    figure.update_layout(
        title="Encapsulation efficiency i process efficiency",
        xaxis_title="Próbka",
        yaxis_title="Wartość [%]",
        barmode="group",
        xaxis_tickangle=-45,
    )
    return figure


def show_ribogreen_module(combined_data: pd.DataFrame) -> None:
    st.subheader("Analiza RiboGreen: Free RNA, Total RNA, EE i PE")
    st.write(
        "Krzywa TE1 służy do oznaczenia Free RNA, a krzywa TX3 do "
        "oznaczenia Total RNA. Typ pomiaru próbki jest określany na "
        "podstawie położenia dołka."
    )

    enable_exclusions = st.checkbox(
        "Włącz ręczne wykluczanie pojedynczych dołków",
        value=False,
        key="ribogreen_enable_exclusions",
    )
    analysis_data = combined_data.copy()
    excluded_results = pd.DataFrame()
    if enable_exclusions:
        analysis_data, excluded_results = show_exclusion_editor(
            combined_data,
            key_prefix="ribogreen",
        )

    classified = classify_ribogreen_wells(analysis_data)
    te1_calibration = _fit_linear_calibration(classified, "TE1")
    tx3_calibration = _fit_linear_calibration(classified, "TX3")

    st.plotly_chart(
        create_ribogreen_calibration_figure(te1_calibration),
        width="stretch",
    )
    st.dataframe(
        te1_calibration["summary"].style.format(
            {
                "mean_rfu": "{:.4g}",
                "sd_rfu": "{:.4g}",
                "cv_pct": "{:.2f}",
                "nominal_concentration_ug_ml": "{:.4f}",
                "corrected_rfu": "{:.4g}",
                "calculated_concentration_ug_ml": "{:.4f}",
                "agreement_pct": "{:.2f}",
            },
            na_rep="brak",
        ),
        width="stretch",
        hide_index=True,
    )

    st.plotly_chart(
        create_ribogreen_calibration_figure(tx3_calibration),
        width="stretch",
    )
    st.dataframe(
        tx3_calibration["summary"].style.format(
            {
                "mean_rfu": "{:.4g}",
                "sd_rfu": "{:.4g}",
                "cv_pct": "{:.2f}",
                "nominal_concentration_ug_ml": "{:.4f}",
                "corrected_rfu": "{:.4g}",
                "calculated_concentration_ug_ml": "{:.4f}",
                "agreement_pct": "{:.2f}",
            },
            na_rep="brak",
        ),
        width="stretch",
        hide_index=True,
    )

    free_summary = _summarize_sample_signals(
        classified,
        te1_calibration,
        "sample_free",
        "Free RNA",
    )
    total_summary = _summarize_sample_signals(
        classified,
        tx3_calibration,
        "sample_total",
        "Total RNA",
    )
    sample_ids = sorted(set(free_summary["sample_id"]) | set(total_summary["sample_id"]))

    st.subheader("Parametry rozcieńczenia i stężenia teoretyczne")
    st.caption(
        "DF = (V1 / objętość pobranej próbki) × (V2 / V3). "
        "V2 i V3 mają domyślne wartości 200 µL i 50 µL."
    )
    sample_config = st.data_editor(
        create_default_ribogreen_config(sample_ids),
        width="stretch",
        hide_index=True,
        disabled=["sample_id"],
        column_config={
            "sample_id": st.column_config.TextColumn("Próbka"),
            "final_volume_v1_ul": st.column_config.NumberColumn("V1 [µL]", min_value=0.0, format="%.2f"),
            "sample_volume_ul": st.column_config.NumberColumn("Objętość pobranej próbki [µL]", min_value=0.0, format="%.4f"),
            "plate_final_volume_v2_ul": st.column_config.NumberColumn("V2 [µL]", min_value=0.0, format="%.2f"),
            "plate_sample_volume_v3_ul": st.column_config.NumberColumn("V3 [µL]", min_value=0.0, format="%.2f"),
            "theoretical_rna_ug_ml": st.column_config.NumberColumn("Teoretyczne RNA [µg/mL]", min_value=0.0, format="%.4f"),
        },
        key="ribogreen_sample_config",
    )

    results = calculate_ribogreen_results(
        free_summary,
        total_summary,
        sample_config,
    )

    st.subheader("Wyniki próbek")
    st.dataframe(
        results.style.format(
            {
                "mean_rfu_free": "{:.4g}",
                "sd_rfu_free": "{:.4g}",
                "cv_pct_free": "{:.2f}",
                "mean_rfu_total": "{:.4g}",
                "sd_rfu_total": "{:.4g}",
                "cv_pct_total": "{:.2f}",
                "dilution_factor": "{:.2f}",
                "free_rna_ug_ml": "{:.4f}",
                "total_rna_ug_ml": "{:.4f}",
                "rna_in_lnp_ug_ml": "{:.4f}",
                "theoretical_rna_ug_ml": "{:.4f}",
                "ee_pct": "{:.2f}",
                "pe_pct": "{:.2f}",
            },
            na_rep="brak",
        ),
        width="stretch",
        hide_index=True,
    )

    st.plotly_chart(create_rna_concentration_figure(results), width="stretch")
    st.plotly_chart(create_efficiency_figure(results), width="stretch")

    if not excluded_results.empty:
        st.subheader("Wyniki wykluczone z obliczeń")
        st.dataframe(excluded_results, width="stretch", hide_index=True)

    missing_theoretical = results.loc[
        results["theoretical_rna_ug_ml"].isna()
        | results["theoretical_rna_ug_ml"].le(0),
        "sample_id",
    ].tolist()
    if missing_theoretical:
        st.info(
            "Nie obliczono PE dla próbek bez dodatniego stężenia teoretycznego: "
            + ", ".join(missing_theoretical)
            + "."
        )

    logical_warnings = results.loc[
        ~results["logical_status"].eq("OK"),
        ["sample_id", "logical_status"],
    ]
    if not logical_warnings.empty:
        st.warning("Część wyników wymaga sprawdzenia.")
        st.dataframe(logical_warnings, width="stretch", hide_index=True)
