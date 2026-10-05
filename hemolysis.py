import pandas as pd
import plotly.express as px
import streamlit as st


def calculate_hemolysis_results(
    combined_data: pd.DataFrame,
    negative_control_id: str,
    positive_control_id: str,
    blank_sample_id: str | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    float,
    float,
    float | None,
]:
    """
    Oblicza procent hemolizy względem kontroli
    ujemnej i dodatniej.

    Hemoliza [%] =
    (próbka - kontrola ujemna)
    / (kontrola dodatnia - kontrola ujemna)
    * 100

    Jeśli wskazano blank, jest on odejmowany
    od wszystkich surowych wyników przed obliczeniem.
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

    blank_mean = None

    if blank_sample_id is not None:
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

        data["corrected_value"] = (
            data["raw_value"] - blank_mean
        )

    else:
        data["corrected_value"] = (
            data["raw_value"]
        )

    negative_values = data.loc[
        data["sample_id"] == negative_control_id,
        "corrected_value",
    ]

    positive_values = data.loc[
        data["sample_id"] == positive_control_id,
        "corrected_value",
    ]

    if negative_values.empty:
        raise ValueError(
            "Nie znaleziono wyników dla "
            "wybranej kontroli ujemnej."
        )

    if positive_values.empty:
        raise ValueError(
            "Nie znaleziono wyników dla "
            "wybranej kontroli dodatniej."
        )

    negative_mean = float(
        negative_values.mean()
    )

    positive_mean = float(
        positive_values.mean()
    )

    denominator = (
        positive_mean - negative_mean
    )

    if denominator <= 0:
        raise ValueError(
            "Średnia kontroli dodatniej musi być "
            "większa od średniej kontroli ujemnej."
        )

    data["hemolysis_pct"] = (
        (
            data["corrected_value"]
            - negative_mean
        )
        / denominator
        * 100
    )

    data["is_negative_control"] = (
        data["sample_id"]
        == negative_control_id
    )

    data["is_positive_control"] = (
        data["sample_id"]
        == positive_control_id
    )

    data["is_blank"] = False

    if blank_sample_id is not None:
        data["is_blank"] = (
            data["sample_id"]
            == blank_sample_id
        )

    summary = (
        data
        .groupby(
            "sample_id",
            as_index=False,
        )
        .agg(
            n=("raw_value", "count"),
            mean_raw_value=(
                "raw_value",
                "mean",
            ),
            sd_raw_value=(
                "raw_value",
                "std",
            ),
            mean_corrected_value=(
                "corrected_value",
                "mean",
            ),
            sd_corrected_value=(
                "corrected_value",
                "std",
            ),
            mean_hemolysis_pct=(
                "hemolysis_pct",
                "mean",
            ),
            sd_hemolysis_pct=(
                "hemolysis_pct",
                "std",
            ),
            is_negative_control=(
                "is_negative_control",
                "first",
            ),
            is_positive_control=(
                "is_positive_control",
                "first",
            ),
            is_blank=(
                "is_blank",
                "first",
            ),
        )
    )

    summary["raw_cv_pct"] = (
        summary["sd_raw_value"]
        / summary["mean_raw_value"]
        * 100
    )

    summary["hemolysis_cv_pct"] = (
        summary["sd_hemolysis_pct"]
        / summary["mean_hemolysis_pct"].abs()
        * 100
    )

    summary.loc[
        summary["mean_hemolysis_pct"] == 0,
        "hemolysis_cv_pct",
    ] = pd.NA

    return (
        data,
        summary,
        negative_mean,
        positive_mean,
        blank_mean,
    )


def create_hemolysis_chart(
    summary: pd.DataFrame,
    show_controls: bool,
):
    """
    Tworzy wykres średniej hemolizy z SD.
    """

    chart_data = summary.loc[
        ~summary["is_blank"]
    ].copy()

    if not show_controls:
        chart_data = chart_data.loc[
            ~chart_data["is_negative_control"]
            & ~chart_data["is_positive_control"]
        ].copy()

    figure = px.bar(
        chart_data,
        x="sample_id",
        y="mean_hemolysis_pct",
        error_y="sd_hemolysis_pct",
        color="mean_hemolysis_pct",
        color_continuous_scale="Blues",
        labels={
            "sample_id": "Próbka",
            "mean_hemolysis_pct": (
                "Hemoliza [%]"
            ),
        },
        title=(
            "Hemoliza próbek względem "
            "kontroli ujemnej i dodatniej"
        ),
    )

    figure.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Kontrola ujemna = 0%",
    )

    figure.add_hline(
        y=100,
        line_dash="dash",
        line_color="red",
        annotation_text="Kontrola dodatnia = 100%",
    )

    figure.update_layout(
        xaxis_tickangle=-45,
        yaxis_title="Hemoliza [%]",
    )

    return figure


def create_hemolysis_summary(
    summary: pd.DataFrame,
    negative_control_id: str,
    positive_control_id: str,
    cv_limit: float,
) -> list:
    """
    Tworzy opisowe podsumowanie wyników hemolizy.
    """

    sample_results = summary.loc[
        ~summary["is_blank"]
        & ~summary["is_negative_control"]
        & ~summary["is_positive_control"]
    ].copy()

    if sample_results.empty:
        return [
            "Nie znaleziono właściwych próbek "
            "do podsumowania."
        ]

    conclusions = []

    conclusions.append(
        "Analiza obejmowała "
        f"{len(sample_results)} identyfikatorów "
        "właściwych próbek."
    )

    maximum_row = sample_results.loc[
        sample_results[
            "mean_hemolysis_pct"
        ].idxmax()
    ]

    conclusions.append(
        "Najwyższą średnią hemolizę "
        f"({float(maximum_row['mean_hemolysis_pct']):.2f}%) "
        "zaobserwowano dla próbki "
        f"{maximum_row['sample_id']}."
    )

    high_cv_mask = (
        sample_results["raw_cv_pct"]
        .gt(cv_limit)
        .fillna(False)
    )

    high_cv_samples = (
        sample_results.loc[
            high_cv_mask,
            "sample_id",
        ]
        .astype(str)
        .tolist()
    )

    if high_cv_samples:
        conclusions.append(
            "Limit CV = "
            f"{cv_limit:.1f}% przekroczyły próbki: "
            + ", ".join(high_cv_samples)
            + "."
        )
    else:
        conclusions.append(
            "Żadna właściwa próbka nie przekroczyła "
            f"limitu CV = {cv_limit:.1f}%."
        )

    conclusions.append(
        "Kontrola ujemna "
        f"'{negative_control_id}' została ustawiona "
        "jako 0% hemolizy, a kontrola dodatnia "
        f"'{positive_control_id}' jako 100%."
    )

    return conclusions


def show_hemolysis_sample_module(
    combined_data: pd.DataFrame,
) -> None:
    """
    Wyświetla analizę właściwych próbek hemolizy.
    """

    st.subheader("Hemoliza: właściwe próbki")

    st.write(
        "Wybierz kontrolę ujemną i dodatnią. "
        "Blank jest opcjonalny i służy do korekcji "
        "surowej absorbancji."
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

    if len(available_sample_ids) < 2:
        st.warning(
            "Do analizy potrzebne są co najmniej "
            "kontrola ujemna i kontrola dodatnia."
        )
        return

    negative_control_id = st.selectbox(
        "Wybierz Negative Control",
        options=available_sample_ids,
        key="hemolysis_negative_control",
    )

    positive_options = [
        sample_id
        for sample_id in available_sample_ids
        if sample_id != negative_control_id
    ]

    positive_control_id = st.selectbox(
        "Wybierz Positive Control",
        options=positive_options,
        key="hemolysis_positive_control",
    )

    blank_options = [
        "Brak korekcji blanku",
        *available_sample_ids,
    ]

    selected_blank = st.selectbox(
        "Wybierz blank",
        options=blank_options,
        key="hemolysis_blank",
    )

    if selected_blank == "Brak korekcji blanku":
        blank_sample_id = None
    else:
        blank_sample_id = selected_blank

    cv_limit = st.number_input(
        "Limit CV [%]",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=1.0,
        key="hemolysis_cv_limit",
    )

    show_controls = st.checkbox(
        "Pokaż kontrole na wykresie",
        value=True,
        key="hemolysis_show_controls",
    )

    try:
        (
            result_data,
            summary,
            negative_mean,
            positive_mean,
            blank_mean,
        ) = calculate_hemolysis_results(
            combined_data,
            negative_control_id,
            positive_control_id,
            blank_sample_id,
        )

        (
            metric_1,
            metric_2,
            metric_3,
        ) = st.columns(3)

        metric_1.metric(
            "Kontrola ujemna",
            f"{negative_mean:.4g}",
        )

        metric_2.metric(
            "Kontrola dodatnia",
            f"{positive_mean:.4g}",
        )

        metric_3.metric(
            "Zakres kontroli",
            f"{positive_mean - negative_mean:.4g}",
        )

        if blank_mean is not None:
            st.info(
                "Średni sygnał blanku: "
                f"{blank_mean:.4g}"
            )

        st.subheader("Tabela wyników hemolizy")

        st.dataframe(
            summary.style.format(
                {
                    "mean_raw_value": "{:.4g}",
                    "sd_raw_value": "{:.4g}",
                    "mean_corrected_value": "{:.4g}",
                    "sd_corrected_value": "{:.4g}",
                    "mean_hemolysis_pct": "{:.2f}",
                    "sd_hemolysis_pct": "{:.2f}",
                    "raw_cv_pct": "{:.2f}",
                    "hemolysis_cv_pct": "{:.2f}",
                },
                na_rep="brak",
            ),
            width="stretch",
            hide_index=True,
        )

        figure = create_hemolysis_chart(
            summary,
            show_controls,
        )

        st.plotly_chart(
            figure,
            width="stretch",
        )

        st.subheader(
            "Podsumowanie i krótkie wnioski"
        )

        conclusions = create_hemolysis_summary(
            summary,
            negative_control_id,
            positive_control_id,
            cv_limit,
        )

        for conclusion in conclusions:
            st.markdown(
                f"- {conclusion}"
            )

        st.caption(
            "Wyniki i kryteria akceptacji należy "
            "zweryfikować zgodnie z protokołem "
            "eksperymentu."
        )

        results_csv = summary.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Pobierz wyniki hemolizy jako CSV",
            data=results_csv,
            file_name="hemolysis_results.csv",
            mime="text/csv",
        )

    except ValueError as error:
        st.error(str(error))