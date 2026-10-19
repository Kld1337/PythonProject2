import pandas as pd
import streamlit as st


def prepare_exclusion_table(
    combined_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Przygotowuje tabelę pojedynczych dołków
    do ręcznego oznaczania wyników wykluczonych.

    Dane surowe nie są usuwane.
    """

    required_columns = {
        "well",
        "sample_id",
        "raw_value",
    }

    missing_columns = (
        required_columns
        - set(combined_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Nie można przygotować tabeli "
            "wykluczeń. Brakuje kolumn: "
            + ", ".join(
                sorted(missing_columns)
            )
            + "."
        )

    exclusion_table = combined_data[
        [
            "well",
            "sample_id",
            "raw_value",
        ]
    ].copy()

    exclusion_table["sample_id"] = (
        exclusion_table["sample_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    exclusion_table["raw_value"] = (
        pd.to_numeric(
            exclusion_table["raw_value"],
            errors="coerce",
        )
    )

    exclusion_table = exclusion_table.loc[
        exclusion_table["sample_id"].ne("")
    ].copy()

    exclusion_table["exclude"] = False
    exclusion_table["exclusion_reason"] = ""

    return exclusion_table.reset_index(
        drop=True
    )


def show_exclusion_editor(
    combined_data: pd.DataFrame,
    key_prefix: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Wyświetla edytor ręcznego wykluczania wyników.

    Zwraca:
    1. dane używane do obliczeń,
    2. tabelę wyników wykluczonych.
    """

    exclusion_table = prepare_exclusion_table(
        combined_data
    )

    st.write(
        "Zaznacz pojedyncze dołki, które mają "
        "zostać pominięte w obliczeniach. "
        "Wyniki pozostaną widoczne w danych "
        "surowych."
    )

    edited_exclusion_table = st.data_editor(
        exclusion_table,
        width="stretch",
        hide_index=True,
        disabled=[
            "well",
            "sample_id",
            "raw_value",
        ],
        column_config={
            "well": st.column_config.TextColumn(
                "Dołek",
            ),
            "sample_id": (
                st.column_config.TextColumn(
                    "Identyfikator próbki",
                )
            ),
            "raw_value": (
                st.column_config.NumberColumn(
                    "Wartość surowa",
                    format="%.4f",
                )
            ),
            "exclude": (
                st.column_config.CheckboxColumn(
                    "Wyklucz",
                    help=(
                        "Zaznacz, aby pominąć wynik "
                        "w obliczeniach."
                    ),
                    default=False,
                )
            ),
            "exclusion_reason": (
                st.column_config.TextColumn(
                    "Powód wykluczenia",
                    help=(
                        "Np. błąd pipetowania, "
                        "pęcherzyk, uszkodzony dołek."
                    ),
                )
            ),
        },
        key=f"{key_prefix}_exclusion_editor",
    )

    edited_exclusion_table["exclude"] = (
        edited_exclusion_table["exclude"]
        .fillna(False)
        .astype(bool)
    )

    edited_exclusion_table[
        "exclusion_reason"
    ] = (
        edited_exclusion_table[
            "exclusion_reason"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    excluded_results = (
        edited_exclusion_table.loc[
            edited_exclusion_table[
                "exclude"
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    excluded_wells = set(
        excluded_results["well"].tolist()
    )

    analysis_data = combined_data.loc[
        ~combined_data["well"].isin(
            excluded_wells
        )
    ].copy()

    analysis_data["is_excluded"] = False

    if excluded_results.empty:
        st.info(
            "Nie wykluczono żadnych wyników."
        )

    else:
        missing_reasons = (
            excluded_results[
                "exclusion_reason"
            ]
            .eq("")
            .sum()
        )

        st.warning(
            f"Wykluczono {len(excluded_results)} "
            "wyników. Wyniki te nie zostaną "
            "uwzględnione w obliczeniach."
        )

        if missing_reasons > 0:
            st.warning(
                f"Dla {missing_reasons} wykluczonych "
                "wyników nie podano powodu."
            )

        st.dataframe(
            excluded_results,
            width="stretch",
            hide_index=True,
        )

    return (
        analysis_data,
        excluded_results,
    )


def add_exclusion_information(
    original_data: pd.DataFrame,
    excluded_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Dodaje do danych surowych informację,
    które dołki zostały wykluczone.

    Funkcja jest przeznaczona m.in. do raportu.
    """

    report_data = original_data.copy()

    report_data["is_excluded"] = False
    report_data["exclusion_reason"] = ""

    if excluded_results.empty:
        return report_data

    exclusion_lookup = (
        excluded_results.set_index("well")[
            [
                "exclude",
                "exclusion_reason",
            ]
        ]
    )

    exclusion_mask = report_data[
        "well"
    ].isin(
        exclusion_lookup.index
    )

    report_data.loc[
        exclusion_mask,
        "is_excluded",
    ] = True

    report_data.loc[
        exclusion_mask,
        "exclusion_reason",
    ] = (
        report_data.loc[
            exclusion_mask,
            "well",
        ]
        .map(
            exclusion_lookup[
                "exclusion_reason"
            ]
        )
        .fillna("")
    )

    return report_data
