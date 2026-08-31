from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

SOLVENT_COLUMNS = (
    "EC",
    "PC",
    "DMC",
    "EMC",
    "DEC",
    "DME",
    "DMSO",
    "AN",
    "MOEMC",
    "TFP",
    "EA",
    "MA",
    "FEC",
    "DOL",
    "2-MeTHF",
    "DMM",
    "Freon 11",
    "Methylene chloride",
    "THF",
    "Toluene",
    "Sulfolane",
    "2-Glyme",
    "3-Glyme",
    "4-Glyme",
    "3-Me-2-Oxazolidinone",
    "3-MeSulfolane",
    "Ethyldiglyme",
    "DMF",
    "Ethylbenzene",
    "Ethylmonoglyme",
    "Benzene",
    "g-Butyrolactone",
    "Cumene",
    "Propylsulfone",
    "Pseudocumeme",
    "TEOS",
    "m-Xylene",
    "o-Xylene",
)
MOLE_FRACTION_COLUMNS = tuple(f"x_{name}" for name in SOLVENT_COLUMNS)

# Values used by the CALiSol-23 authors' public conversion notebook. Molar masses are
# in g/mol and pure-solvent densities are in g/cm3 at 25 C.
MOLAR_MASS_G_MOL = dict(
    zip(
        SOLVENT_COLUMNS,
        (
            88.06,
            102.08,
            90.08,
            104.10,
            118.132,
            90.12,
            78.13,
            41.05,
            134.13,
            344.07,
            88.10,
            74.08,
            106.05,
            74.08,
            86.13,
            162.2,
            137.36,
            84.93,
            72.10,
            92.14,
            120.17,
            134.17,
            178.22,
            222.28,
            101.10,
            134.20,
            134.17,
            73.09,
            106.17,
            76.10,
            78.11,
            86.09,
            120.19,
            150.24,
            120.19,
            208.33,
            106.17,
            106.16,
        ),
        strict=True,
    )
)
DENSITY_G_CM3 = dict(
    zip(
        SOLVENT_COLUMNS,
        (
            1.3210,
            1.205,
            1.07,
            0.902,
            0.975,
            0.86,
            1.1004,
            0.786,
            1.5,
            1.487,
            0.902,
            0.932,
            1.454,
            1.06,
            0.854,
            0.902,
            1.49,
            1.33,
            0.888,
            0.867,
            1.26,
            0.937,
            0.986,
            1.009,
            1.17,
            1.20,
            0.937,
            0.944,
            0.866,
            0.965,
            0.876,
            1.13,
            0.862,
            1.109,
            0.876,
            0.940,
            0.860,
            0.87596,
        ),
        strict=True,
    )
)

SALT_ALIASES = {"LiN(CF3SO2)2": "LiTFSI"}


@dataclass(frozen=True)
class CleaningResult:
    canonical: pd.DataFrame
    model_v1: pd.DataFrame
    report: dict[str, object]


def _normalize_text(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    return unicodedata.normalize("NFKC", str(value)).strip()


def _mole_fractions(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    raw = frame.loc[:, SOLVENT_COLUMNS].to_numpy(dtype=np.float64)
    basis = frame["solvent ratio type"].astype(str).to_numpy()
    molar_mass = np.asarray([MOLAR_MASS_G_MOL[name] for name in SOLVENT_COLUMNS])
    density = np.asarray([DENSITY_G_CM3[name] for name in SOLVENT_COLUMNS])
    amount = raw.copy()
    mass_rows = basis == "w"
    volume_rows = basis == "v"
    molar_rows = basis == "mol"
    amount[mass_rows] = raw[mass_rows] / molar_mass
    amount[volume_rows] = raw[volume_rows] * density / molar_mass
    valid_basis = mass_rows | volume_rows | molar_rows
    row_sum = amount.sum(axis=1)
    valid = (
        valid_basis
        & np.isfinite(amount).all(axis=1)
        & (amount >= 0.0).all(axis=1)
        & np.isfinite(row_sum)
        & (row_sum > 0.0)
    )
    fractions = np.full_like(amount, np.nan)
    fractions[valid] = amount[valid] / row_sum[valid, None]
    return fractions, valid


def clean_calisol23_frame(
    frame: pd.DataFrame,
    *,
    conductivity_floor_ms_cm: float = 1e-12,
    merge_tfsi_aliases: bool = True,
) -> CleaningResult:
    """Canonicalize CALiSol-23 and select rows suitable for v1 log regression."""
    if conductivity_floor_ms_cm < 0.0:
        raise ValueError("conductivity_floor_ms_cm must be non-negative")
    required = {
        "doi",
        "k",
        "T",
        "c",
        "salt",
        "c units",
        "solvent ratio type",
        *SOLVENT_COLUMNS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CALiSol-23 file is missing columns: {sorted(missing)}")

    canonical = frame.copy()
    exported_index = next(
        (name for name in canonical.columns if str(name).startswith("Unnamed:")), None
    )
    if exported_index is None:
        canonical.insert(0, "source_row_id", np.arange(len(canonical), dtype=np.int64))
    else:
        canonical = canonical.rename(columns={exported_index: "source_row_id"})

    canonical["salt_raw"] = canonical["salt"].astype("string")
    for name in ("doi", "salt", "c units", "solvent ratio type"):
        canonical[name] = canonical[name].map(_normalize_text).astype("string")
    canonical["doi"] = canonical["doi"].str.lower()
    canonical["c units"] = canonical["c units"].str.lower().replace({"mol/l": "mol/L"})
    canonical["solvent ratio type"] = canonical["solvent ratio type"].str.lower()
    if merge_tfsi_aliases:
        canonical["salt"] = canonical["salt"].replace(SALT_ALIASES)
    canonical["salt_canonical"] = canonical["salt"]

    for name in ("k", "T", "c", *SOLVENT_COLUMNS):
        canonical[name] = pd.to_numeric(canonical[name], errors="coerce")

    fractions, valid_composition = _mole_fractions(canonical)
    for column, values in zip(MOLE_FRACTION_COLUMNS, fractions.T, strict=True):
        canonical[column] = values

    conductivity = canonical["k"].to_numpy(dtype=np.float64)
    temperature = canonical["T"].to_numpy(dtype=np.float64)
    concentration = canonical["c"].to_numpy(dtype=np.float64)
    finite_target = np.isfinite(conductivity)
    canonical["missing_target"] = ~finite_target
    canonical["invalid_temperature"] = ~np.isfinite(temperature)
    canonical["invalid_concentration"] = ~np.isfinite(concentration) | (concentration < 0.0)
    canonical["invalid_composition"] = ~valid_composition
    canonical["conductivity_censored"] = finite_target & (
        conductivity <= conductivity_floor_ms_cm
    )
    canonical["temperature_typical_230_330_k"] = (
        np.isfinite(temperature) & (temperature >= 230.0) & (temperature <= 330.0)
    )

    valid_model = (
        finite_target
        & (conductivity > conductivity_floor_ms_cm)
        & np.isfinite(temperature)
        & np.isfinite(concentration)
        & (concentration >= 0.0)
        & valid_composition
    )
    canonical["model_v1_eligible"] = valid_model

    condition_columns = [
        "doi",
        "T",
        "c",
        "salt_canonical",
        "c units",
        *MOLE_FRACTION_COLUMNS,
    ]
    repeat_count = canonical.groupby(condition_columns, dropna=False)["source_row_id"].transform(
        "size"
    )
    canonical["condition_repeat_count"] = repeat_count.astype(np.int64)
    canonical["condition_weight"] = 1.0 / canonical["condition_repeat_count"]

    model_v1 = canonical.loc[valid_model].reset_index(drop=True)
    report: dict[str, object] = {
        "rows_raw": len(canonical),
        "rows_model_v1": len(model_v1),
        "dropped_rows": int((~valid_model).sum()),
        "missing_target": int(canonical["missing_target"].sum()),
        "invalid_temperature": int(canonical["invalid_temperature"].sum()),
        "invalid_concentration": int(canonical["invalid_concentration"].sum()),
        "invalid_composition": int(canonical["invalid_composition"].sum()),
        "conductivity_censored": int(canonical["conductivity_censored"].sum()),
        "conductivity_floor_ms_cm": conductivity_floor_ms_cm,
        "tfsi_aliases_merged": merge_tfsi_aliases,
        "salt_categories_raw": int(canonical["salt_raw"].nunique(dropna=True)),
        "salt_categories_canonical": int(canonical["salt_canonical"].nunique(dropna=True)),
        "repeated_condition_rows": int((canonical["condition_repeat_count"] > 1).sum()),
        "mole_fraction_sum_max_error": float(
            np.nanmax(np.abs(fractions.sum(axis=1) - 1.0))
        ),
    }
    return CleaningResult(canonical=canonical, model_v1=model_v1, report=report)
