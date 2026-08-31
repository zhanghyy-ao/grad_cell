import numpy as np
import pandas as pd

from gradcell.data import MOLE_FRACTION_COLUMNS, SOLVENT_COLUMNS, clean_calisol23_frame


def make_frame() -> pd.DataFrame:
    rows = []
    for index, values in enumerate(
        (
            {"k": 5.0, "c": 1.0, "salt": " LiBPFPB", "ratio": "w"},
            {"k": 0.0, "c": 1.0, "salt": "LiPF6", "ratio": "v"},
            {"k": np.nan, "c": 1.0, "salt": "LiPF6", "ratio": "mol"},
            {"k": 2.0, "c": -0.01, "salt": "LiN(CF3SO2)2", "ratio": "mol"},
        )
    ):
        row = {
            "Unnamed: 0": index,
            "doi": " 10.1/ABC ",
            "k": values["k"],
            "T": 298.15,
            "c": values["c"],
            "salt": values["salt"],
            "c units": "mol/l",
            "solvent ratio type": values["ratio"],
            **{name: 0.0 for name in SOLVENT_COLUMNS},
        }
        row["EC"] = 0.5
        row["DMC"] = 0.5
        rows.append(row)
    return pd.DataFrame(rows)


def test_cleaner_canonicalizes_and_filters_model_rows():
    result = clean_calisol23_frame(make_frame())
    canonical = result.canonical
    assert canonical["doi"].unique().tolist() == ["10.1/abc"]
    assert canonical.loc[0, "salt_canonical"] == "LiBPFPB"
    assert canonical.loc[3, "salt_canonical"] == "LiTFSI"
    assert canonical["c units"].unique().tolist() == ["mol/L"]
    assert result.report["missing_target"] == 1
    assert result.report["conductivity_censored"] == 1
    assert result.report["invalid_concentration"] == 1
    assert len(result.model_v1) == 1


def test_cleaner_converts_all_ratio_bases_to_mole_fractions():
    result = clean_calisol23_frame(make_frame())
    fractions = result.canonical.loc[:, MOLE_FRACTION_COLUMNS].to_numpy()
    assert np.allclose(fractions.sum(axis=1), 1.0)
    assert not np.allclose(fractions[0], fractions[1])
    assert np.allclose(fractions[2], np.array([0.5, 0.0, 0.5, *([0.0] * 35)]))
