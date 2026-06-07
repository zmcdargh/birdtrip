"""
Tests for taxonomy resolution and life-list parsing.

The fixture is built from REAL taxonomy entries (one of each category) so we never
hard-code fragile name strings: we ask the taxonomy for a species, an issf, a spuh,
a slash, and a hybrid, write them into a mock eBird export, and assert the parser
keeps the right ones.
"""
import csv
import pandas as pd
import pytest

from birdtrip.taxonomy import Taxonomy
from birdtrip.lifelist import parse_life_list


@pytest.fixture(scope="module")
def tax():
    return Taxonomy()


def _first(tax, category, needs_report_as=False):
    df = tax.df
    sub = df[df.CATEGORY == category]
    if needs_report_as:
        sub = sub[sub.REPORT_AS != ""]
    row = sub.iloc[0]
    return row.PRIMARY_COM_NAME, row.SCI_NAME, row.SPECIES_CODE, row.REPORT_AS


def test_resolve_rules(tax):
    # species resolves to itself
    _, _, sp_code, _ = _first(tax, "species")
    assert tax.resolve_to_species(sp_code) == sp_code
    # issf rolls up to its parent species
    _, _, issf_code, parent = _first(tax, "issf", needs_report_as=True)
    assert tax.resolve_to_species(issf_code) == tax.resolve_to_species(parent)
    assert tax.category(tax.resolve_to_species(issf_code)) == "species"
    # spuh / slash / hybrid are not definite species
    for cat in ("spuh", "slash", "hybrid"):
        _, _, code, _ = _first(tax, cat)
        assert tax.resolve_to_species(code) is None


def test_parse_life_list(tax, tmp_path):
    species = _first(tax, "species")
    issf = _first(tax, "issf", needs_report_as=True)
    spuh = _first(tax, "spuh")
    slash = _first(tax, "slash")
    hybrid = _first(tax, "hybrid")

    rows = [species, issf, spuh, slash, hybrid]
    csv_path = tmp_path / "MyEBirdData.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Submission ID", "Common Name", "Scientific Name", "Count", "Date"])
        for i, (com, sci, *_rest) in enumerate(rows):
            w.writerow([f"S{i}", com, sci, "1", "2024-05-01"])

    result = parse_life_list(csv_path, taxonomy=tax)

    expected = {tax.resolve_to_species(species[2]), tax.resolve_to_species(issf[2])}
    assert result.species_codes == expected           # species + issf(rolled up) kept
    assert result.n_species == 2
    assert len(result.dropped) == 3                    # spuh + slash + hybrid dropped
    assert {c for _, c in result.dropped} == {"spuh", "slash", "hybrid"}
    assert len(result.unmatched) == 0


def test_subspecies_collapses_to_one(tax, tmp_path):
    """Two subspecies of the same species count as a single life-list entry."""
    df = tax.df
    issfs = df[(df.CATEGORY == "issf") & (df.REPORT_AS != "")]
    parent = issfs.iloc[0].REPORT_AS
    two = issfs[issfs.REPORT_AS == parent].head(2)
    if len(two) < 2:
        pytest.skip("need two subspecies of one species")
    csv_path = tmp_path / "MyEBirdData.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Common Name", "Scientific Name"])
        for _, r in two.iterrows():
            w.writerow([r.PRIMARY_COM_NAME, r.SCI_NAME])
    result = parse_life_list(csv_path, taxonomy=tax)
    assert result.species_codes == {tax.resolve_to_species(parent)}
    assert result.n_species == 1
