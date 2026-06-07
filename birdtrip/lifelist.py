"""
Parse a user's eBird data export ("Download My Data" -> MyEBirdData.csv) into a
life list: the set of countable species codes they have already seen.

Apply the exact-match rule via the taxonomy: subspecies/forms roll up to their
species; spuh/slash/hybrid records are dropped (they don't tick a species). The
returned report surfaces what was dropped or couldn't be matched, so the user can
sanity-check before the planner treats a species as "already have it".
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

from .taxonomy import Taxonomy


@dataclass
class LifeListResult:
    species_codes: set[str]
    n_rows: int
    n_species: int
    dropped: list[tuple[str, str]] = field(default_factory=list)    # (name, category)
    unmatched: list[tuple[str, str]] = field(default_factory=list)  # (common, sci)

    def summary(self) -> str:
        return (f"{self.n_species} species on life list "
                f"(from {self.n_rows} observation rows; "
                f"{len(self.dropped)} non-species taxa dropped, "
                f"{len(self.unmatched)} unmatched).")


def _col(df: pd.DataFrame, name: str) -> str | None:
    for c in df.columns:
        if c.strip().lower() == name:
            return c
    return None


def parse_life_list(csv_path: str | Path, taxonomy: Taxonomy | None = None) -> LifeListResult:
    tax = taxonomy or Taxonomy()
    df = pd.read_csv(csv_path, dtype=str, na_filter=False)
    com_col = _col(df, "common name")
    sci_col = _col(df, "scientific name")
    if com_col is None and sci_col is None:
        raise ValueError("CSV has neither a 'Common Name' nor 'Scientific Name' column "
                         "- is this an eBird 'Download My Data' export?")

    codes: set[str] = set()
    dropped: list[tuple[str, str]] = []
    unmatched: list[tuple[str, str]] = []
    for common, sci in zip(df[com_col] if com_col else [""] * len(df),
                           df[sci_col] if sci_col else [""] * len(df)):
        code = tax.code_for(common=common, sci=sci)
        if code is None:
            unmatched.append((common, sci))
            continue
        species = tax.resolve_to_species(code)
        if species is None:
            dropped.append((common or sci, tax.category(code) or "?"))
            continue
        codes.add(species)

    return LifeListResult(species_codes=codes, n_rows=len(df), n_species=len(codes),
                          dropped=dropped, unmatched=unmatched)
