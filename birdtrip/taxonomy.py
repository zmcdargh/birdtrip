"""
eBird taxonomy: name <-> species_code lookup, and resolution of any taxon to the
countable species it represents.

Countability (the birder's exact-match rule):
  - 'species'                       -> counts as itself
  - 'issf' / 'form' / 'intergrade'  -> a subspecies/form; rolls UP to its parent
                                       species via REPORT_AS (seeing the Oregon
                                       junco means you have Dark-eyed Junco)
  - 'spuh' / 'slash' / 'hybrid'     -> NOT a definite species; does not count
  - 'domestic'                      -> rolls up via REPORT_AS if it points to a
                                       species, else does not count

So `resolve_to_species` returns a species code or None, and the life list is the
set of non-None resolutions over a user's observations.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

DEFAULT_TAXONOMY = (Path(__file__).resolve().parent.parent
                    / "data" / "taxonomy" / "eBird_taxonomy_v2025-4.csv")


class Taxonomy:
    def __init__(self, path: str | Path = DEFAULT_TAXONOMY):
        df = pd.read_csv(path, dtype=str, na_filter=False)
        df.columns = [c.strip().upper() for c in df.columns]
        self.df = df
        self._cat = dict(zip(df.SPECIES_CODE, df.CATEGORY))
        self._report_as = dict(zip(df.SPECIES_CODE, df.REPORT_AS))
        self._com_name = dict(zip(df.SPECIES_CODE, df.PRIMARY_COM_NAME))
        self._sci_name = dict(zip(df.SPECIES_CODE, df.SCI_NAME))
        # lowercased name -> code (scientific is the more reliable join key)
        self._by_com = {n.strip().lower(): c for n, c in zip(df.PRIMARY_COM_NAME, df.SPECIES_CODE)}
        self._by_sci = {n.strip().lower(): c for n, c in zip(df.SCI_NAME, df.SPECIES_CODE)}

    # --- lookups --------------------------------------------------------------
    def code_for(self, common: str | None = None, sci: str | None = None) -> str | None:
        """Find a taxon's species_code from its scientific (preferred) or common name."""
        if sci:
            c = self._by_sci.get(sci.strip().lower())
            if c:
                return c
        if common:
            c = self._by_com.get(common.strip().lower())
            if c:
                return c
        return None

    def category(self, code: str) -> str | None:
        return self._cat.get(code)

    def common_name(self, code: str) -> str:
        return self._com_name.get(code, code)

    def scientific_name(self, code: str) -> str:
        return self._sci_name.get(code, code)

    # --- the core rule --------------------------------------------------------
    def resolve_to_species(self, code: str | None, _depth: int = 0) -> str | None:
        """Countable species code for a taxon, or None if it is not a definite species."""
        if not code or code not in self._cat or _depth > 5:
            return None
        cat = self._cat[code]
        if cat == "species":
            return code
        target = self._report_as.get(code) or ""
        if target and target in self._cat and target != code:
            return self.resolve_to_species(target, _depth + 1)
        return None  # spuh / slash / hybrid with no parent species
