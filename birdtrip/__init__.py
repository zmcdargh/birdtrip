"""
birdtrip — plan birding trips from eBird data.

Pipeline:
  precompute   raw EBD + Sampling Event Data  ->  per-(species, place, week) table
               of frequency (shrunk), inter-year occupancy, and p_lifer.
  taxonomy     name <-> species_code, and resolve any taxon to its countable species.
  lifelist     parse an eBird "Download My Data" export into a species-code life list.
  recommend    rank destinations by rarity-weighted expected lifers (the alpha slider).
  summary      human-readable trip summary: expected lifers + the likely birds.
"""
from .taxonomy import Taxonomy
from .lifelist import parse_life_list, LifeListResult

__all__ = ["Taxonomy", "parse_life_list", "LifeListResult"]
__version__ = "0.1.0"
