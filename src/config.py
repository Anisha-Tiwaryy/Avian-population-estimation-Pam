"""
Central configuration: target species, their analysis bands, and the
development-set ground truth.

ASSUMPTION (documented in the report): the frequency bands below are literature
approximations for each species' dominant call energy. They are *hyper-parameters
of the DSP detector*, not ground truth. `scripts/02_calibrate_bands.py` re-checks
them against the data and prints suggested adjustments.
"""

from dataclasses import dataclass, field

SAMPLE_RATE = 48_000
CLIP_SECONDS = 3.0


@dataclass(frozen=True)
class Species:
    key: str                    # short internal id
    common_name: str            # must match metadata/ground_truth.csv `common_name`
    scientific_name: str
    band_hz: tuple              # (low, high) dominant call energy
    is_main_target: bool = True
    notes: str = ""
    # Aliases that a pretrained detector (BirdNET / ARIA) may output.
    detector_aliases: tuple = field(default_factory=tuple)


SPECIES = {
    "flamingo": Species(
        key="flamingo",
        common_name="Greater flamingo",
        scientific_name="Phoenicopterus roseus",
        band_hz=(350, 2200),
        notes=(
            "Flock/chorus caller. Honking calls with strong harmonic stack. "
            "Individuals are acoustically inseparable during synchronous calling, "
            "so per-call detection counts saturate with flock size."
        ),
        detector_aliases=("Greater Flamingo", "Phoenicopterus roseus"),
    ),
    "quelea": Species(
        key="quelea",
        common_name="Red-billed quelea",
        scientific_name="Quelea quelea",
        band_hz=(2000, 8000),
        notes=(
            "Dense high-frequency chattering from large flocks. Effectively a "
            "continuous texture rather than discrete events; band occupancy and "
            "noise-floor elevation carry more information than event counts."
        ),
        detector_aliases=("Red-billed Quelea", "Quelea quelea"),
    ),
    "ibis": Species(
        key="ibis",
        common_name="Hadada ibis",
        scientific_name="Bostrychia hagedash",
        band_hz=(700, 3500),
        notes=(
            "Very small populations (4-6). Loud, impulsive, well-separated "
            "'haa-haa-de-dah' bouts. The only target where counting distinct "
            "callers / distinct bouts is plausibly meaningful."
        ),
        detector_aliases=("Hadada Ibis", "Hadeda Ibis", "Bostrychia hagedash"),
    ),
    "avocet": Species(
        key="avocet",
        common_name="Pied avocet",
        scientific_name="Recurvirostra avosetta",
        band_hz=(1500, 4000),
        is_main_target=False,
        notes="Optional target (evaluation set only). Distinct 'kluit' calls.",
        detector_aliases=("Pied Avocet", "Recurvirostra avosetta"),
    ),
}

MAIN_TARGETS = [k for k, s in SPECIES.items() if s.is_main_target]

# Development-set ground truth (from metadata/ground_truth.csv, is_target == 1).
# Hard-coded here so the modelling code can run even on a subsampled download.
# THIS IS THE ENTIRE LABEL SET: 8 (aviary, species) points. See report §2.
DEV_LABELS = [
    # aviary_id,       species_key, count
    ("dev_aviary_1", "quelea", 153),
    ("dev_aviary_2", "flamingo", 107),
    ("dev_aviary_2", "ibis", 6),
    ("dev_aviary_3", "quelea", 61),
    ("dev_aviary_4", "flamingo", 161),
    ("dev_aviary_4", "ibis", 4),
    ("dev_aviary_5", "flamingo", 52),
    ("dev_aviary_6", "flamingo", 52),
]

# dev_aviary_5 and dev_aviary_6 are the SAME physical population recorded on
# different dates. Treating them as two independent points inflates apparent
# performance, so cross-validation groups them together.
CV_GROUPS = {
    "dev_aviary_1": "g1",
    "dev_aviary_2": "g2",
    "dev_aviary_3": "g3",
    "dev_aviary_4": "g4",
    "dev_aviary_5": "g5",
    "dev_aviary_6": "g5",   # same location + population as aviary 5
}

# Which target species must be counted for each evaluation aviary
# (from metadata/eval_recording_info.csv, reproduced for convenience).
EVAL_TARGETS = {
    "eval_aviary_1": "quelea",
    "eval_aviary_2": "quelea",
    "eval_aviary_3": "avocet",
    "eval_aviary_4": "flamingo",
    "eval_aviary_5": "avocet",
    "eval_aviary_6": "ibis",
    "eval_aviary_7": "ibis",
    "eval_aviary_8": "flamingo",
    "eval_aviary_9": "avocet",
    "eval_aviary_10": "avocet",
}

HF_REPO_ID = "Emreargin/BioDCASE2026_Bird_Counting"
