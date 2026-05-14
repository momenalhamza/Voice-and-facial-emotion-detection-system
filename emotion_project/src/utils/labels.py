"""Single source of truth for emotion labels used across the whole project."""
from __future__ import annotations

EMOTIONS: list[str] = ["Happy", "Sad", "Angry", "Fearful", "Neutral"]
NUM_CLASSES: int = len(EMOTIONS)

LABEL_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(EMOTIONS)}
INDEX_TO_LABEL: dict[int, str] = {i: name for i, name in enumerate(EMOTIONS)}

# Common aliases from public datasets → canonical label.
# (RAVDESS / CREMA-D / FER2013 use varied names; we normalize here.)
ALIASES: dict[str, str | None] = {
    # RAVDESS numeric codes (filename position 3 in RAVDESS scheme)
    "01": "Neutral",
    "02": "Neutral",      # "calm" → Neutral
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": None,           # "disgust" — not in our 5-class set, drop
    "08": None,           # "surprised" — not in our 5-class set, drop
    # CREMA-D 3-letter codes
    "ANG": "Angry",
    "DIS": None,          # "disgust" — drop
    "FEA": "Fearful",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
    # FER2013 text labels
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "fear": "Fearful",
    "fearful": "Fearful",
    "neutral": "Neutral",
    "disgust": None,      # drop
    "surprise": None,     # drop
    "surprised": None,    # drop
    "calm": "Neutral",
}

# Emotions present in source datasets that we do NOT keep.
DROPPED = {"disgust", "surprise", "surprised", "07", "08", "DIS"}


def canonicalize(raw: str) -> str | None:
    """Map any raw label string to one of the 5 canonical emotions, or None to drop."""
    if raw in EMOTIONS:
        return raw
    if raw in ALIASES:
        return ALIASES[raw]
    lowered = raw.lower()
    if lowered in ALIASES:
        return ALIASES[lowered]
    return None
