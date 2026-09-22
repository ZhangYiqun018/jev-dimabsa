"""Valence/arousal rubrics and question wording for the DimABSA tasks.

The question wording follows the official starter kit's query templates
(``starter_kit/task2task3/pipeline-based method/DataProcess.py``):

    "what valence given the aspect and the opinion?"
    "what arousal given the aspect and the opinion?"
    "what category given the aspect and the opinion?"
    "what aspects?"
    "what opinion given the aspect?"

The official model regresses valence/arousal directly from a CLS head, so it has
no level descriptions to copy. The nine level descriptions below are therefore
ours; the scale itself (VA 1..9, matching the official data) is not.
"""

from __future__ import annotations

from .fewshot import calibration_clause, focus_prefix

# Nine levels, index 0..8, describing the official VA range 1..9.
# `score_to_va` shifts the returned score by +1.

VALENCE_LEVELS = [
    "Strongly negative: a severe fault, harsh or contemptuous complaint",
    "Clearly negative: the aspect is described as bad or disappointing",
    "Moderately negative: real criticism, but not emphatic",
    "Mildly negative: a small complaint or a slight reservation",
    "Neutral or mixed: no clear polarity, or praise and criticism cancel out",
    "Mildly positive: a small or lukewarm compliment",
    "Moderately positive: the aspect is described as good",
    "Clearly positive: strong approval, the aspect is praised",
    "Strongly positive: enthusiastic praise, superlatives, delight",
]

# Arousal is the activation of the *emotional state*, per the official
# annotation guideline: "Arousal captures the intensity or activation level of
# the emotional state and can range from 1 (very calm/low energy) to 9 (very
# excited/high energy)", with 'calm' ~ 2.5 and 'furious' ~ 8.5.
#
# An earlier version of this rubric anchored on writing style instead --
# "emphatic", "exclamatory", "superlatives", "capitals" -- which is a different
# construct. Plainly worded text can carry strongly activated feeling, and that
# version measured style, so it compressed every prediction toward the low end.
AROUSAL_LEVELS = [
    "Very calm, low energy: the aspect arouses no feeling at all; the writer is indifferent",
    "Calm: the aspect is regarded without emotional charge",
    "Somewhat calm: only the faintest feeling about the aspect",
    "Mildly calm: a low-energy, subdued feeling",
    "Moderate: an ordinary, middle-of-the-road level of feeling",
    "Moderately activated: the feeling runs a little above ordinary",
    "Activated, excited: a clearly energised feeling about the aspect",
    "Strongly activated: high energy, intensely felt",
    "Extremely activated, high energy: furious or thrilled; the strongest feeling",
]

# Verbatim from the official subtask-1 prompt (Appendix D of the dataset paper).
_DEFINITION_VALENCE = "Valence: 1 = most negative, 9 = most positive."
_DEFINITION_AROUSAL = "Arousal: 1 = calm/low intensity, 9 = excited/high intensity."

_FOCUS_VALENCE = (
    "Judge only the sentiment directed at this aspect; ignore sentiment toward "
    "any other aspect in the text."
)
# The valence question already scopes to one aspect; arousal needs the same
# scoping or the model reads the whole sentence's emotional temperature.
_FOCUS_AROUSAL = (
    "Judge only the feeling directed at this aspect; ignore feeling toward any "
    "other aspect in the text. Arousal is how activated that feeling is -- how "
    "calm or how excited -- not how positive or negative it is."
)


NULL = "NULL"


def _aspect_slot(aspect: str) -> str:
    """Render the aspect for the question.

    ``NULL`` is the dataset's sentinel for an implicit aspect that is not named
    in the text -- not a literal word to be looked up. Sending it through as a
    quoted string asks the model about a nonexistent term.
    """
    if aspect.strip().upper() == NULL:
        return "the aspect that is left implicit and never named in the text"
    return f'the aspect "{aspect}"'


def _opinion_slot(opinion: str) -> str:
    if opinion.strip().upper() == NULL:
        return "the opinion that is left implicit and never named in the text"
    return f'the opinion "{opinion}"'


def _asked(verb: str, aspect: str, opinion: str | None, n_examples: int) -> str:
    slot = _aspect_slot(aspect)
    if n_examples > 0:
        # With examples the state is a structured object, so the target sentence
        # has to be named -- otherwise "the text" is ambiguous between the
        # calibration examples and the review under evaluation.
        slot = f"{slot} in `review_to_score`"
    if opinion is None:
        return f"What {verb} given {slot}?"
    return f"What {verb} given {slot} and {_opinion_slot(opinion)}?"


def valence_question(aspect: str, opinion: str | None = None, n_examples: int = 0) -> dict:
    """Official valence query wording, as a Jev Score question.

    `n_examples=0` (the default) reproduces the original zero-shot question
    byte-for-byte, so recorded zero-shot numbers stay comparable.
    """
    return {
        "type": "score",
        "instructions": {
            "question": (
                f"{_asked('valence', aspect, opinion, n_examples)} "
                f"{_DEFINITION_VALENCE}{calibration_clause(n_examples)}"
            ),
            "focus": f"{focus_prefix(n_examples)}{_FOCUS_VALENCE}",
        },
        "criteria": VALENCE_LEVELS,
    }


def arousal_question(aspect: str, opinion: str | None = None, n_examples: int = 0) -> dict:
    """Official arousal query wording, as a Jev Score question."""
    return {
        "type": "score",
        "instructions": {
            "question": (
                f"{_asked('arousal', aspect, opinion, n_examples)} "
                f"{_DEFINITION_AROUSAL}{calibration_clause(n_examples)}"
            ),
            "focus": f"{focus_prefix(n_examples)}{_FOCUS_AROUSAL}",
        },
        "criteria": AROUSAL_LEVELS,
    }


CATEGORY_INVENTORY: dict[str, list[str]] = {
    # Transcribed verbatim from the official runner
    # (run_task2&3_trainer_multilingual.py lines 19-34) plus the hotel list from
    # the task README. Labels are combined as ENTITY#ATTRIBUTE.
    "res": [
        f"{entity}#{attribute}"
        for entity in ["RESTAURANT", "FOOD", "DRINKS", "AMBIENCE", "SERVICE", "LOCATION"]
        for attribute in ["GENERAL", "PRICES", "QUALITY", "STYLE_OPTIONS", "MISCELLANEOUS"]
    ],
    "lap": [
        f"{entity}#{attribute}"
        for entity in [
            "LAPTOP", "DISPLAY", "KEYBOARD", "MOUSE", "MOTHERBOARD", "CPU", "FANS_COOLING",
            "PORTS", "MEMORY", "POWER_SUPPLY", "OPTICAL_DRIVES", "BATTERY", "GRAPHICS",
            "HARD_DISK", "MULTIMEDIA_DEVICES", "HARDWARE", "SOFTWARE", "OS", "WARRANTY",
            "SHIPPING", "SUPPORT", "COMPANY", "OUT_OF_SCOPE",
        ]
        for attribute in [
            "GENERAL", "PRICE", "QUALITY", "DESIGN_FEATURES", "OPERATION_PERFORMANCE",
            "USABILITY", "PORTABILITY", "CONNECTIVITY", "MISCELLANEOUS",
        ]
    ],
    "hot": [
        f"{entity}#{attribute}"
        for entity in [
            "HOTEL", "ROOMS", "FACILITIES", "ROOM_AMENITIES", "SERVICE", "LOCATION",
            "FOOD_DRINKS",
        ]
        for attribute in [
            "GENERAL", "PRICE", "COMFORT", "CLEANLINESS", "QUALITY", "DESIGN_FEATURES",
            "STYLE_OPTIONS", "MISCELLANEOUS",
        ]
    ],
    "fin": [
        f"{entity}#{attribute}"
        for entity in ["MARKET", "COMPANY", "BUSINESS", "PRODUCT"]
        for attribute in ["GENERAL", "SALES", "PROFIT", "AMOUNT", "PRICE", "COST"]
    ],
}


def category_question(aspect: str, opinion: str, domain: str) -> dict:
    """Official category query wording, as a Jev Choice question."""
    inventory = CATEGORY_INVENTORY[domain]
    return {
        "type": "choice",
        "instructions": {
            "question": (
                f"What category given {_aspect_slot(aspect)} "
                f"and {_opinion_slot(opinion)}?"
            ),
            "focus": "Choose from the official aspect-category inventory for this domain.",
        },
        "criteria": {label: None for label in inventory},
    }
