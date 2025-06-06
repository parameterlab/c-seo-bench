from enum import Enum


class AdoptionMode(Enum):
    NONE = "none"  # baseline, no boosting applied
    UNILATERAL = "unilateral"  # original behavior: only the item at Boost Product Index is boosted
    FULL = "full"  # 100% adoption
