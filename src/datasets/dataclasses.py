from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

@dataclass
class Dialogue:
    text: str
    bb: BoundingBox

@dataclass
class Panel:
    # To be added: features, etc.
    dialogues: List[Dialogue]

@dataclass
class Sample:
    context_panels: List[Panel]
    answer_candidates: Optional[List[str]]
    answer_target: Optional[Dialogue] = None