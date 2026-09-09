# dataclasses?

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
@dataclass
class WebAttachedEvent:
    start_time: float
    anchor: tuple[float, float]
    L: float
    end_time: Optional[float] = None
    states: list[tuple[float, np.ndarray]] = field(default_factory=list) # create a new list for each instane
    name: str = field(default="attached", init=False) # not accepted as an arg
@dataclass
class ReleasedEvent:
    start_time: float
    end_time: Optional[float] = None
    states: list[tuple[float, np.ndarray]] = field(default_factory=list)
    name: str = field(default="released", init=False)