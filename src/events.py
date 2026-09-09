# dataclasses?

from dataclasses import dataclass
from typing import Dict, Optional
@dataclass
class WebAttachedEvent:
    """
        Swinging
    """
    name: str = "attached"
    start_time: float
    end_time: float
    metadata: Optional[Dict] = None # storage of state data -> theta omega

@dataclass
class ReleasedEvent:
    """
        Projectile 
    """
    name: str = "released"
    start_time: float
    end_time: float
    metadata: Optional[Dict] = None # storage of state data -> x,y,vx,vy

@dataclass
class LandEvent:
    name: str = "landed"
    start_time: float
    end_time: float
    metadata: Optional[Dict] = None 