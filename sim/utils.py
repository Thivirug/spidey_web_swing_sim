import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.objectives import event_state_to_cartesian
from src.events import WebAttachedEvent

