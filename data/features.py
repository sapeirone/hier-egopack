from dataclasses import dataclass


@dataclass
class Features:
    """Dataclass to store features information."""
    name: str
    stride: int
    window: int
    fps: float
    size: int
