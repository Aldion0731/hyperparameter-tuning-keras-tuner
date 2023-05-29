from dataclasses import dataclass


@dataclass
class FixedHyperparams:
    epochs: int
    validation_split: float
