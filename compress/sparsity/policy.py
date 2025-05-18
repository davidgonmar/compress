from .groupers import AbstractGrouper
from typing import Type
from typing import Dict
import dataclasses


@dataclasses.dataclass
class Metric:
    name: str
    value: float


@dataclasses.dataclass
class PruningPolicy:
    grouper: Type[AbstractGrouper]
    inter_group_metric: Metric
    intra_group_metric: Metric

    def __repr__(self) -> str:
        inter = ", ".join(f"{k}: {v}" for k, v in self.inter_group_metric.items())
        intra = ", ".join(f"{k}: {v}" for k, v in self.intra_group_metric.items())
        return f"PruningPolicy({inter}, {intra})"


PolicyDict = Dict[str, PruningPolicy]
