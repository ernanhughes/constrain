# certum/protocols/policy.py

from typing import Optional, Protocol

from constrain.energy.axes.bundle import AxisBundle
from constrain.energy.custom_types import Verdict


class Policy(Protocol):

    # Required attributes
    tau_accept: float
    tau_review: Optional[float]
    hard_negative_gap: float

    # Required behavior
    def decide(
        self,
        axes: AxisBundle,
        effectiveness_score: float
    ) -> Verdict:
        ...

    @property
    def name(self) -> str:
        ...
