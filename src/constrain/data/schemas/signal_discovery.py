# constrain/data/schemas/signal_discovery.py

from typing import Optional

from pydantic import BaseModel


class SignalDiscoveryDTO(BaseModel):
    id: Optional[int] = None
    run_id: Optional[str]
    horizon: int
    auc_score: float
    feature_name: str
    importance: float
    created_at: float
