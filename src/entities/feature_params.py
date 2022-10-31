from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureParams:
    transformers: List[str]
    target_col: Optional[str]
