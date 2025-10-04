from pydantic import BaseModel
from typing import List, Optional, Dict

class AnalysisRequest(BaseModel):
    id: str
    title: str
    description: str
    images: Optional[List[str]] = []

class RecommendRequest(BaseModel):
    reference_id: str
    top_n: int = 5
    style: Optional[str] = "nested"  # nested | flat | wide

class AdviceRequest(BaseModel):
    budget: Optional[float] = None
    budget_range: Optional[List[float]] = None
    city: Optional[str] = None
    reference_id: Optional[str] = None
    title: Optional[str] = None
    top_n: int = 5
    include_condition: bool = False
