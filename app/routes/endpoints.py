from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.recommender import Recommender
from app.models.advisor import Advisor
from app.models.analyzer import PropertyAnalyzer
from app.models.condition_classifier import ConditionClassifier
from app.models.preprocessor import Preprocessor
import pandas as pd
import json

router = APIRouter(prefix="/api")

# Load data and initialize modules
df = pd.read_csv("listings.csv")
prep = Preprocessor()
df_p = prep.transform(df)
rec = Recommender().fit(df_p)
adv = Advisor(rec)
analyzer = PropertyAnalyzer()

# ---- SCHEMAS ----
class AnalysisInput(BaseModel):
    id: str
    title: str
    description: str
    images: list[str] = []

class AdviceInput(BaseModel):
    budget: float | None = None
    city: str | None = None
    top_n: int = 5

class RecommenderInput(BaseModel):
    reference_id: str
    top_n: int = 5

# ---- ROUTES ----
@router.post("/analysis")
def analyze_property(input: AnalysisInput):
    try:
        result = analyzer.analyze(input.images, input.description)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/advice")
def get_advice(input: AdviceInput):
    try:
        result = adv.advise(budget=input.budget, city=input.city, top_n=input.top_n)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommendations")
def get_recommendations(input: RecommenderInput):
    try:
        data = rec.build_json_recommendation_table(top_n=input.top_n)
        filtered = [r for r in data if r["reference_id"] == input.reference_id]
        return filtered[0] if filtered else {"message": "No match found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
