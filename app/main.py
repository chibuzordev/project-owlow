from fastapi import FastAPI
from app.schemas import AnalysisRequest, RecommendRequest, AdviceRequest
import pandas as pd, json

# services
from app.models.datafetcher import DataFetcher
from app.models.preprocessor import Preprocessor
from app.models.condition_classifier import ConditionClassifier
from app.models.recommender import Recommender
from app.models.advisor import Advisor
from app.models.analyzer import PropertyAnalyzer

app = FastAPI(title="Owlow API")
url = "https://fvwtwkcrmm.us-east-1.awsapprunner.com/api/listings"

fetcher = DataFetcher(url)
prep = Preprocessor()
cond = ConditionClassifier()
rec = Recommender()
advisor = Advisor()
analyzer = PropertyAnalyzer()

listings_df = fetcher.fetch_dataframe()
df_prepped = prep.transform(listings_df)

@app.on_event("startup")
def startup():
    global df_prepped, advisor
    # load local csv (or swap to a DB fetch)
    try:
        df = pd.read_csv("listings.csv")
    except Exception:
        df = pd.DataFrame([])  # fallback
    if not df.empty:
        df_p = prep.transform(df)
        # rule-based classification
        df_p["pred_condition"] = df_p["description"].apply(lambda t: cond.classify_overall(t) if isinstance(t,str) else "Nieznany")
        elems = df_p["description"].apply(lambda t: cond.classify_elements(t) if isinstance(t,str) else {})
        elems_df = pd.DataFrame(elems.tolist()).fillna("Brak danych")
        df_p = pd.concat([df_p.reset_index(drop=True), elems_df.reset_index(drop=True)], axis=1)
        rec.fit(df_p)
        advisor = Advisor(rec, df_history=None)
        df_prepped = df_p

@app.post("/api/analysis")
def analyze(req: AnalysisRequest):
    out = analyzer.analyze(req.images, req.description, use_llm=False)
    # also apply rule result if analyzer didn't return
    if "pred_condition" not in out or out["pred_condition"]=="Nieznany":
        out["pred_condition"] = cond.classify_overall(req.description)
    return {"id": req.id, "analysis": out}

@app.post("/api/recommendations")
def recommendations(req: RecommendRequest):
    if df_prepped is None:
        return {"error":"No data loaded"}
    matches = df_prepped.index[df_prepped["id"] == req.reference_id].tolist()
    if not matches:
        return {"error":"reference_id not found"}
    idx = int(matches[0])
    df_res = rec.recommend_by_index(idx, top_n=req.top_n)
    # convert to nested/flat/wide based on req.style (you can implement this)
    return df_res.to_dict(orient="records")

@app.post("/api/advice")
def advice(req: AdviceRequest):
    # call advisor.advise with mapped params
    if advisor is None:
        return {"error":"advisor not ready"}
    payload = advisor.advise(budget=req.budget, budget_range=req.budget_range, city=req.city, title=req.title,
                             reference_id=req.reference_id, top_n=req.top_n, include_condition=req.include_condition)
    return payload






