from typing import Optional, List, Dict, Any
import pandas as pd
import math
import json
import requests
from .recommender import Recommender

class Advisor:
    """
    Wraps a fitted Recommender instance to provide:
     - alternatives within budget / range / location
     - historical trend analysis (from df_history if provided)
     - relaxation suggestions when too few matches
     - export/post helpers
    Non-invasive: does not modify Recommender internals.
    """

    def __init__(self, recommender: Recommender, df_history: Optional[pd.DataFrame] = None):
        self.rec = recommender
        self.df_history = df_history

    def _get_recs(self, *, budget=None, budget_pct=0.1, location=None, title=None, reference_id=None, top_n=5):
        # Prefer recommend_by_preferences if present
        try:
            if hasattr(self.rec, "recommend_by_preferences"):
                return self.rec.recommend_by_preferences(budget=budget, budget_pct=budget_pct, location=location, top_n=top_n)
        except Exception:
            pass

        # title fallback
        if title:
            try:
                return self.rec.recommend_by_title(title, top_n=top_n)
            except Exception:
                pass

        # reference_id -> index fallback
        if reference_id:
            try:
                if hasattr(self.rec, "df") and "id" in self.rec.df.columns:
                    matches = self.rec.df.index[self.rec.df["id"] == reference_id].tolist()
                    if matches:
                        return self.rec.recommend_by_index(matches[0], top_n=top_n)
            except Exception:
                pass

        # Last: return empty frame
        return pd.DataFrame(columns=self.rec.OUTPUT_COLS)

    def suggest_alternatives(self, *, budget: Optional[float] = None, budget_range: Optional[List[float]] = None,
                             budget_pct: float = 0.1, location: Optional[Dict[str,str]] = None,
                             title: Optional[str] = None, reference_id: Optional[str] = None,
                             top_n: int = 5, include_condition: bool = False) -> List[Dict[str,Any]]:
        # normalize budget into single or range
        budget_arg = None
        if budget_range and isinstance(budget_range, (list,tuple)) and len(budget_range)==2:
            # pass the tuple through as "budget" - Recommender will understand as range if implemented
            budget_arg = (budget_range[0], budget_range[1])
        elif budget is not None:
            budget_arg = budget

        df_res = self._get_recs(budget=budget_arg, budget_pct=budget_pct, location=location, title=title, reference_id=reference_id, top_n=top_n)

        out = []
        for _, r in (df_res.reset_index(drop=True).iterrows() if not df_res.empty else []):
            rec = {
                "id": str(r.get("id","")),
                "title": str(r.get("title","")),
                "price": float(r.get("price", math.nan)) if not pd.isna(r.get("price", None)) else None,
                "similarity_score": float(r.get("similarity_score", 0.0)),
                "rank": int(r.get("rank", 0))
            }
            if include_condition and hasattr(self.rec, "df") and "pred_condition" in self.rec.df.columns:
                try:
                    idx = int(self.rec.df.index[self.rec.df["id"]==rec["id"]][0])
                    rec["pred_condition"] = self.rec.df.iloc[idx].get("pred_condition", "Nieznany")
                except Exception:
                    rec["pred_condition"] = "Nieznany"
            out.append(rec)
        return out

    def analyze_trends(self, city: Optional[str] = None, months: int = 6) -> List[Dict[str,Any]]:
        if self.df_history is None:
            return []
        df = self.df_history.copy()
        if "createdAt" not in df.columns or "pricePerM2_value" not in df.columns:
            return []
        df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
        df = df.dropna(subset=["createdAt"])
        df["month"] = df["createdAt"].dt.to_period("M")
        if city:
            df = df[df["city"].str.lower()==str(city).lower()]
        trend = df.groupby("month")["pricePerM2_value"].mean().dropna().tail(months)
        return [{"month": str(m), "avg_price_per_m2": float(v)} for m,v in trend.items()]

    def relax_criteria(self, alternatives: List[Dict[str,Any]], budget: Optional[float], budget_range: Optional[List[float]], city: Optional[str]) -> List[str]:
        suggestions = []
        if len(alternatives) >= 5:
            suggestions.append("Wystarczająca liczba dopasowań — brak potrzeby luzowania kryteriów.")
            return suggestions
        if city:
            suggestions.append(f"Znaleziono niewiele ofert w {city} przy aktualnych kryteriach.")
        if budget_range:
            low, high = budget_range
            suggestions.append(f"Rozważ poszerzenie budżetu poza zakres {int(low)}–{int(high)} PLN.")
        elif budget:
            suggestions.append(f"Rozważ zwiększenie budżetu o 10% (z {int(budget)} PLN).")
        else:
            suggestions.append("Rozważ większy budżet lub rozszerzenie lokalizacji.")
        suggestions.append("Rozważ dopuszczenie ofert wymagających lekkiego odświeżenia (np. 'Do odświeżenia').")
        suggestions.append("Rozszerz obszar poszukiwań na sąsiednie dzielnice/miasta.")
        return suggestions

    def advise(self, *, budget: Optional[float] = None, budget_range: Optional[List[float]] = None,
               budget_pct: float = 0.1, city: Optional[str] = None, title: Optional[str] = None,
               reference_id: Optional[str] = None, top_n: int = 5, include_condition: bool = False) -> Dict[str,Any]:
        location = {"city": city} if city else None
        alternatives = self.suggest_alternatives(budget=budget, budget_range=budget_range, budget_pct=budget_pct,
                                                 location=location, title=title, reference_id=reference_id,
                                                 top_n=top_n, include_condition=include_condition)
        trends = self.analyze_trends(city=city)
        advice = self.relax_criteria(alternatives, budget, budget_range, city)
        return {"budget": budget if budget is not None else budget_range, "city": city, "alternatives": alternatives, "historical_trends": trends, "advice": advice}

    def export_advice_json(self, out_path: Optional[str] = None, **advise_kwargs) -> Dict[str,Any]:
        payload = self.advise(**advise_kwargs)
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload

    def post_advice(self, api_url: str, auth_token: Optional[str] = None, timeout: int = 30, **advise_kwargs) -> Optional[Dict[str,Any]]:
        payload = self.advise(**advise_kwargs)
        headers = {"Content-Type":"application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        try:
            resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            try:
                return resp.json()
            except Exception:
                return {"status":"ok","http_status":resp.status_code}
        except Exception as e:
            print(f"[ERROR] Failed to POST advice: {e}")
            return None





