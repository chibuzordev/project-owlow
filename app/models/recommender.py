import preprocess, condition_classifier, analyzer, advisor

class Recommender:
    """
    Enhanced Recommender with:
      - recommend_by_index(...)
      - recommend_by_title(...)
      - recommend_by_preferences(...)
      - build_json_recommendation_table(...)
    Supports precomputing & exporting similarity tables.
    """

    OUTPUT_COLS = ["id", "title", "price", "similarity_score", "rank"]

    def __init__(self,
                 text_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 weight_text: float = 0.6,
                 weight_numeric: float = 0.2,
                 weight_cat: float = 0.1,
                 geo_max_km: float = 50.0):
        self.text_model_name = text_model_name
        self.weight_text = weight_text
        self.weight_numeric = weight_numeric
        self.weight_cat = weight_cat
        self.geo_max_km = geo_max_km

        self._sbert = None
        self._tfidf = None

        self.df = None
        self.combined = None
        self.text_emb = None
        self.nn = None
        self.cat_columns = None
        self.scaler = None

    # ---------------- Fit model ----------------
    def fit(self, df: pd.DataFrame,
            categorical_cols: Optional[List[str]] = None,
            use_numeric: Optional[List[str]] = None):
        """
        Fit recommender on preprocessed df.
        """
        self.df = df.reset_index(drop=True).copy()

        if use_numeric is None:
            use_numeric = ["price", "areaM2", "bedrooms", "pricePerM2_value"]

        # text embeddings
        self.df["text"] = (
            self.df.get("title", "").fillna("") + ". " +
            self.df.get("description", "").fillna("")
        ).str.replace("\n", " ").str.strip()
        texts = self.df["text"].fillna("").tolist()
        text_emb = self._encode_text(texts)
        self.text_emb = self._row_normalize(text_emb)

        # numeric matrix
        num_df = self.df.reindex(columns=use_numeric).fillna(0).astype(float)
        num_df = np.log1p(np.clip(num_df.values, a_min=0, a_max=None))
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        num_scaled = self._row_normalize(self.scaler.fit_transform(num_df))

        # categorical one-hot
        if categorical_cols is None:
            categorical_cols = ["voivodeship", "propertyType", "finishing", "market", "transactionType", "city"]
        for c in categorical_cols:
            if c not in self.df.columns:
                self.df[c] = ""
        cat_df = pd.get_dummies(self.df[categorical_cols].fillna("").astype(str), dummy_na=False)
        self.cat_columns = cat_df.columns.tolist()
        cat_mat = self._row_normalize(cat_df.values.astype(float)) if cat_df.shape[1] > 0 else np.zeros((len(self.df), 1))

        # combine weighted
        combined = np.hstack([
            self.text_emb * self.weight_text,
            num_scaled * self.weight_numeric,
            cat_mat * self.weight_cat
        ])
        self.combined = self._row_normalize(combined).astype("float32")

        # fallback nearest neighbors
        from sklearn.neighbors import NearestNeighbors
        self.nn = NearestNeighbors(n_neighbors=min(50, len(self.df)), metric="cosine").fit(self.combined)

        return self

    # ---------------- Similarity search helpers ----------------
    def _encode_text(self, texts: List[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
            if self._sbert is None:
                self._sbert = SentenceTransformer(self.text_model_name)
            emb = self._sbert.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return emb
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            if self._tfidf is None:
                self._tfidf = TfidfVectorizer(max_features=2048, ngram_range=(1, 2))
                return self._tfidf.fit_transform(texts).toarray()
            return self._tfidf.transform(texts).toarray()

    @staticmethod
    def _row_normalize(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def _nn_search(self, idx: int, top_k: int = 5) -> List[Tuple[int, float]]:
        dists, ids = self.nn.kneighbors(self.combined[idx].reshape(1, -1), n_neighbors=min(len(self.df), top_k))
        ids, dists = ids[0].tolist(), dists[0].tolist()
        return [(int(i), max(0.0, 1.0 - d)) for i, d in zip(ids, dists)]

    # ---------------- Core recommenders ----------------
    def recommend_by_index(self, idx: int, top_n: int = 5) -> pd.DataFrame:
        neighbors = self._nn_search(idx, top_k=top_n + 1)
        rows = []
        rank = 1
        for j, score in neighbors:
            if j == idx:
                continue
            rows.append({
                "id": str(self.df.iloc[j].get("id", j)),
                "title": self.df.iloc[j].get("title", ""),
                "price": float(self.df.iloc[j].get("price", np.nan)),
                "similarity_score": float(score),
                "rank": rank
            })
            rank += 1
            if rank > top_n:
                break
        return pd.DataFrame(rows)

    def recommend_by_title(self, title: str, top_n: int = 5) -> pd.DataFrame:
        # Simple: find property containing this string in title
        idxs = self.df.index[self.df["title"].str.contains(title, case=False, na=False)].tolist()
        if not idxs:
            return pd.DataFrame(columns=self.OUTPUT_COLS)
        return self.recommend_by_index(idxs[0], top_n=top_n)

    def recommend_by_preferences(self, *, budget: Optional[float] = None, budget_pct: float = 0.1,
                                 location: Optional[Dict[str, str]] = None, top_n: int = 5) -> pd.DataFrame:
        # Filter candidates
        candidates = self.df.copy()
        if budget:
            low, high = budget * (1 - budget_pct), budget * (1 + budget_pct)
            candidates = candidates[(candidates["price"] >= low) & (candidates["price"] <= high)]
        if location and "city" in location:
            candidates = candidates[candidates["city"].str.lower() == location["city"].lower()]
        if candidates.empty:
            return pd.DataFrame(columns=self.OUTPUT_COLS)
        # Pick first candidate and recommend similar
        idx = candidates.index[0]
        return self.recommend_by_index(idx, top_n=top_n)

    # ---------------- Exporters ----------------
    def build_json_recommendation_table(self, top_n: int = 5) -> List[dict]:
        all_recs = []
        for idx in range(len(self.df)):
            neighbors = self._nn_search(idx, top_k=top_n + 1)
            matches = []
            for j, score in neighbors:
                if j == idx:
                    continue
                matches.append({
                    "match_id": str(self.df.iloc[j].get("id", j)),
                    "title": self.df.iloc[j].get("title", ""),
                    "price": float(self.df.iloc[j].get("price", np.nan)),
                    "similarity_score": float(score),
                    "location": {
                        "city": self.df.iloc[j].get("city", ""),
                        "voivodeship": self.df.iloc[j].get("voivodeship", ""),
                        "district": self.df.iloc[j].get("district", ""),
                        "neighborhood": self.df.iloc[j].get("neighborhood", "")
                    }
                })
                if len(matches) >= top_n:
                    break
            all_recs.append({
                "reference_id": str(self.df.iloc[idx].get("id", idx)),
                "recommendations": matches
            })
        return all_recs

    def export_recommendations(self, path: str = None, top_n: int = 5, fmt: str = "json", style: str = "wide"):
        """
        Export recommendations with multiple output styles:
        - style="nested": {reference_id, recommendations:[{match_id,title,...}, ...]}
        - style="normalized": {"recommendationTable":[...], "listingTable":[...]}
        - style="wide": {reference_id, title, condition, match_1_id, match_1_title, ...}
        """
        data = self.build_json_recommendation_table(top_n=top_n)

        if style == "nested":
            output = data

        elif style == "normalized":
            rec_table = []
            listing_table = {}
            for r in data:
                ref_id = r["reference_id"]
                for rank, m in enumerate(r["recommendations"], start=1):
                    rec_table.append({
                        "reference_id": ref_id,
                        "match_id": m["match_id"],
                        "similarity_score": m["similarity_score"],
                        "rank": rank
                    })
                    listing_table[m["match_id"]] = {
                        "id": m["match_id"],
                        "title": m.get("title", ""),
                        "price": m.get("price", None),
                        "city": m.get("location", {}).get("city", ""),
                        "voivodeship": m.get("location", {}).get("voivodeship", ""),
                    }
            output = {"recommendationTable": rec_table, "listingTable": list(listing_table.values())}

        elif style == "wide":
            rows = []
            for r in data:
                ref_id = r["reference_id"]
                ref_row = {
                    "reference_id": ref_id,
                    "reference_title": next((x.get("title","") for x in self.df.to_dict("records") if str(x.get("id"))==ref_id), ""),
                    "reference_condition": next((x.get("pred_condition","Nieznany") for x in self.df.to_dict("records") if str(x.get("id"))==ref_id), "Nieznany")
                }
                # add matches as flattened fields
                for i, m in enumerate(r["recommendations"], start=1):
                    ref_row[f"match_{i}_id"] = m["match_id"]
                    ref_row[f"match_{i}_title"] = m.get("title", "")
                    ref_row[f"match_{i}_price"] = m.get("price", None)
                    ref_row[f"match_{i}_location"] = m.get("location", {}).get("city", "")
                    ref_row[f"match_{i}_condition"] = "Nieznany"  # placeholder unless ConditionClassifier applied
                rows.append(ref_row)
            output = rows

        else:
            raise ValueError("Unsupported style. Choose from ['nested','normalized','wide'].")

        # write out
        if fmt == "json":
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)
                print(f"[INFO] Saved recommendations ({style}) to {path}")
            return output

        elif fmt == "csv":
            import pandas as pd
            df = pd.DataFrame(output if isinstance(output, list) else [output])
            if path:
                df.to_csv(path, index=False)
                print(f"[INFO] Saved recommendations ({style}) to {path}")
            return df

        else:
            raise ValueError("Unsupported format. Use 'csv' or 'json'.")


    def post_recommendations(self, api_url: str, top_n: int = 5, auth_token: str = None, timeout: int = 30):
        data = self.build_json_recommendation_table(top_n=top_n)
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        try:
            resp = requests.post(api_url, json=data, headers=headers, timeout=timeout)
            resp.raise_for_status()
            print(f"[INFO] Successfully POSTed {len(data)} recommendations to {api_url}")
            return resp.json() if resp.headers.get("Content-Type", "").startswith("application/json") else resp.text
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to POST recommendations: {e}")
            return None
