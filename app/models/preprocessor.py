import analyzer, condition_classifier, recommender, advisor 

class Preprocessor:
    """Flatten nested fields and create useful engineered columns."""

    @staticmethod
    def _safe_literal(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) else x
        except Exception:
            return x

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- Price per m2 ---
        df["pricePerM2"] = df.get("pricePerM2").apply(self._safe_literal)
        df["pricePerM2_value"] = df["pricePerM2"].apply(lambda x: x.get("value") if isinstance(x, dict) else np.nan)
        df["pricePerM2_currency"] = df["pricePerM2"].apply(lambda x: x.get("currency") if isinstance(x, dict) else None)

        # --- Location ---
        df["location"] = df.get("location").apply(self._safe_literal)
        df["voivodeship"] = df["location"].apply(lambda x: x.get("voivodeship") if isinstance(x, dict) else "")
        df["city"] = df["location"].apply(lambda x: x.get("city") if isinstance(x, dict) else "")
        df["district"] = df["location"].apply(lambda x: x.get("district") if isinstance(x, dict) else "")
        df["neighborhood"] = df["location"].apply(lambda x: x.get("neighborhood") if isinstance(x, dict) else "")

        # --- Geo ---
        df["geoLocation"] = df.get("geoLocation").apply(self._safe_literal)
        df["latitude"] = df["geoLocation"].apply(
            lambda x: x.get("coordinates")[0] if isinstance(x, dict) and x.get("coordinates") else np.nan
        )
        df["longitude"] = df["geoLocation"].apply(
            lambda x: x.get("coordinates")[1] if isinstance(x, dict) and x.get("coordinates") else np.nan
        )

        # --- Additional Info ---
        df["additionalInfo"] = df.get("additionalInfo").apply(self._safe_literal)

        def parse_additional(x):
            if not isinstance(x, list):
                return {}
            out = {}
            for item in x:
                label = item.get("label")
                val = item.get("value")
                if label:
                    out[label] = val
            return out

        add = df["additionalInfo"].apply(parse_additional)
        df["finishing"] = add.apply(lambda d: d.get("Stan wykończenia", ""))
        df["floor_raw"] = add.apply(lambda d: d.get("Piętro", ""))
        df["heating"] = add.apply(lambda d: d.get("Ogrzewanie", ""))
        df["ownership"] = add.apply(lambda d: d.get("Forma własności", ""))
        df["market"] = add.apply(lambda d: d.get("Rynek", ""))
        df["extras"] = add.apply(lambda d: d.get("Informacje dodatkowe", ""))

        # --- Parse floor ---
        def parse_floor(x):
            if not isinstance(x, str) or x.strip() == "":
                return (np.nan, np.nan)
            m = re.search(r"(\d+)\s*/\s*(\d+)", x)
            if m:
                return (float(m.group(1)), float(m.group(2)))
            m2 = re.search(r"(\d+)", x)
            if m2:
                return (float(m2.group(1)), np.nan)
            return (np.nan, np.nan)

        floors = df["floor_raw"].apply(parse_floor)
        df["floor_num"] = floors.apply(lambda t: t[0])
        df["total_floors"] = floors.apply(lambda t: t[1])

        # --- Clean text fields ---
        for c in ["description", "title", "neighborhood", "voivodeship", "city", "district", "finishing", "market"]:
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str)

        # --- Ensure numeric ---
        for n in ["price", "areaM2", "bedrooms", "pricePerM2_value"]:
            if n not in df.columns:
                df[n] = np.nan
            df[n] = pd.to_numeric(df[n], errors="coerce")

        # --- Ensure image url column exists ---
        if "url" not in df.columns:
            df["url"] = ""

        return df
