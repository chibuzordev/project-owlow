# -------------------- ConditionClassifier --------------------
class ConditionClassifier:
    """
    Rule-based condition classifier using Polish keywords.
    - Evaluates overall condition
    - Evaluates elements: kuchnia, łazienka, podłogi, okna, instalacje
    Labels are in Polish (comments in English).
    """

    def __init__(self):
        # Global keywords
        self.renovated = ["po remoncie", "wyremontowane", "odnowione", "świeżo", "nowe"]
        self.refresh = ["do odświeżenia", "kosmetyczne", "lekkiego remontu"]
        self.full_reno = ["kapitalny remont", "zniszczone", "stare instalacje", "do remontu"]

        # Element-specific
        self.kitchen_new = ["nowa kuchnia", "wyremontowana kuchnia", "aneks kuchenny nowy"]
        self.bathroom_new = ["nowa łazienka", "wyremontowana łazienka"]
        self.floors_new = ["nowa podłoga", "panele nowe", "parkiet po renowacji"]
        self.windows_new = ["nowe okna", "wymienione okna"]
        self.installations_new = ["nowe instalacje", "wymieniona instalacja", "nowa elektryka"]

    def classify_overall(self, text: str) -> str:
        if not isinstance(text, str) or text.strip() == "":
            return "Nieznany"
        t = text.lower()
        if any(k in t for k in self.renovated):
            return "Wyremontowane"
        if any(k in t for k in self.refresh):
            return "Do odświeżenia"
        if any(k in t for k in self.full_reno):
            return "Do kapitalnego remontu"
        return "Nieznany"

    def classify_elements(self, text: str) -> dict:
        if not isinstance(text, str) or text.strip() == "":
            return {}
        t = text.lower()
        return {
            "kuchnia": "Nowa" if any(k in t for k in self.kitchen_new) else "Brak danych",
            "łazienka": "Nowa" if any(k in t for k in self.bathroom_new) else "Brak danych",
            "podłogi": "Nowe" if any(k in t for k in self.floors_new) else "Brak danych",
            "okna": "Nowe" if any(k in t for k in self.windows_new) else "Brak danych",
            "instalacje": "Nowe" if any(k in t for k in self.installations_new) else "Brak danych"
        }
