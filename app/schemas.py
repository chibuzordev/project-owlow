# -------------------- Batch 3: Advisor + Image/LLM PropertyAnalyzer + Pipeline --------------------
import time
import math
import json
import io
import base64
import requests
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple

from app.models.datafetcher import DataFetcher
from app.models.preprocessor import Preprocessor
from app.models.condition_classifier import ConditionClassifier
from app.models.recommender import Recommender
from app.models.analyzer import PropertyAnalyzer
from app.models.advisor import Advisor


# -------------------- Image + LLM Property Analyzer --------------------
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
    _openai_client = OpenAI()
except Exception:
    _OPENAI_AVAILABLE = False
    _openai_client = None

class PropertyAnalyzer:
    """
    Analyze an individual listing using:
      - cheap rule-based ConditionClassifier (text)
      - optional LLM+vision pass (images + description) using OpenAI (if configured)
    The LLM pass is optional and safe-guarded (errors won't break pipeline).
    """

    def __init__(self, llm_model: str = "gpt-4o-mini", max_images: int = 2, max_size: int = 512):
        self.llm_model = llm_model
        self.max_images = max_images
        self.max_size = max_size
        self.cc = ConditionClassifier()

    def _download_resize_b64(self, url: str) -> Optional[str]:
        """Return base64 JPEG string or None on failure."""
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            img.thumbnail((self.max_size, self.max_size), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:
            # non-fatal — return None
            # print(f"[WARN] Image fetch/resize failed: {e}")
            return None

    def rule_analyze(self, description: str) -> Dict[str,Any]:
        """Cheap baseline: rule-based classification (Polish labels)."""
        overall = self.cc.classify_overall(description)
        elements = self.cc.classify_elements(description)
        return {"pred_condition": overall, "elements": elements}

    def llm_analyze(self, image_urls: List[str], description: str) -> Dict[str,Any]:
        """
        Use LLM + images to produce structured JSON:
        {condition, kitchen, bathroom, floors, windows, installations}
        If OpenAI client not available or request fails, returns {}.
        """
        if not _OPENAI_AVAILABLE or _openai_client is None:
            return {}

        # Build multimodal inputs: compress images to small base64s
        inputs = []
        if description and isinstance(description, str):
            inputs.append({"type":"text","text": f"Opis nieruchomości: {description}"})

        added = 0
        for u in (image_urls or [])[:self.max_images]:
            if not isinstance(u, str) or u.strip()=="":
                continue
            b64 = self._download_resize_b64(u)
            if b64:
                # embed data URI style
                inputs.append({"type":"image_url","image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                added += 1

        if len(inputs) == 0:
            return {}

        system_prompt = (
            "Jesteś analitykiem stanu technicznego nieruchomości. "
            "Na podstawie opisu i przesłanych zdjęć zwróć JSON z polskimi etykietami: "
            "condition (Wyremontowane / Do odświeżenia / Do kapitalnego remontu / Nieznany), "
            "kuchnia, łazienka, podłogi, okna, instalacje. Krótko, rzeczowo."
        )

        try:
            resp = _openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role":"system","content":system_prompt},
                    {"role":"user","content": inputs}
                ],
                max_tokens=300
            )
            # Access content safely (may be ChatCompletionMessage object)
            content = None
            # Newer client formats:
            try:
                # support object with .message.content
                content = resp.choices[0].message.content
            except Exception:
                try:
                    content = resp.choices[0].text
                except Exception:
                    content = None

            if not content:
                return {}

            # Try to parse JSON out of reply (robust)
            parsed = None
            try:
                parsed = json.loads(content)
            except Exception:
                # If model returned plain text, heuristically extract JSON substring
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end>start:
                    try:
                        parsed = json.loads(content[start:end+1])
                    except Exception:
                        parsed = None

            return parsed if isinstance(parsed, dict) else {}
        except Exception as e:
            # don't fail the pipeline on LLM errors
            # print(f"[WARN] LLM analyze failed: {e}")
            return {}

    def analyze(self, image_urls: List[str], description: str, use_llm: bool = False) -> Dict[str,Any]:
        """Run rule-based then optional LLM analysis; return merged result."""
        out = self.rule_analyze(description)
        if use_llm:
            llm_out = self.llm_analyze(image_urls, description)
            # merge: llm results override rule-based when present
            for k,v in (llm_out or {}).items():
                out[k] = v
        return out





