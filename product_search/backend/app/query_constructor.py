"""OpenAI-backed natural-language query constructor."""

import json

from openai import OpenAI
from pydantic import ValidationError

from .exceptions import InvalidModelOutputError, UpstreamModelError
from .schemas import SearchQuery


QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "product_description": {"type": "string"},
        "price_min": {"type": ["number", "null"]},
        "price_max": {"type": ["number", "null"]},
        "min_stars": {"type": ["number", "null"]},
        "min_reviews": {"type": ["integer", "null"]},
        "brand": {"type": ["string", "null"]},
        "sort_by": {
            "type": "string",
            "enum": [
                "relevance",
                "price_low_to_high",
                "price_high_to_low",
                "rating",
                "popularity",
            ],
        },
    },
    "required": [
        "product_description",
        "price_min",
        "price_max",
        "min_stars",
        "min_reviews",
        "brand",
        "sort_by",
    ],
    "additionalProperties": False,
}

QUERY_SYSTEM_PROMPT = """
You are a shopping-query parser. Convert the request into the supplied schema.

- product_description must be a concise, embedding-friendly product phrase.
  Include important use, recipient, size, compatibility, style, and feature
  terms, but exclude values represented by other fields.
- Under/below/up to/budget amounts are price_max. Over/above/at least amounts
  are price_min. Explicit ranges fill both. Around/about/roughly an amount uses
  a +/- 20 percent range. Never convert currencies.
- "highly/well/good rated" means min_stars 4.0; "top/very highly/excellent
  rated" means 4.5. Do not invent a review count.
- Extract brand or main_category only when supported by the request.
- sort_by is relevance by default; use price_low_to_high for cheapest,
  price_high_to_low for most expensive, rating for best rated, and popularity
  for popular/bestselling/most reviewed.
- Use null for every unspecified optional value. Never invent constraints.
""".strip()


class OpenAIQueryConstructor:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        client: OpenAI | None = None,
    ) -> None:
        self._model = model
        self._client = client or OpenAI(api_key=api_key, timeout=timeout_seconds)

    def construct(self, user_input: str) -> SearchQuery:
        try:
            response = self._client.responses.create(
                model=self._model,
                input=[
                    {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "shopping_query",
                        "schema": QUERY_SCHEMA,
                        "strict": True,
                    }
                },
            )
        except Exception as exc:
            raise UpstreamModelError("Query construction model is unavailable") from exc

        try:
            return SearchQuery.model_validate(json.loads(response.output_text))
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise InvalidModelOutputError(
                "Query construction model returned invalid structured output"
            ) from exc
