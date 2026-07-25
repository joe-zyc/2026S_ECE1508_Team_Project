"""Grounded recommendation explanation generation."""

import json

from openai import OpenAI

from .exceptions import InvalidModelOutputError, UpstreamModelError
from .schemas import Product, SearchQuery


RECOMMENDATION_SYSTEM_PROMPT = """
Write one concise, friendly shopping recommendation reason for each product.
Tailor it to the request using only the supplied request, parsed query, and
product facts. Do not infer unlisted features from a title. If a requested
feature is not confirmed, naturally suggest verifying it. Do not repeat price,
rating, review count, rank, URL, or metadata already displayed. Return every
product exactly once in its original order; never rerank, replace, or omit it.
""".strip()


class OpenAIRecommendationGenerator:
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

    def generate(
        self,
        user_request: str,
        parsed_query: SearchQuery,
        products: list[Product],
    ) -> list[str]:
        if not products:
            return []
        count = len(products)
        schema = {
            "type": "object",
            "properties": {
                "recommendations": {
                    "type": "array",
                    "minItems": count,
                    "maxItems": count,
                    "items": {
                        "type": "object",
                        "properties": {
                            "rank": {"type": "integer"},
                            "asin": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["rank", "asin", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["recommendations"],
            "additionalProperties": False,
        }
        payload = {
            "user_request": user_request,
            "parsed_query": parsed_query.model_dump(),
            "products_in_required_order": [
                {
                    "rank": rank,
                    **product.model_dump(exclude={"img_url", "product_url"}),
                }
                for rank, product in enumerate(products, start=1)
            ],
        }
        try:
            response = self._client.responses.create(
                model=self._model,
                input=[
                    {"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "product_recommendations",
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
        except Exception as exc:
            raise UpstreamModelError(
                "Recommendation model is unavailable"
            ) from exc

        try:
            generated = json.loads(response.output_text)["recommendations"]
            expected_asins = [product.asin for product in products]
            actual_asins = [str(item["asin"]) for item in generated]
            actual_ranks = [item["rank"] for item in generated]
            reasons = [item["reason"].strip() for item in generated]
            if (
                actual_asins != expected_asins
                or actual_ranks != list(range(1, count + 1))
                or any(not reason for reason in reasons)
            ):
                raise ValueError("identity, order, or reason mismatch")
            return reasons
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise InvalidModelOutputError(
                "Recommendation model returned invalid structured output"
            ) from exc
