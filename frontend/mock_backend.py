"""
mock_backend.py

This module is the CONTRACT between the frontend (app.py) and the
backend (query parsing -> retrieval -> ranking -> explanation generation).

Right now it fakes that whole pipeline with simple keyword/price matching
over a small hardcoded catalog, so the UI can be built and demoed today.

Whoever wires up the real backend just needs to replace the body of
get_recommendations() with a real call, while keeping the same input/output
shape:

    Input:  a natural language string, e.g. "32 inch TV under $500"
    Output: a list of up to `top_k` dicts, each shaped like:
        {
            "title": str,
            "imgUrl": str,
            "productURL": str,
            "stars": float,
            "price": float,
            "explanation": str,   # grounded, why this matches the request
        }
"""

import re

# ---------------------------------------------------------------------------
# Mock catalog. Real fields (title, imgUrl, productURL, stars, price) match
# the schema confirmed for the real product database.
# ---------------------------------------------------------------------------
CATALOG = [
    {
        "title": "Sion Softside Expandable Roller Luggage, Black, Checked-Large 29-Inch",
        "imgUrl": "https://m.media-amazon.com/images/I/815dLQKYIYL._AC_UL320_.jpg",
        "productURL": "https://www.amazon.com/dp/B014TMV5YE",
        "stars": 4.5,
        "price": 139.99,
        "category": "luggage",
        "tags": ["luggage", "suitcase", "expandable", "checked", "black"],
    },
    {
        "title": "Luggage Sets Expandable PC+ABS Durable Suitcase Double Wheels TSA Lock Blue",
        "imgUrl": "https://m.media-amazon.com/images/I/81bQlm7vf6L._AC_UL320_.jpg",
        "productURL": "https://www.amazon.com/dp/B07GDLCQXV",
        "stars": 4.5,
        "price": 169.99,
        "category": "luggage",
        "tags": ["luggage", "suitcase", "set", "tsa lock", "blue"],
    },
    {
        "title": "Platinum Elite Softside Expandable Checked Luggage, 8 Wheel Spinner Suitcase, TSA Lock, True Navy Blue, Checked Medium 25-Inch",
        "imgUrl": "https://m.media-amazon.com/images/I/71EA35zvJBL._AC_UL320_.jpg",
        "productURL": "https://www.amazon.com/dp/B07XSCCZYG",
        "stars": 4.6,
        "price": 365.49,
        "category": "luggage",
        "tags": ["luggage", "suitcase", "spinner", "tsa lock", "navy"],
    },
    {
        "title": "Vizio 32-inch Class V-Series LED HD TV with Vivid Picture Engine",
        "imgUrl": "https://placehold.co/320x320?text=32in+TV",
        "productURL": "https://www.amazon.com/dp/EXAMPLE001",
        "stars": 4.4,
        "price": 189.99,
        "category": "tv",
        "tags": ["tv", "television", "32 inch", "hd", "picture quality"],
    },
    {
        "title": "TCL 32-inch 4-Series HD Smart TV with Bright, Sharp Picture",
        "imgUrl": "https://placehold.co/320x320?text=32in+Smart+TV",
        "productURL": "https://www.amazon.com/dp/EXAMPLE002",
        "stars": 4.6,
        "price": 229.99,
        "category": "tv",
        "tags": ["tv", "television", "32 inch", "smart tv", "picture quality"],
    },
    {
        "title": "Samsung 32-inch QLED HD TV with Quantum Dot Picture Technology",
        "imgUrl": "https://placehold.co/320x320?text=32in+QLED+TV",
        "productURL": "https://www.amazon.com/dp/EXAMPLE003",
        "stars": 4.7,
        "price": 479.99,
        "category": "tv",
        "tags": ["tv", "television", "32 inch", "qled", "picture quality", "premium"],
    },
    {
        "title": "Dell Inspiron 15 Laptop, 16GB RAM, 512GB SSD",
        "imgUrl": "https://placehold.co/320x320?text=Laptop",
        "productURL": "https://www.amazon.com/dp/EXAMPLE004",
        "stars": 4.3,
        "price": 649.99,
        "category": "laptop",
        "tags": ["laptop", "computer", "ssd", "16gb ram"],
    },
    {
        "title": "Sony WH-1000XM4 Wireless Noise Cancelling Headphones",
        "imgUrl": "https://placehold.co/320x320?text=Headphones",
        "productURL": "https://www.amazon.com/dp/EXAMPLE005",
        "stars": 4.8,
        "price": 279.99,
        "category": "headphones",
        "tags": ["headphones", "wireless", "noise cancelling", "audio"],
    },
    {
        "title": "Ninja 12-Cup Programmable Coffee Maker",
        "imgUrl": "https://placehold.co/320x320?text=Coffee+Maker",
        "productURL": "https://www.amazon.com/dp/EXAMPLE006",
        "stars": 4.5,
        "price": 89.99,
        "category": "coffee maker",
        "tags": ["coffee maker", "kitchen", "programmable"],
    },
]

CATEGORY_KEYWORDS = {
    "tv": ["tv", "television"],
    "luggage": ["luggage", "suitcase", "roller bag", "carry-on", "carry on"],
    "laptop": ["laptop", "notebook computer"],
    "headphones": ["headphones", "earbuds", "headset"],
    "coffee maker": ["coffee maker", "coffee machine"],
}


def _parse_budget(query: str):
    """Extract a dollar budget like '$500' or '500 dollars' from the query.

    Prioritizes an explicit '$' amount so other numbers in the query
    (e.g. '32 inch') aren't mistaken for the budget.
    """
    q = query.lower()

    # Highest priority: an explicit dollar sign amount, e.g. "$500"
    match = re.search(r"\$\s?(\d{2,6}(?:\.\d{1,2})?)", q)
    if match:
        return float(match.group(1))

    # Next: "500 dollars" / "500 usd"
    match = re.search(r"(\d{2,6}(?:\.\d{1,2})?)\s?(?:dollars|usd)\b", q)
    if match:
        return float(match.group(1))

    # Next: "budget of 500" / "under 500" / "budget: 500"
    match = re.search(r"(?:budget(?:\s+of)?|under|below|less than)\D{0,5}(\d{2,6}(?:\.\d{1,2})?)", q)
    if match:
        return float(match.group(1))

    return None


def _parse_category(query: str):
    q = query.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return category
    return None


def get_recommendations(query: str, top_k: int = 3):
    """
    Fake pipeline: parse -> filter -> rank -> explain.
    Returns a list of up to `top_k` product dicts (see module docstring for shape).
    """
    budget = _parse_budget(query)
    category = _parse_category(query)

    candidates = CATALOG
    if category:
        candidates = [p for p in candidates if p["category"] == category]
    if budget:
        candidates = [p for p in candidates if p["price"] <= budget]

    # Fallback: if filters were too strict and nothing matched, widen to full catalog
    if not candidates:
        candidates = CATALOG

    # Rank by rating (stand-in for the real relevance + business-signal ranker)
    ranked = sorted(candidates, key=lambda p: p["stars"], reverse=True)[:top_k]

    results = []
    for p in ranked:
        reasons = []
        if category:
            reasons.append(f"matches the '{category}' category you asked about")
        if budget:
            reasons.append(f"fits within your ${budget:.0f} budget at ${p['price']:.2f}")
        reasons.append(f"rated {p['stars']}/5 stars")
        explanation = "This option " + ", and ".join(reasons) + "."

        results.append({
            "title": p["title"],
            "imgUrl": p["imgUrl"],
            "productURL": p["productURL"],
            "stars": p["stars"],
            "price": p["price"],
            "explanation": explanation,
        })

    return results
