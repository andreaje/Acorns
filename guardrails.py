GUARDRAIL_CATEGORIES = {
    "personalized_financial_advice_boundary",
    "out_of_domain_request",
}

IN_DOMAIN_FINANCIAL_TOPICS = {
    "budgeting",
    "saving",
    "investing education",
    "retirement planning",
    "debt",
    "emergency funds",
    "college savings",
    "high-level tax education",
    "financial goals",
    "acorns/product-related financial questions",
}

OUT_OF_DOMAIN_EXAMPLES = {
    "cooking / baking",
    "medical",
    "legal",
    "technical support",
    "home repair",
    "relationship advice",
    "general trivia",
    "travel planning not tied to budgeting",
}


def evaluate_guardrails(
    user_message: str,
    llm_categories: list[str] | None = None,
    out_of_domain_answer: str | None = None,
) -> dict:
    del user_message  # Guardrail categories come from the semantic understanding layer, not keyword matching.
    categories = list(
        dict.fromkeys(
            category
            for category in (llm_categories or [])
            if category in GUARDRAIL_CATEGORIES
        )
    )
    guardrail_triggered = next(
        (
            category
            for category in ["out_of_domain_request", "personalized_financial_advice_boundary"]
            if category in categories
        ),
        None,
    )
    return {
        "mode": "educational_only" if categories else "standard",
        "categories": categories,
        "guardrail_triggered": guardrail_triggered,
        "out_of_domain_answer": out_of_domain_answer,
    }


def guardrail_message(guardrail_result: dict) -> str:
    categories = guardrail_result.get("categories", [])
    if "out_of_domain_request" in categories:
        helpful_answer = str(guardrail_result.get("out_of_domain_answer") or "").strip()
        opening = f"{helpful_answer}\n\n" if helpful_answer else ""
        return (
            opening +
            "I'm primarily here to help with financial goals, budgeting, saving, investing education, retirement "
            "planning, debt, emergency funds, college savings, high-level tax education, and Acorns-related financial "
            "questions. Is there a financial topic I can help with today?"
        )
    if "personalized_financial_advice_boundary" in categories:
        return (
            "I can explain the relevant considerations and tradeoffs in general terms, but I cannot recommend a "
            "specific choice, percentage, or range for your personal situation. For guidance tailored to your "
            "circumstances, consider consulting an appropriate qualified professional."
        )
    if guardrail_result.get("mode") != "standard":
        return (
            "I can explain the risks and tradeoffs in general terms, but I cannot tell you what is right for your "
            "personal situation. For guidance tailored to your circumstances, consider consulting an appropriate "
            "qualified professional."
        )
    return ""
