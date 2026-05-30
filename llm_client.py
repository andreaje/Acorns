import re

from guardrails import evaluate_guardrails, guardrail_message


REFLECTIONS = {
    "emotional_concern": "I hear that this feels uncertain, and it makes sense to slow down and make the risk easier to understand.",
    "decision_support": "You are weighing a financial decision and want to understand the tradeoffs before acting.",
    "product_exploration": "You are exploring a financial product and want a clearer sense of how it fits into the bigger picture.",
    "learn_before_investing": "You are interested in investing, but you want to understand the product before putting money into it.",
}


def asks_for_speed(user_message: str) -> bool:
    return bool(
        re.search(
            r"\b(?:fast\w*|quick\w*|rapid\w*|soon|schnell\w*|rasch\w*|r(?:á|a)pid\w*|pronto)\b",
            user_message.lower(),
        )
    )


def format_tool_results(tool_results: dict) -> str:
    parts = []
    if "market_price" in tool_results:
        result = tool_results["market_price"]
        parts.append(f"For this demo, the mocked market price for {result['asset']} is ${result['value']:,}.")
    if "interest_rate" in tool_results:
        result = tool_results["interest_rate"]
        parts.append(f"For this demo, the mocked interest rate for {result['product']} is {result['value']}.")
    return " ".join(parts)


def address_question_or_concern(user_message: str, understanding: dict) -> str:
    concern = understanding.get("concern_type")
    goal = understanding.get("current_goal")
    parent_topic = understanding.get("parent_topic")
    tokens = set(re.findall(r"[a-z0-9']+", user_message.lower()))

    if understanding.get("primary_topic") == "bitcoin" and {"safe", "risky", "risk"} & tokens:
        return (
            "Bitcoin is not guaranteed to be safe: its price can move sharply, and it can lose value. It is worth "
            "understanding that volatility and protecting money you may need soon before considering any investment."
        )
    if concern == "starting too late":
        return (
            "Starting later than you hoped does not mean it is too late. A useful first step is to look at your "
            "timeline, what you have already saved, and an amount you could contribute consistently."
        )
    if concern in ["losing money", "losing money again"]:
        return (
            "That concern is worth taking seriously. You can reduce avoidable risk by protecting emergency savings, "
            "using diversification, and choosing a level of investment risk you can live with over time."
        )
    if concern == "not saving enough":
        return (
            "It is understandable to worry about whether you are saving enough. You do not need a perfect number "
            "before you begin; a realistic contribution and a plan to revisit it can create momentum."
        )
    if concern == "affordability uncertainty":
        if understanding.get("current_goal") == "help pay for a child's education":
            return (
                "Paying for college can feel like a large, uncertain target. You do not need to solve the full cost "
                "today; the useful first step is to understand the time available and what contribution could fit "
                "alongside your other priorities."
            )
        return (
            "It makes sense to pause on affordability before choosing a next step. A useful starting point is to "
            "separate the total cost from what would fit realistically into your monthly budget."
        )
    if concern == "falling into debt":
        return (
            "Wanting to stay comfortable and avoid debt is a concrete goal. A useful starting point is to protect "
            "some monthly breathing room, build a small emergency cushion, and make a plan for any high-interest "
            "debt before taking on new financial commitments."
        )
    if concern == "not understanding investing":
        return (
            "You do not need to understand every investing term before taking a next step. It helps to begin with "
            "risk, diversification, time horizon, and the difference between saving and investing."
        )
    if concern == "general financial uncertainty" and (
        parent_topic == "retirement planning"
        or understanding.get("primary_topic") == "retirement planning"
        or goal == "retirement security"
    ):
        return (
            "You do not need a complete retirement plan before you begin. A simple first pass at your timeline and "
            "current savings is enough to make the next step more concrete."
        )
    if goal == "comfortable retirement":
        return (
            "That is a clear and human goal. A comfortable retirement becomes easier to plan for when you translate "
            "it into a rough timeline, expected everyday expenses, and a sustainable saving habit."
        )
    if goal == "financial comfort and stability":
        return (
            "That is a useful goal. Financial comfort often starts with a little monthly breathing room and a buffer "
            "for unexpected expenses, then grows from there."
        )
    if understanding.get("goal_category") == "wealth_building":
        if asks_for_speed(user_message):
            return (
                "There is no reliable shortcut to building wealth quickly. The strongest general levers are increasing "
                "income, consistently keeping part of what you earn, avoiding high-interest debt, and investing for "
                "the long term in a diversified way. Promises of fast returns usually mean taking a real risk of loss."
            )
        return (
            "Got it: building wealth is the goal. That is broader than a single product or tactic, so the useful "
            "starting point is to choose the lever that can make the biggest practical difference first."
        )
    if understanding.get("stated_goal"):
        return (
            f"You said your goal is to {understanding['stated_goal']}. Let us make that concrete enough to identify "
            "a useful first step."
        )
    if understanding.get("asked_question") and not understanding.get("primary_topic") == "general finances":
        return "It makes sense to pause and get a clear answer before deciding what to do next."
    return ""


def plan_guidance(dialogue_plan: dict, understanding: dict) -> str:
    if dialogue_plan.get("dialogue_act") != "summarize":
        return ""
    goal = understanding.get("current_goal")
    if goal and goal not in ["Not yet detected", "general financial progress"]:
        return (
            f"So far, the useful anchor is your goal: {goal}. A practical next step is to turn that into one "
            "small, concrete action before adding more questions."
        )
    return (
        "A useful place to pause is to choose one priority, such as building emergency savings, preparing for "
        "retirement, or learning about investing. That gives the next step a clearer purpose."
    )


def captured_information_response(dialogue_plan: dict) -> str:
    slots = dialogue_plan.get("captured_information", {})
    if "retirement_timeline_years" in slots:
        years = slots["retirement_timeline_years"]
        return (
            f"About {years} years gives us a useful planning horizon. That does not answer every retirement question "
            "at once, but it is enough to move from worry toward a realistic plan."
        )
    if "education_timeline_years" in slots:
        years = slots["education_timeline_years"]
        return (
            f"About {years} years gives you a planning horizon for the college goal. The next useful step is to see "
            "what is already set aside and what a manageable contribution could look like."
        )
    if slots.get("education_savings_status") == "starting from scratch":
        return (
            "Starting from scratch is okay. The helpful move is to choose a contribution that fits your budget now, "
            "then revisit it as your circumstances and the college timeline change."
        )
    if slots.get("education_savings_status") == "has some savings":
        return (
            "Having something set aside already gives you a base to build from. The next step is choosing a monthly "
            "amount that supports the goal without squeezing your other priorities."
        )
    if "monthly_education_savings" in slots:
        amount = slots["monthly_education_savings"]
        return (
            f"${amount:,} per month gives you a concrete starting point for the college goal. You can treat it as a "
            "working amount and revisit it over time rather than trying to solve the full cost today."
        )
    if slots.get("current_debt_status") == "no current debt":
        return (
            "Being debt-free today is a strong starting point. The next useful check is whether an unexpected expense "
            "could push you into debt, because a small emergency cushion can protect that progress."
        )
    if slots.get("emergency_savings_status") == "needs an emergency cushion":
        return (
            "Then a starter emergency cushion is a sensible first move. It does not need to be fully funded at once; "
            "the useful next step is choosing an amount you can set aside consistently without creating new strain."
        )
    if "monthly_starter_savings" in slots:
        amount = slots["monthly_starter_savings"]
        return (
            f"${amount:,} per month is a concrete starting point. Consistency matters more than making the first "
            "contribution perfect, and you can revisit the amount as your cushion grows."
        )
    return ""


def repeated_turn_guidance(understanding: dict) -> str:
    if understanding.get("concern_type") == "falling into debt":
        return (
            "Let us make that practical. Start with one small snapshot: monthly take-home income, essential expenses, "
            "minimum debt payments, and the amount left over. That shows whether the first move is creating breathing "
            "room, building a starter emergency cushion, or tackling high-interest debt."
        )
    return (
        "Let us turn that into one practical move. Start with the smallest action that would make the situation feel "
        "more concrete, then use what you learn to choose the next step."
    )


def correction_response(understanding: dict) -> str:
    correction = understanding["user_correction"]
    corrected_away_from = correction["corrected_away_from"]
    goal = understanding.get("current_goal")
    if goal and goal not in ["Not yet detected", "general financial progress"]:
        return (
            f"You are right. I pulled this toward {corrected_away_from}, but the goal you named is to {goal}. "
            "Let us stay with that."
        )
    return f"You are right. I pulled this toward {corrected_away_from}. Let us stay with the concern you actually raised."


def structured_goal_fallback(
    understanding: dict,
    dialogue_plan: dict,
    retrieved_knowledge: dict,
    guardrail_result: dict,
) -> str:
    intent = understanding.get("intent")
    goal = understanding.get("current_goal")
    topic = understanding.get("topic_label") or understanding.get("primary_topic")
    parts = []

    if guardrail_result["mode"] == "educational_only":
        parts.append(guardrail_message(guardrail_result))

    if intent == "budgeting_guidance":
        parts.append(
            "It sounds like travel is important to you, but you want to understand what you can spend without putting "
            "your broader financial security at risk. A useful way to think about this is to separate essentials, "
            "savings or debt goals, and flexible spending."
        )
    elif intent == "retirement_goal_planning":
        parts.append(
            "Being debt-free in retirement is a clear planning goal. A useful first step is to understand any debt "
            "you have now, your retirement timeline, and the monthly room available to reduce balances without "
            "crowding out other essentials."
        )
    elif intent == "college_affordability_planning":
        parts.append(
            "College costs can feel like a large target, but you do not need to solve the full amount today. A useful "
            "starting point is the time available, what is already set aside, and a contribution that can fit "
            "alongside your other priorities."
        )
    else:
        readable_goal = goal if goal not in [None, "Not yet detected", "general financial progress"] else topic
        parts.append(
            f"It sounds like {readable_goal} is the priority. A useful next step is to make the goal concrete enough "
            "to compare it with your essentials and other commitments."
        )

    if retrieved_knowledge:
        parts.append(retrieved_knowledge["content"])
    parts.append(dialogue_plan.get("follow_up_question"))
    return "\n\n".join(part for part in parts if part)


def _generate_template_response(
    user_message: str,
    understanding: dict,
    retrieved_knowledge: dict,
    tool_results: dict,
    conversation_context: dict,
    dialogue_plan: dict,
    guardrail_result: dict,
) -> str:
    parts = []

    if dialogue_plan.get("dialogue_act") == "clarify_reference":
        return "\n\n".join([
            "I want to make sure I understand what you mean before answering.",
            dialogue_plan["follow_up_question"],
        ])
    if understanding.get("user_correction"):
        return "\n\n".join(part for part in [
            correction_response(understanding),
            dialogue_plan.get("follow_up_question"),
        ] if part)
    if understanding.get("intent") == "conversation_frustration":
        return "\n\n".join(part for part in [
            "You are right. Let us keep this concrete.",
            dialogue_plan.get("follow_up_question"),
        ] if part)
    if dialogue_plan.get("avoid_repetition"):
        return "\n\n".join(part for part in [
            repeated_turn_guidance(understanding),
            dialogue_plan.get("follow_up_question"),
        ] if part)

    if guardrail_result["mode"] == "educational_only":
        parts.append(guardrail_message(guardrail_result))

    if understanding["intent"] == "follow_up_question" and understanding["resolved_reference"] != "None":
        parts.append(f"You are asking about {understanding['resolved_reference']}.")
    elif understanding["intent"] not in ["educational_query", "emotional_concern"]:
        reflection = REFLECTIONS.get(understanding["intent"])
        if reflection:
            parts.append(reflection)

    slot_response = captured_information_response(dialogue_plan)
    if slot_response:
        parts.append(slot_response)

    concern_response = address_question_or_concern(user_message, understanding)
    if concern_response and not slot_response and (
        understanding.get("expressed_concern")
        or not retrieved_knowledge
        or "safe" in re.findall(r"[a-z0-9']+", user_message.lower())
    ):
        parts.append(concern_response)

    if retrieved_knowledge and not (understanding["intent"] == "follow_up_question" and concern_response):
        parts.append(retrieved_knowledge["content"])

    tool_text = format_tool_results(tool_results)
    if tool_text:
        parts.append(tool_text)

    if understanding["intent"] == "decision_support" and not retrieved_knowledge:
        parts.append("A useful next step is to compare the choice with your goal, budget, timeline, and other commitments.")

    parts.append(plan_guidance(dialogue_plan, understanding))
    parts.append(dialogue_plan.get("follow_up_question"))
    return "\n\n".join(part for part in parts if part)


def generate_response_details(
    user_message: str,
    understanding: dict,
    retrieved_knowledge: dict,
    tool_results: dict,
    conversation_context: dict,
    dialogue_plan: dict,
    guardrail_result: dict | None = None,
) -> dict:
    guardrail_result = guardrail_result or evaluate_guardrails(user_message)
    structured_intents = {"budgeting_guidance", "retirement_goal_planning", "college_affordability_planning"}
    use_structured_fallback = understanding.get("intent") in structured_intents

    if use_structured_fallback:
        text = structured_goal_fallback(understanding, dialogue_plan, retrieved_knowledge, guardrail_result)
        response_source = "structured_fallback"
    else:
        text = _generate_template_response(
            user_message,
            understanding,
            retrieved_knowledge,
            tool_results,
            conversation_context,
            dialogue_plan,
            guardrail_result,
        )
        response_source = "template"

    return {
        "text": text,
        "response_source": response_source,
        "retrieved_knowledge_used": retrieved_knowledge.get("topic") if retrieved_knowledge else None,
    }


def generate_response(
    user_message: str,
    understanding: dict,
    retrieved_knowledge: dict,
    tool_results: dict,
    conversation_context: dict,
    dialogue_plan: dict,
    guardrail_result: dict | None = None,
) -> str:
    return generate_response_details(
        user_message,
        understanding,
        retrieved_knowledge,
        tool_results,
        conversation_context,
        dialogue_plan,
        guardrail_result,
    )["text"]
