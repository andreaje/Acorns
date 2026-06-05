import re

from guardrails import evaluate_guardrails


FOLLOW_UPS = {
    "cd": "Are you exploring CDs as a place to keep savings, or comparing them with investing options?",
    "bitcoin": "Would it help to talk through how volatility, time horizon, and diversification affect that decision?",
    "etf": "Are you learning about ETFs for a first investment, or comparing them with another option?",
    "roth_ira": "Are you trying to understand how a Roth IRA works, or whether opening one belongs on your next-step list?",
    "traditional_ira": "Are you comparing a Traditional IRA with another retirement account?",
    "emergency_fund": "What kind of unexpected expense would you most want that fund to cover?",
    "diversification": "Would an example of a more diversified approach make this clearer?",
    "compound_interest": "Would it help to walk through a small numerical example?",
}

EXPLORATORY_ACTS = {"explore_goal", "gather_information"}
REFERENCE_WORDS = {"it", "that", "this", "they", "them"}


def normalize_question(question: str | None) -> str:
    return re.sub(r"\s+", " ", (question or "").strip().lower())


def normalize_message(message: str | None) -> str:
    return re.sub(r"\s+", " ", (message or "").strip().lower())


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def needs_reference_clarification(understanding: dict, latest_user_message: str) -> bool:
    return (
        bool(understanding.get("asked_question"))
        and bool(tokens(latest_user_message) & REFERENCE_WORDS)
        and understanding.get("resolved_reference") == "None"
    )


def choose_follow_up(understanding: dict, user_model: dict) -> tuple[str | None, str | None]:
    topic = understanding.get("primary_topic")
    concern = understanding.get("concern_type")
    goal = user_model.get("current_goal")
    stated_goal = understanding.get("stated_goal")
    goal_category = understanding.get("goal_category")
    slots = understanding.get("planning_slots", {})

    if understanding.get("intent") == "budgeting_guidance":
        return "travel budget context", "What would help most: estimating a safe monthly travel budget, planning for one specific trip, or balancing travel against other goals?"
    if understanding.get("intent") == "retirement_goal_planning":
        return "retirement debt context", "Do you currently have debt you want to pay down before retirement, or are you trying to avoid carrying new debt into retirement?"
    if "family_education_goal" in slots and "education_timeline_years" not in slots:
        return "education timeline", "About how many years are there before your child may start college?"
    if "education_timeline_years" in slots and "education_savings_status" not in slots:
        return "current education savings", "Have you already set aside anything for that goal, or would you be starting from scratch?"
    if "education_savings_status" in slots and "monthly_education_savings" not in slots:
        return "manageable education savings amount", "What monthly amount would feel realistic without crowding out your other priorities?"
    if "monthly_education_savings" in slots:
        return "education next-step confirmation", "Would you like to use that as a starting point and revisit it as the college timeline gets closer?"
    if concern == "starting too late" and "retirement_timeline_years" not in slots:
        return "retirement timeline", "Roughly how many years do you have until you would like to retire?"
    if concern == "falling into debt" and "debt_goal" not in slots:
        return "whether the user wants to prevent or reduce debt", "Is your main concern avoiding new debt, paying down debt you already have, or both?"
    if concern == "falling into debt" and "current_debt_status" not in slots:
        return "current debt situation", "Do you currently have debt you want to work down, or is the priority staying debt-free?"
    if slots.get("current_debt_status") == "no current debt" and "emergency_savings_status" not in slots:
        return "emergency savings status", "Do you already have some savings set aside for unexpected expenses?"
    if slots.get("emergency_savings_status") == "needs an emergency cushion" and "monthly_starter_savings" not in slots:
        return "manageable starter savings amount", "What amount could you comfortably set aside each month to start that cushion?"
    if "monthly_starter_savings" in slots:
        return "next-step confirmation", "Would you like to make that your first step and revisit retirement contributions after the cushion is underway?"
    if concern == "general financial uncertainty" and (
        understanding.get("parent_topic") == "retirement planning"
        or topic == "retirement planning"
        or goal == "retirement security"
    ):
        return "preferred retirement starting point", "Would it be easier to start with your timeline or with a manageable monthly saving target?"
    if goal == "comfortable retirement" and "desired_outcome" not in slots:
        return "meaning of a comfortable retirement", "When you picture comfortable, is the priority covering everyday expenses, having room for extras, or both?"
    if "retirement_timeline_years" in slots and goal == "retirement security":
        return "meaning of a comfortable retirement", "When you picture a comfortable retirement, what matters most: covering essentials, having room for extras, or both?"
    if goal == "financial comfort and stability":
        return "first priority for financial stability", "Would it help to start with monthly breathing room, an emergency cushion, or a plan for existing debt?"
    if goal_category == "wealth_building":
        return "wealth-building baseline", "Do you currently have income left after monthly expenses that you could consistently save or invest?"
    if stated_goal:
        return "goal baseline", "What would meaningful progress look like over the next year?"
    if topic in FOLLOW_UPS:
        return f"how {topic} fits the user's goal", FOLLOW_UPS[topic]
    if understanding.get("intent") == "emotional_concern":
        return "most important source of uncertainty", "What part of this feels most important to make less uncertain first?"
    if understanding.get("intent") == "goal_discovery":
        return "desired financial outcome", "What outcome would make the biggest difference for you right now?"
    return "remaining uncertainty", "What would you like to make clearer next?"


def decide_dialogue_plan(
    understanding: dict,
    user_model: dict,
    latest_user_message: str,
    conversation_context: dict | None = None,
) -> dict:
    context = conversation_context or {}
    guardrail_categories = understanding.get("guardrail_categories", [])
    guardrail_triggered = understanding.get("guardrail_triggered")
    if (
        guardrail_triggered == "out_of_domain_request"
        or "out_of_domain_request" in guardrail_categories
        or evaluate_guardrails(latest_user_message, guardrail_categories)["guardrail_triggered"] == "out_of_domain_request"
    ):
        return {
            "dialogue_act": "brief_boundary_and_redirect",
            "response_goal": "Briefly set the financial-coaching boundary and redirect back to financial coaching.",
            "must_address": ["out-of-domain boundary", "financial coaching redirect"],
            "next_information_needed": None,
            "follow_up_question": None,
            "avoid_repetition": False,
            "captured_information": understanding.get("extracted_slots", {}),
        }
    asked_question = bool(understanding.get("asked_question"))
    expressed_concern = bool(understanding.get("expressed_concern"))
    intent = understanding.get("intent")
    exploratory_streak = int(context.get("exploratory_questions_in_a_row", 0))
    repeated_user_message = normalize_message(latest_user_message) == normalize_message(context.get("last_analyzed_text"))
    captured_information = bool(understanding.get("extracted_slots"))

    if intent == "budgeting_guidance":
        dialogue_act = "answer_and_gather_budget_context"
        response_goal = "Frame the life goal as a budgeting question and gather the most useful spending context."
        must_address = ["travel goal", "affordability"]
        next_information_needed, follow_up_question = choose_follow_up(understanding, user_model)
    elif intent == "retirement_goal_planning":
        dialogue_act = "reflect_goal_and_gather_debt_context"
        response_goal = "Reflect the debt-free retirement goal and gather debt context."
        must_address = ["retirement goal", "debt context"]
        next_information_needed, follow_up_question = choose_follow_up(understanding, user_model)
    elif intent == "college_affordability_planning":
        dialogue_act = "answer_and_gather_college_context"
        response_goal = "Address college affordability and gather the timeline needed for planning."
        must_address = ["college affordability", "education timeline"]
        next_information_needed, follow_up_question = choose_follow_up(understanding, user_model)
    elif intent == "wealth_building_guidance":
        dialogue_act = "answer_and_gather_wealth_context"
        response_goal = "Address the wealth-building goal directly and gather the most useful starting point."
        must_address = ["wealth-building goal", "realistic path", "avoid get-rich-quick framing"]
        next_information_needed, follow_up_question = choose_follow_up(understanding, user_model)
    elif intent == "goal_statement":
        dialogue_act = "reflect_goal_and_gather_context"
        response_goal = "Reflect the user's stated goal and gather the first useful planning detail."
        must_address = ["stated goal", "next planning detail"]
        next_information_needed, follow_up_question = choose_follow_up(understanding, user_model)
    elif understanding.get("user_correction"):
        dialogue_act = "summarize"
        response_goal = "Acknowledge the correction, drop the mistaken frame, and return to the user's actual goal."
        must_address = ["user correction", "actual goal"]
        next_information_needed, follow_up_question = choose_follow_up(understanding, user_model)
    elif needs_reference_clarification(understanding, latest_user_message):
        dialogue_act = "clarify_reference"
        response_goal = "Clarify the user's reference before giving guidance."
        must_address = ["ambiguous reference"]
        next_information_needed = "the product, goal, or concept the user means"
        follow_up_question = "What does that refer to?"
    elif asked_question:
        dialogue_act = "answer_question"
        response_goal = "Answer the user's question directly before offering any next step."
        must_address = ["direct question"]
        if expressed_concern:
            must_address.append("expressed concern")
        next_information_needed, follow_up_question = choose_follow_up(understanding, user_model)
    elif expressed_concern:
        dialogue_act = "reassure"
        response_goal = "Acknowledge the concern and reduce uncertainty with a concrete, supportive perspective."
        must_address = ["expressed concern"]
        next_information_needed, follow_up_question = choose_follow_up(understanding, user_model)
    elif captured_information:
        dialogue_act = "gather_information"
        response_goal = "Use the information the user just provided and gather the next missing planning detail."
        must_address = ["newly provided information"]
        next_information_needed, follow_up_question = choose_follow_up(understanding, user_model)
    elif intent in ["educational_query", "learn_before_investing", "product_exploration"]:
        dialogue_act = "educate"
        response_goal = "Explain the relevant concept in plain language and connect it to the user's goal."
        must_address = ["knowledge gap"]
        next_information_needed, follow_up_question = choose_follow_up(understanding, user_model)
    elif user_model.get("current_goal") in [None, "Not yet detected", "general financial progress"]:
        dialogue_act = "explore_goal"
        response_goal = "Help the user identify the outcome that matters most."
        must_address = ["unclear goal"]
        next_information_needed, follow_up_question = choose_follow_up(understanding, user_model)
    elif intent == "decision_support":
        dialogue_act = "suggest_next_step"
        response_goal = "Offer a practical way to compare tradeoffs without making the decision for the user."
        must_address = ["decision tradeoffs"]
        next_information_needed, follow_up_question = choose_follow_up(understanding, user_model)
    else:
        dialogue_act = "gather_information"
        response_goal = "Gather one useful detail that moves the user toward a concrete next step."
        must_address = ["next planning detail"]
        next_information_needed, follow_up_question = choose_follow_up(understanding, user_model)

    if repeated_user_message and dialogue_act != "clarify_reference":
        dialogue_act = "suggest_next_step"
        response_goal = "Offer a concrete next step instead of restating the previous response."
        must_address = ["practical next step"]
        if understanding.get("goal_category") == "wealth_building":
            next_information_needed = "wealth-building baseline"
            follow_up_question = "Do you currently have income left after monthly expenses that you could consistently save or invest?"
        else:
            follow_up_question = "Would it help to focus on one practical next step together?"

    previous_question = normalize_question(context.get("last_assistant_question"))
    if normalize_question(follow_up_question) == previous_question:
        if next_information_needed == "education timeline":
            next_information_needed = "child age"
            follow_up_question = "How old is your child now?"
        elif understanding.get("goal_category") == "wealth_building":
            next_information_needed = "most useful wealth-building lever"
            follow_up_question = "Which feels most realistic to work on first: earning more, saving more of your current income, or learning about long-term investing?"
        else:
            follow_up_question = "Would it help to focus on one practical next step together?"
    if normalize_question(follow_up_question) == previous_question:
        follow_up_question = None
    if dialogue_act in EXPLORATORY_ACTS and exploratory_streak >= 2 and not captured_information:
        dialogue_act = "summarize"
        response_goal = "Summarize what is known and provide guidance before asking for more information."
        must_address = ["conversation progress", "useful guidance"]
        follow_up_question = None

    return {
        "dialogue_act": dialogue_act,
        "response_goal": response_goal,
        "must_address": must_address,
        "next_information_needed": next_information_needed,
        "follow_up_question": follow_up_question,
        "avoid_repetition": repeated_user_message,
        "captured_information": understanding.get("extracted_slots", {}),
    }


def record_dialogue_plan(context: dict, dialogue_plan: dict) -> dict:
    if dialogue_plan["dialogue_act"] in EXPLORATORY_ACTS and dialogue_plan.get("follow_up_question"):
        context["exploratory_questions_in_a_row"] = int(context.get("exploratory_questions_in_a_row", 0)) + 1
    else:
        context["exploratory_questions_in_a_row"] = 0
    context["last_dialogue_act"] = dialogue_plan["dialogue_act"]
    context["pending_information"] = dialogue_plan.get("next_information_needed") if dialogue_plan.get("follow_up_question") else None
    return context
