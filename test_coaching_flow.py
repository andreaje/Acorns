import unittest

from conversation_state import (
    analyze_message,
    classify_goal_category,
    create_conversation_context,
    extract_stated_goal,
    record_assistant_response,
    update_conversation_context,
)
from dialogue_manager import decide_dialogue_plan, record_dialogue_plan
from guardrails import evaluate_guardrails
from i18n import localize_coach_response
from knowledge_base import retrieve_knowledge
from llm_client import generate_response_details
from tools import collect_tool_results


def run_turn(prompt: str, context: dict) -> tuple[dict, dict, str]:
    understanding = analyze_message(prompt, context)
    user_model = {
        key: understanding.get(key)
        for key in [
            "current_goal",
            "current_fear",
            "confidence_level",
            "financial_literacy",
            "coaching_style",
            "persona",
            "risk_level",
        ]
    }
    dialogue_plan = decide_dialogue_plan(understanding, user_model, prompt, context)
    response = generate_response_details(
        prompt,
        understanding,
        retrieve_knowledge(understanding["primary_topic"]),
        collect_tool_results(prompt, understanding["primary_topic"]),
        context,
        dialogue_plan,
        evaluate_guardrails(prompt),
    )["text"]
    update_conversation_context(context, prompt, understanding)
    record_assistant_response(context, response)
    record_dialogue_plan(context, dialogue_plan)
    return understanding, dialogue_plan, response


class GoalExtractionTests(unittest.TestCase):
    def test_extracts_common_goal_statement_shapes(self):
        examples = {
            "I want to buy a home.": "buy a home",
            "I'd like to pay off my credit cards.": "pay off my credit cards",
            "My goal is financial freedom.": "financial freedom",
            "We hope to save for a wedding!": "save for a wedding",
        }
        for prompt, expected_goal in examples.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(extract_stated_goal(prompt), expected_goal)

    def test_groups_equivalent_wealth_building_phrases(self):
        for prompt in [
            "I want to be rich.",
            "My goal is financial independence.",
            "How do I grow my net worth?",
            "I would like to build wealth.",
        ]:
            with self.subTest(prompt=prompt):
                self.assertEqual(classify_goal_category(prompt), "wealth_building")


class CoachingFlowTests(unittest.TestCase):
    def test_reflects_unknown_goal_instead_of_asking_for_goal_again(self):
        understanding, _, response = run_turn("I'd like to save for a sabbatical.", create_conversation_context())

        self.assertEqual(understanding["intent"], "goal_statement")
        self.assertEqual(understanding["stated_goal"], "save for a sabbatical")
        self.assertIn("your goal is to save for a sabbatical", response)
        self.assertNotIn("What outcome would make the biggest difference", response)

    def test_answers_fast_wealth_question_for_equivalent_phrasing(self):
        _, _, response = run_turn("What is the quickest path to financial independence?", create_conversation_context())

        self.assertIn("no reliable shortcut", response.lower())
        self.assertIn("increasing income", response.lower())
        self.assertNotIn("What outcome would make the biggest difference", response)

    def test_retains_goal_after_conversation_friction(self):
        context = create_conversation_context()
        run_turn("My goal is financial freedom.", context)
        understanding, _, response = run_turn("Why are you asking me that again?", context)

        self.assertEqual(understanding["intent"], "conversation_frustration")
        self.assertEqual(understanding["goal_category"], "wealth_building")
        self.assertIn("You are right. Let us keep this concrete.", response)
        self.assertNotIn("What outcome would make the biggest difference", response)

    def test_preserves_users_goal_wording_across_related_question(self):
        context = create_conversation_context()
        run_turn("My goal is financial freedom.", context)
        understanding, _, _ = run_turn("What is the quickest way to grow my net worth?", context)

        self.assertEqual(understanding["current_goal"], "financial freedom")
        self.assertEqual(understanding["stated_goal"], "financial freedom")

    def test_localizes_german_wealth_building_response(self):
        understanding, _, response = run_turn("Ich will reich werden.", create_conversation_context())
        localized_response = localize_coach_response(response, "de")

        self.assertEqual(understanding["goal_category"], "wealth_building")
        self.assertIn("Vermögensaufbau ist das Ziel", localized_response)
        self.assertIn("monatlichen Ausgaben", localized_response)
        self.assertNotIn("Got it:", localized_response)

    def test_localizes_spanish_fast_wealth_response(self):
        understanding, _, response = run_turn(
            "¿Cuál es la forma más rápida de lograr independencia financiera?",
            create_conversation_context(),
        )
        localized_response = localize_coach_response(response, "es")

        self.assertEqual(understanding["goal_category"], "wealth_building")
        self.assertIn("No existe un atajo fiable", localized_response)
        self.assertIn("riesgo real de pérdida", localized_response)
        self.assertNotIn("There is no reliable shortcut", localized_response)


if __name__ == "__main__":
    unittest.main()
