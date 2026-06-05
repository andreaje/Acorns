import unittest

from conversation_state import analyze_message, create_conversation_context
from dialogue_manager import decide_dialogue_plan
from guardrails import evaluate_guardrails
from llm_client import generate_llm_response, generate_response


class OutOfDomainGuardrailTests(unittest.TestCase):
    def assert_out_of_domain(self, message: str):
        context = create_conversation_context()
        understanding = analyze_message(message, context)
        guardrail_result = evaluate_guardrails(
            message,
            ["out_of_domain_request"],
            "This is a helpful answer to the non-financial question. It can include enough context to be useful.",
        )
        understanding["guardrail_categories"] = guardrail_result["categories"]
        understanding["guardrail_triggered"] = guardrail_result["guardrail_triggered"]
        user_model = {
            key: understanding.get(key)
            for key in [
                "current_goal",
                "current_fear",
                "confidence_level",
                "financial_literacy",
                "knowledge_level",
                "coaching_style",
                "persona",
                "risk_level",
            ]
        }
        dialogue_plan = decide_dialogue_plan(understanding, user_model, message, context)
        response = generate_response(message, understanding, {}, {}, context, dialogue_plan, guardrail_result)

        self.assertEqual("out_of_domain_request", guardrail_result["guardrail_triggered"])
        self.assertEqual("brief_boundary_and_redirect", dialogue_plan["dialogue_act"])
        self.assertIsNone(dialogue_plan["follow_up_question"])
        self.assertIn("financial", response.lower())

    def test_clear_out_of_domain_examples_trigger_guardrail(self):
        for message in [
            "How do I create a sourdough starter?",
            "How does one make sourdough bread?",
            "How do I change a car battery?",
            "Should I remove my own stitches?",
            "Can you help me debug Python?",
            "Where should I travel in Spain?",
        ]:
            with self.subTest(message=message):
                self.assert_out_of_domain(message)

    def test_guardrails_do_not_use_local_keyword_matching(self):
        result = evaluate_guardrails("Explain how to make sourdough starter.")

        self.assertIsNone(result["guardrail_triggered"])

    def test_financially_framed_travel_stays_standard_without_llm_category(self):
        result = evaluate_guardrails("How much should I budget for a trip to Spain?")

        self.assertIsNone(result["guardrail_triggered"])

    def test_out_of_domain_response_preserves_helpful_answer(self):
        message = "How does one make sourdough bread?"
        context = create_conversation_context()
        understanding = analyze_message(message, context)
        guardrail_result = evaluate_guardrails(
            message,
            ["out_of_domain_request"],
            (
                "Sourdough bread is made with a fermented starter that helps leaven the dough. The starter contains "
                "wild yeast and bacteria that create flavor and lift. In general, bakers mix starter into dough, let "
                "it ferment, shape it, proof it, and bake it."
            ),
        )
        understanding["guardrail_categories"] = guardrail_result["categories"]
        understanding["guardrail_triggered"] = guardrail_result["guardrail_triggered"]
        dialogue_plan = decide_dialogue_plan(understanding, {}, message, context)
        response = generate_response(message, understanding, {}, {}, context, dialogue_plan, guardrail_result)

        self.assertIn("fermented", response.lower())
        self.assertIn("starter", response.lower())
        self.assertIn("proof", response.lower())
        self.assertIn("financial", response.lower())

    def test_out_of_domain_llm_response_uses_deterministic_boundary(self):
        message = "Explain how to make sourdough starter."
        context = create_conversation_context()
        understanding = analyze_message(message, context)
        guardrail_result = evaluate_guardrails(
            message,
            ["out_of_domain_request"],
            (
                "A sourdough starter is a fermented mixture of flour and water used to cultivate natural yeast. "
                "It is maintained over time and used to help bread rise while adding tangy flavor."
            ),
        )
        understanding["guardrail_categories"] = guardrail_result["categories"]
        understanding["guardrail_triggered"] = guardrail_result["guardrail_triggered"]
        dialogue_plan = decide_dialogue_plan(understanding, {}, message, context)
        details = generate_llm_response(
            message,
            context,
            dialogue_plan,
            {},
            guardrail_result,
            "en",
            understanding=understanding,
            api_key="test-api-key",
        )

        self.assertFalse(details["openai_api_call_attempted"])
        self.assertIn("fermented mixture of flour and water", details["response_text"])
        self.assertIn("financial", details["response_text"].lower())


if __name__ == "__main__":
    unittest.main()
