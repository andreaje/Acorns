import hmac
import importlib
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

import pandas as pd
import streamlit as st

from conversation_state import (
    TOPIC_RULES,
    analyze_message,
    apply_profile_update_operations,
    create_conversation_context,
    record_assistant_response,
    update_conversation_context,
)
from dialogue_manager import decide_dialogue_plan, record_dialogue_plan
from guardrails import evaluate_guardrails
from i18n import localize_coach_response, t
from knowledge_base import retrieve_knowledge
from llm_client import classify_knowledge_level, generate_llm_response, resolve_openai_api_key
import mock_data as mock_data_module
from tools import collect_tool_results

st.set_page_config(page_title="AJ's AI Coach", layout="wide")


def require_authentication():
    try:
        app_password = str(st.secrets["APP_PASSWORD"])
    except Exception:
        st.error("APP_PASSWORD is missing from Streamlit secrets. Add it before using the app.")
        st.stop()

    if st.session_state.get("authenticated"):
        return

    gate = st.empty()
    with gate.container():
        st.title("🌴 AJ's AI Coach")
        st.subheader("Demo access required")
        st.caption("Enter the prototype access code to continue.")
        show_access_code = st.toggle("Show access code", key="show_access_code")
        if not show_access_code:
            st.markdown(
                """
                <style>
                input[aria-label="Prototype Access Code"] {
                    -webkit-text-security: disc;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
        with st.form("prototype_access_form"):
            entered_access_code = st.text_input(
                "Prototype Access Code",
                autocomplete="one-time-code",
                key="prototype_access_code",
            )
            submitted = st.form_submit_button("Enter")
    if submitted:
        if hmac.compare_digest(entered_access_code, app_password):
            st.session_state.authenticated = True
            gate.empty()
            st.rerun()
        st.error("Incorrect access code.")
    st.stop()


require_authentication()

if "language" not in st.session_state:
    st.session_state.language = "en"

lang = st.session_state.language

ASSISTANT_AVATAR = "✨"
MOCK_DATA_SCHEMA_VERSION = 6

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": t("opening_message", lang),
            "message_id": "assistant-opening-message",
            "feedback_context": {
                "user_message_context": "",
                "detected_intent": None,
                "primary_topic": None,
                "persona": None,
                "risk_level": None,
                "language": lang,
            },
        }
    ]

# In production, this feedback would be logged to an analytics/event pipeline rather than Streamlit session state.
if "feedback_records" not in st.session_state:
    st.session_state.feedback_records = {}

fresh_context = create_conversation_context()
existing_context = st.session_state.get("conversation_context", {})
if existing_context.get("active_topic") and not existing_context.get("primary_topic"):
    existing_context["primary_topic"] = existing_context["active_topic"]
if existing_context.get("active_topic_label") and not existing_context.get("last_product_or_concept"):
    existing_context["last_product_or_concept"] = existing_context["active_topic_label"]
st.session_state.conversation_context = {**fresh_context, **existing_context}

if "last_understanding" not in st.session_state:
    st.session_state.last_understanding = {}

if "last_retrieved_knowledge" not in st.session_state:
    st.session_state.last_retrieved_knowledge = {}

if "last_tool_results" not in st.session_state:
    st.session_state.last_tool_results = {}

if "last_guardrail_result" not in st.session_state:
    st.session_state.last_guardrail_result = {"mode": "standard", "categories": []}

if "last_dialogue_plan" not in st.session_state:
    st.session_state.last_dialogue_plan = {}

if "last_response_debug" not in st.session_state:
    st.session_state.last_response_debug = {}


@st.cache_resource
def load_mock_data(schema_version: int):
    return mock_data_module.generate_mock_data()


@st.cache_data
def load_design_log(path: str, modified_ns: int) -> str:
    return Path(path).read_text(encoding="utf-8")


def render_architectural_design_log():
    design_log_path = Path(__file__).resolve().parent / "AJs_AI-Coach_Design_Log.md"
    if not design_log_path.is_file():
        st.error("Architectural design log not found.")
        return
    design_log = load_design_log(str(design_log_path), design_log_path.stat().st_mtime_ns)
    if not design_log.strip():
        st.info("The architectural design log is currently empty.")
        return
    st.markdown(design_log)


def reset_conversation():
    st.session_state.clear()
    st.rerun()


def update_opening_message():
    if len(st.session_state.get("messages", [])) == 1 and st.session_state.messages[0]["role"] == "assistant":
        st.session_state.messages[0]["content"] = t("opening_message", st.session_state.language)
        st.session_state.messages[0].setdefault("feedback_context", {})["language"] = st.session_state.language


def get_openai_key_debug(key_source: str) -> dict:
    try:
        secrets_contains_key = "OPENAI_API_KEY" in st.secrets
    except Exception:
        secrets_contains_key = False
    return {
        "cwd": str(Path.cwd()),
        ".streamlit/secrets.toml exists": Path(".streamlit/secrets.toml").is_file(),
        "st.secrets contains OPENAI_API_KEY": secrets_contains_key,
        "OPENAI_API_KEY source": key_source,
    }


def ensure_assistant_message_metadata():
    for message in st.session_state.messages:
        if message["role"] != "assistant":
            continue
        message.setdefault("message_id", f"assistant-{uuid4()}")
        message.setdefault(
            "feedback_context",
            {
                "user_message_context": "",
                "detected_intent": None,
                "primary_topic": None,
                "persona": None,
                "risk_level": None,
                "language": st.session_state.language,
            },
        )


def record_feedback(message: dict, feedback_value: str):
    context = message["feedback_context"]
    st.session_state.feedback_records[message["message_id"]] = {
        "message_id": message["message_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feedback_value": feedback_value,
        "user_message_context": context["user_message_context"],
        "assistant_response": message["content"],
        "detected_intent": context["detected_intent"],
        "primary_topic": context["primary_topic"],
        "persona": context["persona"],
        "risk_level": context["risk_level"],
        "language": context["language"],
    }


def render_feedback_controls(message: dict):
    message_id = message["message_id"]
    down_column, up_column, _ = st.columns([0.07, 0.07, 0.86])
    if down_column.button("👎", key=f"feedback-down-{message_id}"):
        record_feedback(message, "thumbs_down")
    if up_column.button("👍", key=f"feedback-up-{message_id}"):
        record_feedback(message, "thumbs_up")
    if message_id in st.session_state.feedback_records:
        st.caption(t("feedback_recorded", st.session_state.language))


def render_feedback_summary():
    records = list(st.session_state.feedback_records.values())
    thumbs_up_count = sum(record["feedback_value"] == "thumbs_up" for record in records)
    thumbs_down_count = sum(record["feedback_value"] == "thumbs_down" for record in records)
    assistant_response_count = sum(message["role"] == "assistant" for message in st.session_state.messages)
    feedback_rate = len(records) / assistant_response_count if assistant_response_count else 0
    thumbs_up_percentage = thumbs_up_count / len(records) if records else 0

    st.header("Coach Response Feedback")
    feedback_metrics = st.columns(4)
    feedback_metrics[0].metric("Thumbs Up", thumbs_up_count)
    feedback_metrics[1].metric("Thumbs Down", thumbs_down_count)
    feedback_metrics[2].metric("Feedback Rate", f"{feedback_rate:.1%}")
    feedback_metrics[3].metric("Thumbs Up Percentage", f"{thumbs_up_percentage:.1%}")


def multi_filter(label: str, values: pd.Series, key: str) -> list[str]:
    options = sorted(values.dropna().unique().tolist())
    return st.multiselect(label, options, default=options, key=key)


REQUIRED_USER_COLUMNS = {
    "user_id",
    "persona",
    "user_type",
    "age_band",
    "financial_literacy",
    "coaching_style",
    "risk_level",
    "funnel_stage",
    "activated",
    "inferred_satisfaction",
    "confidence_before",
    "confidence_after",
    "confidence_lift",
    "helpfulness",
    "trust",
    "confidence_building",
    "efficiency",
    "adaptability",
    "recurring_deposit_started",
    "signup_date",
}

REQUIRED_SESSION_COLUMNS = {
    "user_id",
    "persona",
    "topic_category",
    "recommended_action_taken",
    "escalated_to_human",
    "resolved_reference_success",
    "intent_accuracy",
    "slot_completion_accuracy",
    "context_retention",
    "dialogue_policy_accuracy",
    "retrieval_relevance",
    "groundedness",
    "perceived_understanding",
    "factual_accuracy",
    "safety_compliance",
    "integrity_policy_compliance",
    "session_date",
    "completed_session",
}


def validate_dashboard_schema(users: pd.DataFrame, sessions: pd.DataFrame, show_warning: bool = True) -> bool:
    missing_user_columns = sorted(REQUIRED_USER_COLUMNS - set(users.columns))
    missing_session_columns = sorted(REQUIRED_SESSION_COLUMNS - set(sessions.columns))

    if not missing_user_columns and not missing_session_columns:
        return True

    missing_details = []
    if missing_user_columns:
        missing_details.append(f"users DataFrame: {', '.join(missing_user_columns)}")
    if missing_session_columns:
        missing_details.append(f"sessions DataFrame: {', '.join(missing_session_columns)}")

    if show_warning:
        st.warning(
            "Analytics cannot render because the mock-data schema is missing required columns. "
            + " | ".join(missing_details)
            + ". Clear the Streamlit cache or update mock_data.py so the generated schema matches the dashboard."
        )
    return False


def horizontal_bar_chart(data: pd.DataFrame, category: str, values: list[str]):
    chart_data = data.reset_index()
    if category not in chart_data.columns:
        chart_data = chart_data.rename(columns={chart_data.columns[0]: category})

    if len(values) == 1:
        value = values[0]
        spec = {
            "mark": {"type": "bar", "cornerRadiusEnd": 3},
            "encoding": {
                "y": {"field": category, "type": "nominal", "sort": "-x", "title": None},
                "x": {"field": value, "type": "quantitative", "title": value.replace("_", " ").title()},
                "tooltip": [
                    {"field": category, "type": "nominal", "title": category.replace("_", " ").title()},
                    {"field": value, "type": "quantitative", "title": value.replace("_", " ").title()},
                ],
            },
        }
    else:
        chart_data = chart_data.melt(id_vars=[category], value_vars=values, var_name="measure", value_name="value")
        spec = {
            "mark": {"type": "bar", "cornerRadiusEnd": 3},
            "encoding": {
                "y": {"field": category, "type": "nominal", "title": None},
                "x": {"field": "value", "type": "quantitative", "title": "Average Confidence"},
                "yOffset": {"field": "measure"},
                "color": {"field": "measure", "type": "nominal", "title": None},
                "tooltip": [
                    {"field": category, "type": "nominal", "title": category.replace("_", " ").title()},
                    {"field": "measure", "type": "nominal", "title": "Measure"},
                    {"field": "value", "type": "quantitative", "title": "Value", "format": ".2f"},
                ],
            },
        }
    st.vega_lite_chart(chart_data, spec, width="stretch")


def filter_mock_data(users: pd.DataFrame, sessions: pd.DataFrame, key_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    lang = st.session_state.language
    with st.popover(t("filters", lang)):
        st.caption(t("filter_help", lang))
        personas = multi_filter(t("persona", lang), users["persona"], f"{key_prefix}_persona")
        user_types = multi_filter(t("user_type", lang), users["user_type"], f"{key_prefix}_user_type")
        literacy_levels = multi_filter(t("financial_literacy", lang), users["financial_literacy"], f"{key_prefix}_literacy")
        coaching_styles = multi_filter(t("coaching_style", lang), users["coaching_style"], f"{key_prefix}_coaching_style")
        risk_levels = multi_filter(t("risk_level", lang), users["risk_level"], f"{key_prefix}_risk_level")
        topic_categories = multi_filter(t("topic_category", lang), sessions["topic_category"], f"{key_prefix}_topic_category")

    filtered_users = users[
        users["persona"].isin(personas)
        & users["user_type"].isin(user_types)
        & users["financial_literacy"].isin(literacy_levels)
        & users["coaching_style"].isin(coaching_styles)
        & users["risk_level"].isin(risk_levels)
    ]
    filtered_sessions = sessions[
        sessions["user_id"].isin(filtered_users["user_id"])
        & sessions["topic_category"].isin(topic_categories)
    ]
    filtered_users = filtered_users[filtered_users["user_id"].isin(filtered_sessions["user_id"])]
    return filtered_users, filtered_sessions


def render_funnel(stage_counts: list[tuple[str, int]]):
    counts = [count for _, count in stage_counts]
    colors = ["#2f855a", "#3f956a", "#52a67a", "#69b98d", "#82c9a1"]
    rows = []

    for (stage, count), color in zip(stage_counts, colors * 2):
        width = max(34, round(count / max(counts) * 100))
        rows.append(
            f"""
            <div style="width:{width}%; background:{color}; color:white; margin:0 auto 6px; padding:10px 8px;
                text-align:center; border-radius:4px; font-size:14px; line-height:1.2;">
                <strong>{stage}</strong><br><span>{count:,} users</span>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def load_analytics_data() -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    mock_data = load_mock_data(MOCK_DATA_SCHEMA_VERSION)
    users = mock_data["users"]
    sessions = mock_data["sessions"]

    if not validate_dashboard_schema(users, sessions, show_warning=False):
        load_mock_data.clear()
        importlib.reload(mock_data_module)
        mock_data = load_mock_data(MOCK_DATA_SCHEMA_VERSION)
        users = mock_data["users"]
        sessions = mock_data["sessions"]
        if not validate_dashboard_schema(users, sessions):
            return None, None
    return users, sessions


def active_user_trend(sessions: pd.DataFrame) -> pd.DataFrame:
    activity = sessions[["session_date", "user_id"]].copy()
    activity["session_date"] = pd.to_datetime(activity["session_date"])
    dates = pd.date_range(activity["session_date"].min(), activity["session_date"].max(), freq="D")
    rows = []
    for current_date in dates:
        rows.append({
            "date": current_date,
            "DAU": activity.loc[activity["session_date"] == current_date, "user_id"].nunique(),
            "WAU": activity.loc[activity["session_date"].between(current_date - pd.Timedelta(days=6), current_date), "user_id"].nunique(),
            "MAU": activity.loc[activity["session_date"].between(current_date - pd.Timedelta(days=29), current_date), "user_id"].nunique(),
        })
    return pd.DataFrame(rows).set_index("date")


def retention_trend(users: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    session_counts = sessions["user_id"].value_counts()
    retention = users[["user_id", "signup_date"]].copy()
    retention["signup_week"] = pd.to_datetime(retention["signup_date"]).dt.to_period("W").dt.start_time
    retention["returned_user"] = retention["user_id"].map(session_counts).fillna(0).gt(1)
    return retention.groupby("signup_week")["returned_user"].mean().mul(100).to_frame("Return Usage Rate")


def render_business_metrics():
    lang = st.session_state.language
    users, sessions = load_analytics_data()
    if users is None:
        return
    users, sessions = filter_mock_data(users, sessions, "business")
    if users.empty or sessions.empty:
        st.warning(t("no_matching_sessions", lang))
        return

    st.markdown(f"**{t('business_metrics_answer', lang)}**")
    st.caption(t("filtered_view", lang).format(users=len(users), sessions=len(sessions)))

    st.header(t("acquisition_engagement", lang))
    trend = active_user_trend(sessions)
    latest = trend.iloc[-1]
    engagement = st.columns(5)
    engagement[0].metric("DAU", f"{latest['DAU']:,.0f}")
    engagement[1].metric("WAU", f"{latest['WAU']:,.0f}")
    engagement[2].metric("MAU", f"{latest['MAU']:,.0f}")
    engagement[3].metric("DAU / MAU", f"{latest['DAU'] / max(1, latest['MAU']):.1%}")
    engagement[4].metric(t("average_sessions_user", lang), f"{len(sessions) / len(users):.2f}")
    st.subheader(t("active_user_trend", lang))
    st.line_chart(trend)

    st.header(t("funnel", lang))
    old_stage_rank = {"started_chat": 0, "shared_goal": 1, "received_guidance": 2, "viewed_recommendation": 3, "activated": 4}
    user_stage_rank = users["funnel_stage"].map(old_stage_rank)
    completed_onboarding = sessions.loc[sessions["completed_session"], "user_id"].nunique()
    raw_counts = [
        (t("started_conversation", lang), len(users)),
        (t("completed_onboarding", lang), completed_onboarding),
        (t("goal_identified", lang), user_stage_rank.ge(1).sum()),
        (t("recommendation_delivered", lang), user_stage_rank.ge(2).sum()),
        (t("next_step_accepted", lang), sessions.loc[sessions["recommended_action_taken"], "user_id"].nunique()),
        (t("returned_user", lang), sessions["user_id"].value_counts().gt(1).sum()),
    ]
    funnel_counts = []
    previous_count = len(users)
    for label, count in raw_counts:
        previous_count = min(previous_count, int(count))
        funnel_counts.append((label, previous_count))
    render_funnel(funnel_counts)

    st.header(t("business_outcomes", lang))
    returning_users = sessions["user_id"].value_counts().gt(1).sum() / len(users)
    outcomes = st.columns(5)
    outcomes[0].metric(t("activation_rate", lang), f"{users['activated'].mean():.1%}")
    outcomes[1].metric(t("accepted_next_step_rate", lang), f"{sessions['recommended_action_taken'].mean():.1%}")
    outcomes[2].metric(t("return_usage_rate", lang), f"{returning_users:.1%}")
    outcomes[3].metric(t("retention_proxy", lang), f"{users['recurring_deposit_started'].mean():.1%}")
    outcomes[4].metric(t("escalation_rate", lang), f"{sessions['escalated_to_human'].mean():.1%}")

    left, right = st.columns(2)
    with left:
        st.subheader(t("persona_breakdown", lang))
        horizontal_bar_chart(users["persona"].value_counts().rename_axis("persona").to_frame("users"), "persona", ["users"])
    with right:
        st.subheader(t("retention_trend", lang))
        st.line_chart(retention_trend(users, sessions))


def render_system_metrics():
    lang = st.session_state.language
    render_feedback_summary()
    users, sessions = load_analytics_data()
    if users is None:
        return
    users, sessions = filter_mock_data(users, sessions, "system")
    if users.empty or sessions.empty:
        st.warning(t("no_matching_sessions", lang))
        return

    st.markdown(f"**{t('system_metrics_answer', lang)}**")
    st.caption(t("filtered_view", lang).format(users=len(users), sessions=len(sessions)))

    st.header(t("critical_quality_metrics", lang))
    st.caption(t("quality_gate_caption", lang))
    quality = st.columns(3)
    quality[0].metric(t("accuracy", lang), f"{sessions['factual_accuracy'].mean():.1%}")
    quality[1].metric(t("safety", lang), f"{sessions['safety_compliance'].mean():.1%}")
    quality[2].metric(t("integrity", lang), f"{sessions['integrity_policy_compliance'].mean():.1%}")
    st.markdown(t("quality_definitions", lang))
    st.caption(t("quality_examples", lang))

    st.header(t("user_experience_metrics", lang))
    st.caption(t("ux_caption", lang))
    ux_metrics = {
        t("helpfulness", lang): users["helpfulness"].mean(),
        t("trust", lang): users["trust"].mean(),
        t("confidence_building", lang): users["confidence_building"].mean(),
        t("adaptability", lang): users["adaptability"].mean(),
        t("efficiency", lang): users["efficiency"].mean(),
        t("satisfaction", lang): users["inferred_satisfaction"].mean(),
    }
    scorecards = st.columns(6)
    for column, (label, value) in zip(scorecards, ux_metrics.items()):
        column.metric(label, f"{value:.2f} / 5")
    st.markdown(t("ux_definitions", lang))
    ux_daily = sessions[["session_date", "user_id"]].merge(
        users[["user_id", "helpfulness", "trust", "confidence_building", "adaptability", "efficiency", "inferred_satisfaction"]],
        on="user_id",
    )
    ux_daily["session_date"] = pd.to_datetime(ux_daily["session_date"])
    ux_trend = ux_daily.groupby("session_date")[["helpfulness", "trust", "confidence_building", "adaptability", "efficiency", "inferred_satisfaction"]].mean()
    st.subheader(t("ux_metric_trend", lang))
    st.line_chart(ux_trend)

    st.header(t("diagnostic_metrics", lang))
    st.caption(t("diagnostic_caption", lang))
    diagnostic_chart = pd.DataFrame([
        {"metric": "Intent Accuracy", "module": "Conversation Understanding", "score_pct": sessions["intent_accuracy"].mean() * 100},
        {"metric": "Reference Resolution Accuracy", "module": "Conversation Understanding", "score_pct": sessions["resolved_reference_success"].mean() * 100},
        {"metric": "Slot Completion Accuracy", "module": "Conversation Understanding", "score_pct": sessions["slot_completion_accuracy"].mean() * 100},
        {"metric": "Context Retention", "module": "Conversation Understanding", "score_pct": sessions["context_retention"].mean() * 100},
        {"metric": "Dialogue Policy Accuracy", "module": "Dialogue Management", "score_pct": sessions["dialogue_policy_accuracy"].mean() * 100},
        {"metric": "Perceived Understanding", "module": "Dialogue Management", "score_pct": sessions["perceived_understanding"].mean() / 5 * 100},
        {"metric": "Retrieval Relevance", "module": "Knowledge Retrieval", "score_pct": sessions["retrieval_relevance"].mean() * 100},
        {"metric": "Groundedness", "module": "Knowledge Retrieval", "score_pct": sessions["groundedness"].mean() * 100},
    ])
    diagnostic_spec = {
        "mark": {"type": "bar", "cornerRadiusEnd": 3},
        "encoding": {
            "y": {
                "field": "metric",
                "type": "nominal",
                "sort": "-x",
                "title": None,
                "axis": {"labelLimit": 320},
            },
            "x": {"field": "score_pct", "type": "quantitative", "title": "Score (%)"},
            "color": {"field": "module", "type": "nominal", "title": "Diagnostic Layer"},
            "tooltip": [
                {"field": "metric", "type": "nominal", "title": "Metric"},
                {"field": "module", "type": "nominal", "title": "Diagnostic Layer"},
                {"field": "score_pct", "type": "quantitative", "title": "Score (%)", "format": ".1f"},
            ],
        },
    }
    st.vega_lite_chart(diagnostic_chart, diagnostic_spec, width="stretch")
    st.info(t("diagnostic_insight", lang))


def render_sidebar():
    lang = st.session_state.language
    language_options = {
        t("english", lang): "en",
        t("german", lang): "de",
        t("spanish", lang): "es",
    }
    selected_label = next(label for label, code in language_options.items() if code == lang)
    selected_language = st.selectbox(
        t("language", lang),
        list(language_options),
        index=list(language_options).index(selected_label),
    )
    if language_options[selected_language] != lang:
        st.session_state.language = language_options[selected_language]
        update_opening_message()
        st.rerun()
    st.caption(t("prototype_note", lang))
    st.toggle(t("debug_mode", lang), key="debug_mode")
    if st.button(t("log_out", lang)):
        st.session_state.clear()
        st.rerun()

    if st.session_state.get("debug_mode"):
        context = st.session_state.conversation_context
        st.header("Customer Understanding")
        sidebar_fields = {
            "Primary Topic": context["primary_topic"] or "None",
            "Parent Topic": context["parent_topic"] or "None",
            "Resolved Reference": context["resolved_reference"],
            "Current Goal": context["current_goal"],
            "Current Fear": context["current_fear"],
            "Persona": context["persona"],
            "Knowledge Level": context["knowledge_level"],
            "Knowledge Level Confidence": context["knowledge_level_confidence"],
            "Knowledge Level Evidence": context["knowledge_level_evidence"],
            "Financial Situation Confidence": context["confidence_level"],
            "Coaching Style": context["coaching_style"],
            "Risk Level": context["risk_level"],
        }
        for label, value in sidebar_fields.items():
            st.markdown(f"**{label}:** {value}")
    if st.button(t("reset_conversation", lang)):
        reset_conversation()


def render_understanding_tab():
    lang = st.session_state.language
    st.subheader(t("conversation_context", lang))
    st.json(st.session_state.conversation_context)

    st.subheader(t("current_message_understanding", lang))
    if st.session_state.last_understanding:
        st.json(st.session_state.last_understanding)
    else:
        st.info(t("no_analyzed_message", lang))

    st.subheader(t("dialogue_plan", lang))
    if st.session_state.last_dialogue_plan:
        st.json(st.session_state.last_dialogue_plan)
    else:
        st.info(t("no_dialogue_objective", lang))

    st.subheader(t("recognized_topics", lang))
    topics = st.session_state.last_understanding.get("recognized_topics", [])
    if topics:
        rows = [
            {
                "primary_topic": index == 0,
                "topic": topic["topic"],
                "label": topic["label"],
                "parent_topic": topic["parent_topic"],
                "specificity": topic["specificity"],
                "contextual_salience": topic["contextual_salience"],
            }
            for index, topic in enumerate(topics)
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info(t("no_recognized_topic", lang))

    st.subheader(t("rag_data", lang))
    if st.session_state.last_retrieved_knowledge:
        st.json(st.session_state.last_retrieved_knowledge)
    else:
        st.info(t("no_retrieved_knowledge", lang))

    st.subheader(t("tool_results", lang))
    if st.session_state.last_tool_results:
        st.json(st.session_state.last_tool_results)
    else:
        st.info(t("no_tool_result", lang))

    st.subheader(t("guardrails", lang))
    st.json(st.session_state.last_guardrail_result)


ensure_assistant_message_metadata()

with st.sidebar:
    render_sidebar()

_, openai_api_key_source = resolve_openai_api_key(
    st.secrets,
    api_key=st.session_state.get("openai_api_key"),
)
openai_key_debug = get_openai_key_debug(openai_api_key_source)

st.markdown(
    """
    <style>
    [data-testid="stMain"] {
        overflow: hidden;
    }
    [data-testid="stMainBlockContainer"] {
        height: 100vh;
        max-height: 100vh;
        overflow: hidden;
        padding-top: 3.75rem;
        padding-bottom: 0.75rem;
    }
    .st-key-app_shell_header {
        background: var(--background-color);
        padding-bottom: 0.35rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.35);
        margin-bottom: 0.75rem;
    }
    .st-key-conversation_scroll,
    .st-key-view_content_scroll {
        height: calc(100vh - 16rem) !important;
        min-height: 16rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
with st.container(key="app_shell_header"):
    st.title("🌴 AJ's AI Coach")
    st.caption(t("app_caption", lang))
    active_view = st.radio(
        "View",
        ["coach", "conversational_understanding", "business_metrics", "system_metrics", "architectural_design_log"],
        format_func=lambda view: "Architectural Design Log" if view == "architectural_design_log" else t(view, lang),
        horizontal=True,
        label_visibility="collapsed",
        key="active_view",
    )

if active_view == "coach":
    conversation_scroll = st.container(
        key="conversation_scroll",
        height=480,
        border=False,
        autoscroll=True,
    )
    with conversation_scroll:
        for message in st.session_state.messages:
            avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else None
            with st.chat_message(message["role"], avatar=avatar):
                st.write(message["content"])
                if message["role"] == "assistant" and message["message_id"] != "assistant-opening-message":
                    render_feedback_controls(message)

        if st.session_state.get("debug_mode"):
            with st.expander(t("response_debug", lang), expanded=False):
                st.json({**st.session_state.last_response_debug, **openai_key_debug})

    prompt = st.chat_input(t("chat_placeholder", lang))
    if prompt:
        total_started = perf_counter()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with conversation_scroll:
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                with st.spinner(t("thinking_through_goals", lang)):
                    understanding_started = perf_counter()
                    knowledge_level = classify_knowledge_level(
                        user_message=prompt,
                        conversation_context=st.session_state.conversation_context,
                        recent_conversation_history=[
                            {"role": message.get("role"), "content": message.get("content", "")}
                            for message in st.session_state.messages[-7:-1]
                        ],
                        api_key=st.session_state.get("openai_api_key"),
                        streamlit_secrets=st.secrets,
                    )
                    apply_profile_update_operations(st.session_state.conversation_context, knowledge_level)
                    understanding = analyze_message(prompt, st.session_state.conversation_context)
                    apply_profile_update_operations(understanding, knowledge_level)
                    understanding.update(knowledge_level)
                    understanding["financial_literacy"] = knowledge_level["knowledge_level"]
                    understanding["knowledge_level_evidence"] = knowledge_level["evidence"]
                    guardrails_started = perf_counter()
                    guardrail_result = evaluate_guardrails(
                        prompt,
                        knowledge_level.get("guardrail_categories"),
                    )
                    guardrail_result["out_of_domain_answer"] = knowledge_level.get("out_of_domain_answer")
                    understanding["guardrail_categories"] = guardrail_result["categories"]
                    understanding["guardrail_triggered"] = guardrail_result["guardrail_triggered"]
                    guardrails_ms = (perf_counter() - guardrails_started) * 1000
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
                    dialogue_plan = decide_dialogue_plan(
                        understanding,
                        user_model,
                        prompt,
                        st.session_state.conversation_context,
                    )
                    conversation_understanding_ms = (perf_counter() - understanding_started) * 1000

                    retrieval_started = perf_counter()
                    retrieved_knowledge = retrieve_knowledge(understanding["primary_topic"])
                    tool_results = collect_tool_results(prompt, understanding["primary_topic"])
                    retrieval_ms = (perf_counter() - retrieval_started) * 1000

                    openai_started = perf_counter()
                    response_details = generate_llm_response(
                        user_message=prompt,
                        conversation_context=st.session_state.conversation_context,
                        dialogue_plan=dialogue_plan,
                        retrieved_knowledge=retrieved_knowledge,
                        guardrail_decision=guardrail_result,
                        language=lang,
                        understanding=understanding,
                        tool_results=tool_results,
                        api_key=st.session_state.get("openai_api_key"),
                        streamlit_secrets=st.secrets,
                    )
                    openai_call_ms = (perf_counter() - openai_started) * 1000
        response = response_details["response_text"]
        assistant_content = (
            localize_coach_response(response, lang)
            if response_details["response_source"] != "llm"
            else response
        )
        update_conversation_context(st.session_state.conversation_context, prompt, understanding)
        record_assistant_response(st.session_state.conversation_context, response)
        record_dialogue_plan(st.session_state.conversation_context, dialogue_plan)
        st.session_state.last_understanding = understanding
        st.session_state.last_dialogue_plan = dialogue_plan
        st.session_state.last_retrieved_knowledge = retrieved_knowledge
        st.session_state.last_tool_results = tool_results
        st.session_state.last_guardrail_result = guardrail_result
        st.session_state.last_response_debug = {
            "detected_intent": understanding["intent"],
            "primary_topic": understanding["primary_topic"],
            "parent_topic": understanding["parent_topic"],
            "dialogue_act": dialogue_plan["dialogue_act"],
            "knowledge_level_source": understanding["knowledge_level_source"],
            "updated_fields": knowledge_level["field_updates"],
            "invalidated_fields": knowledge_level["field_invalidations"],
            "unchanged_fields": knowledge_level["unchanged_fields"],
            "update_confidence": knowledge_level["update_confidence"],
            "guardrail_triggered": guardrail_result["guardrail_triggered"],
            "OPENAI_API_KEY detected": response_details["openai_api_key_detected"],
            "OPENAI_API_KEY source": response_details["openai_api_key_source"],
            "openai_model": response_details["openai_model"],
            "OpenAI API call attempted": response_details["openai_api_call_attempted"],
            "OpenAI API call succeeded": response_details["openai_api_call_succeeded"],
            "openai_error_type": response_details["openai_error_type"],
            "openai_error_message": response_details["openai_error_message"],
            "response_source": response_details["response_source"],
            "retrieved_knowledge_used": retrieved_knowledge.get("topic") if retrieved_knowledge else None,
            "conversation_understanding_ms": round(conversation_understanding_ms, 1),
            "retrieval_ms": round(retrieval_ms, 1),
            "guardrails_ms": round(guardrails_ms, 1),
            "openai_call_ms": round(openai_call_ms, 1),
            "total_response_ms": round((perf_counter() - total_started) * 1000, 1),
        }
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "message_id": f"assistant-{uuid4()}",
                "feedback_context": {
                    "user_message_context": prompt,
                    "detected_intent": understanding.get("intent"),
                    "primary_topic": understanding.get("primary_topic"),
                    "persona": understanding.get("persona"),
                    "risk_level": understanding.get("risk_level"),
                    "language": lang,
                },
            }
        )
        st.rerun()

else:
    with st.container(key="view_content_scroll", height=480, border=False):
        if active_view == "conversational_understanding":
            render_understanding_tab()
        elif active_view == "business_metrics":
            render_business_metrics()
        elif active_view == "system_metrics":
            render_system_metrics()
        elif active_view == "architectural_design_log":
            render_architectural_design_log()
