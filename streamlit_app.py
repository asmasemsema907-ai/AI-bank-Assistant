from html import escape
from importlib import import_module
import logging

import requests
import streamlit as st

rag = import_module("07_prompting")

BANK_ICON = "\U0001F3E6"
BOT_ICON = "\U0001F916"
DOC_ICON = "\U0001F4C4"
CLOSE_ICON = "x"

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Bank Assistant",
    page_icon=BANK_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)


def get_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value).strip() if value is not None else None


try:
    rag.configure_generation(
        ollama_host=get_secret("OLLAMA_HOST"),
        ollama_model=get_secret("OLLAMA_MODEL"),
        openrouter_api_key=get_secret("OPENROUTER_API_KEY"),
        openrouter_model=get_secret("OPENROUTER_MODEL"),
        openrouter_base_url=get_secret("OPENROUTER_BASE_URL"),
        openrouter_site_url=get_secret("OPENROUTER_SITE_URL"),
        openrouter_app_name=get_secret("OPENROUTER_APP_NAME"),
        gemini_api_key=get_secret("GEMINI_API_KEY"),
        gemini_model=get_secret("GEMINI_MODEL"),
        gemini_base_url=get_secret("GEMINI_BASE_URL"),
        generation_provider=get_secret("GENERATION_PROVIDER"),
    )
except Exception:
    logger.warning("Streamlit secrets were unavailable or invalid; using .env/defaults.")


@st.cache_resource(show_spinner="Setting up the knowledge base...")
def ensure_vector_store():
    store_module = import_module("05_create_chroma_store")
    return store_module.create_vector_store()


STRINGS = {
    "en": {
        "chat_title": "AI Bank Assistant",
        "greeting": (
            "Hi, I'm the AI Bank Assistant. Ask me about accounts, KYC, loans, "
            "digital banking, security, or banking policy in English or Arabic."
        ),
        "placeholder": "Ask about accounts, KYC, security...",
        "clear": "Clear conversation",
        "toggle_label": "AR",
    },
    "ar": {
        "chat_title": "\u0627\u0644\u0645\u0633\u0627\u0639\u062f \u0627\u0644\u0645\u0635\u0631\u0641\u064a \u0627\u0644\u0630\u0643\u064a",
        "greeting": (
            "\u0645\u0631\u062d\u0628\u0627\u060c \u0623\u0646\u0627 \u0627\u0644\u0645\u0633\u0627\u0639\u062f "
            "\u0627\u0644\u0645\u0635\u0631\u0641\u064a \u0627\u0644\u0630\u0643\u064a. "
            "\u0627\u0633\u0623\u0644\u0646\u064a \u0639\u0646 \u0627\u0644\u062d\u0633\u0627\u0628\u0627\u062a\u060c "
            "KYC\u060c \u0627\u0644\u0642\u0631\u0648\u0636\u060c \u0627\u0644\u062e\u062f\u0645\u0627\u062a "
            "\u0627\u0644\u0631\u0642\u0645\u064a\u0629\u060c \u0623\u0648 \u0633\u064a\u0627\u0633\u0627\u062a "
            "\u0627\u0644\u0628\u0646\u0643."
        ),
        "placeholder": "\u0627\u0633\u0623\u0644 \u0639\u0646 \u0627\u0644\u062d\u0633\u0627\u0628\u0627\u062a\u060c KYC\u060c \u0623\u0648 \u0627\u0644\u0623\u0645\u0627\u0646...",
        "clear": "\u0645\u0633\u062d \u0627\u0644\u0645\u062d\u0627\u062f\u062b\u0629",
        "toggle_label": "EN",
    },
}

SAMPLE_PROMPTS = [
    {
        "category": "Basic Accounts",
        "question": "What is a Savings Account?",
    },
    {
        "category": "Operations & Security",
        "question": "How do I reset my online banking password?",
    },
    {
        "category": "Compliance",
        "question": "What does KYC stand for?",
    },
]


def html_text(value: object) -> str:
    return escape(str(value), quote=True).replace("\n", "<br>")


def init_session_state() -> None:
    st.session_state.setdefault("chat_open", False)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("ui_lang", "en")
    st.session_state.setdefault("pending_chat_query", None)
    st.session_state.setdefault("chat_text_field", "")


def inject_css() -> None:
    st.markdown(
        """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {display: none;}

            :root {
                --sb-page: #f6f8fb;
                --sb-ink: #0e1726;
                --sb-muted: #5d6b82;
                --sb-line: #dfe7f1;
                --sb-blue: #1259a4;
                --sb-blue-bright: #1972df;
                --sb-blue-soft: #eaf3ff;
            }

            body,
            .stApp {
                background: var(--sb-page);
                color: var(--sb-ink);
            }

            .block-container {
                max-width: 1120px;
                padding-top: 1.5rem;
                padding-bottom: 7rem;
            }

            .metric-card {
                height: 128px;
                min-height: 128px;
                padding: 18px 20px;
                background: #ffffff;
                border: 1px solid var(--sb-line);
                border-left: 3px solid rgba(25, 114, 223, 0.65);
                border-radius: 10px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.045);
                display: flex;
                flex-direction: column;
                justify-content: center;
                gap: 8px;
            }

            .metric-label {
                color: #334155;
                font-size: 0.82rem;
                font-weight: 650;
                line-height: 1.2;
            }

            .metric-value {
                color: #172033;
                font-size: 1.68rem;
                line-height: 1.14;
                font-weight: 500;
                letter-spacing: 0;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            .metric-status {
                width: fit-content;
                display: inline-flex;
                align-items: center;
                gap: 6px;
                border-radius: 999px;
                padding: 4px 9px;
                font-size: 0.78rem;
                font-weight: 650;
                line-height: 1;
            }

            .metric-status.is-online {
                background: #dcfce7;
                color: #047857;
            }

            .metric-status.is-offline {
                background: #fff7ed;
                color: #b45309;
            }

            .status-light {
                width: 9px;
                height: 9px;
                border-radius: 999px;
                display: inline-block;
                flex: 0 0 auto;
            }

            .status-light.is-online {
                background: #22c55e;
                box-shadow:
                    0 0 0 3px rgba(34, 197, 94, 0.18),
                    0 0 12px rgba(34, 197, 94, 0.55);
                animation: statusPulse 1.8s ease-in-out infinite;
            }

            .status-light.is-offline {
                background: #f59e0b;
                box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.16);
            }

            @keyframes statusPulse {
                0%, 100% {
                    box-shadow:
                        0 0 0 3px rgba(34, 197, 94, 0.18),
                        0 0 12px rgba(34, 197, 94, 0.55);
                }
                50% {
                    box-shadow:
                        0 0 0 5px rgba(34, 197, 94, 0.10),
                        0 0 18px rgba(34, 197, 94, 0.72);
                }
            }
            .st-key-sample_prompts_panel {
                background: #ffffff;
                border: 1px solid var(--sb-line);
                border-radius: 8px;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
                padding: 14px 18px 12px;
                margin: 18px 0 22px;
            }

            .sample-topic {
                display: inline-flex;
                align-items: center;
                min-height: 20px;
                border-radius: 5px;
                background: var(--sb-blue-soft);
                color: #004f99;
                font-size: 0.78rem;
                line-height: 1;
                padding: 3px 8px;
                white-space: nowrap;
            }

            .sample-divider {
                height: 1px;
                background: repeating-linear-gradient(
                    90deg,
                    #dbe5f0 0,
                    #dbe5f0 4px,
                    transparent 4px,
                    transparent 8px
                );
                margin: 4px 0 6px;
            }

            div[class*="st-key-sample_prompt_"] button {
                justify-content: flex-start !important;
                width: 100%;
                min-height: 28px;
                padding: 0 0 0 2px;
                border: 0;
                border-radius: 6px;
                background: transparent;
                color: #0f172a;
                font-size: 0.94rem;
                font-weight: 500;
                line-height: 1.25;
                text-align: left;
                box-shadow: none;
            }

            div[class*="st-key-sample_prompt_"] button > div,
            div[class*="st-key-sample_prompt_"] button span,
            div[class*="st-key-sample_prompt_"] button [data-testid="stMarkdownContainer"],
            div[class*="st-key-sample_prompt_"] button p {
                display: flex !important;
                justify-content: flex-start !important;
                width: 100% !important;
                margin: 0;
                text-align: left;
                white-space: normal;
            }

            div[class*="st-key-sample_prompt_"] button:hover {
                background: #f1f7ff;
                color: #004f99;
            }

            div[class*="st-key-sample_prompt_"] button:focus-visible,
            .st-key-chat_toggle_btn button:focus-visible,
            .st-key-chat_toggle_btn_open button:focus-visible {
                outline: 3px solid rgba(25, 114, 223, 0.28);
                outline-offset: 3px;
            }

            .assistant-note {
                background: #eaf3ff;
                border: 1px solid #c6ddff;
                border-radius: 8px;
                color: #004384;
                padding: 13px 16px;
                font-size: 0.94rem;
            }

              .st-key-chat_toggle_btn button,
            .st-key-chat_toggle_btn_open button {
                position: fixed;
                right: 24px;
                bottom: 24px;
                z-index: 10000;
                display: flex;
                align-items: center;
                justify-content: center;
                width: 64px !important;
                height: 64px !important;
                min-width: 64px !important;
                min-height: 64px !important;
                padding: 0 !important;
                border: 0;
                border-radius: 20px 20px 6px 20px;
                background: linear-gradient(135deg, #105292 0%, #145fb4 46%, #1972df 100%);
                color: #ffffff;
                line-height: 1;
                overflow: visible;
                box-shadow:
                    0 18px 34px rgba(16, 82, 146, 0.36),
                    0 7px 14px rgba(12, 64, 113, 0.24);
                transition:
                    transform 140ms ease,
                    box-shadow 140ms ease,
                    filter 140ms ease;
            }

            .st-key-chat_toggle_btn button:hover,
            .st-key-chat_toggle_btn_open button:hover {
                transform: translateY(-2px);
                filter: saturate(1.06);
                box-shadow:
                    0 20px 38px rgba(16, 82, 146, 0.40),
                    0 8px 16px rgba(12, 64, 113, 0.26);
            }

            .st-key-chat_toggle_btn button:active,
            .st-key-chat_toggle_btn_open button:active {
                transform: translateY(0);
            }

            .st-key-chat_toggle_btn button p {
                width: 0;
                margin: 0;
                opacity: 0;
                overflow: hidden;
            }

            .st-key-chat_toggle_btn button::before {
                content: "";
                position: absolute;
                width: 31px;
                height: 24px;
                border: 3px solid rgba(255, 255, 255, 0.96);
                border-radius: 11px 11px 11px 4px;
                background:
                    radial-gradient(circle at 10px 11px, #ffffff 0 2px, transparent 2.5px),
                    radial-gradient(circle at 21px 11px, #ffffff 0 2px, transparent 2.5px),
                    rgba(255, 255, 255, 0.14);
                box-shadow:
                    inset 0 0 0 1px rgba(255, 255, 255, 0.20),
                    0 3px 8px rgba(3, 29, 62, 0.18);
            }

            .st-key-chat_toggle_btn button::after {
                content: "";
                position: absolute;
                top: 10px;
                right: 10px;
                width: 12px;
                height: 12px;
                border: 2px solid #ffffff;
                border-radius: 999px;
                background: #22c55e;
                box-shadow:
                    0 0 0 5px rgba(34, 197, 94, 0.20),
                    0 0 14px rgba(34, 197, 94, 0.70);
            }
            .st-key-chat_toggle_btn_open button p {
                margin: 0;
                font-size: 22px;
                font-weight: 800;
                line-height: 1;
            }
            .st-key-chat_toggle_btn_open button::before,
            .st-key-chat_toggle_btn_open button::after {
                display: none;
            }
            .st-key-chat_toggle_btn button::before {
                content: "";
                position: absolute;
                width: 30px;
                height: 24px;
                border-radius: 7px;
                background:
                    radial-gradient(circle at 8px 9px, #1359a4 0 2px, transparent 2.5px),
                    radial-gradient(circle at 16px 9px, #1359a4 0 2px, transparent 2.5px),
                    linear-gradient(#f07fa4, #f07fa4) 50% 14px / 11px 3px no-repeat,
                    linear-gradient(180deg, #b9c8ff 0%, #eda2be 100%);
                box-shadow:
                    inset 0 0 0 1px rgba(255, 255, 255, 0.68),
                    0 1px 2px rgba(3, 29, 62, 0.22);
            }

            .st-key-chat_toggle_btn button::after {
                content: "";
                position: absolute;
                width: 40px;
                height: 31px;
                background:
                    radial-gradient(circle at 16px 2px, #f07fa4 0 2px, transparent 2.6px),
                    linear-gradient(#f07fa4, #f07fa4) 15px 3px / 2px 7px no-repeat,
                    radial-gradient(circle at 4px 15px, #f07fa4 0 3px, transparent 3.6px),
                    radial-gradient(circle at 28px 15px, #f07fa4 0 3px, transparent 3.6px);
            }

            .st-key-chat_toggle_btn_open button p {
                margin: 0;
                font-size: 20px;
                font-weight: 700;
                line-height: 1;
            }

            .st-key-chat_window {
                position: fixed;
                right: 24px;
                bottom: 86px;
                z-index: 9999;
                width: min(400px, calc(100vw - 32px));
                height: min(680px, calc(100vh - 112px));
                max-height: calc(100vh - 112px);
                overflow: hidden;
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 18px;
                box-shadow: 0 18px 54px rgba(15, 23, 42, 0.22);
                padding: 16px;
                display: flex;
                flex-direction: column;
            }

            /* Only the messages list should flex/scroll; the header row and
               the input/clear controls keep their natural height. (Streamlit
               applies the "st-key-chat_window" class directly to the vertical
               block that holds ALL of these as direct children, so a rule
               like ".st-key-chat_window > div { flex: 1 1 auto }" would wrongly
               match every one of them and force them to grow equally - splitting
               the window into equal bands instead of leaving only the messages
               panel scrollable.) */
            .st-key-chat_window [data-testid="stHorizontalBlock"] {
                flex: 0 0 auto;
            }

            .st-key-chat_messages_panel {
                flex: 1 1 auto;
                min-height: 0;
            }

            .st-key-chat_messages_panel > div {
                height: 100% !important;
                max-height: 100% !important;
                overflow-y: auto !important;
            }

            .st-key-chat_input_panel {
                flex: 0 0 auto;
            }

            .st-key-clear_chat_btn {
                flex: 0 0 auto;
            }

            .chat-header {
                color: #0f172a;
                font-size: 0.98rem;
                font-weight: 750;
                line-height: 1.25;
                padding-top: 2px;
            }

            .st-key-lang_toggle_btn button {
                min-height: 28px;
                height: 28px;
                padding: 0 10px;
                border: 1px solid #cfe0f5;
                border-radius: 14px;
                background: #eaf3ff;
                color: #0f4c81;
                font-size: 0.72rem;
                font-weight: 750;
            }

            .chat-bubble-user,
            .chat-bubble-bot {
                width: fit-content;
                padding: 9px 13px;
                margin: 7px 0;
                font-size: 0.9rem;
                line-height: 1.45;
                unicode-bidi: plaintext;
                overflow-wrap: anywhere;
            }

            .chat-rtl {
                direction: rtl;
                text-align: right;
            }

               .chat-ltr {
                direction: ltr;
                text-align: left;
            }

            .typing-bubble {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                max-width: 78%;
                color: #475569;
                background: #f1f5f9;
            }

            .typing-dots {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                direction: ltr;
            }

            .typing-dot {
                width: 6px;
                height: 6px;
                border-radius: 999px;
                background: #1972df;
                animation: typingPulse 1s ease-in-out infinite;
            }

            .typing-dot:nth-child(2) {
                animation-delay: 140ms;
            }

            .typing-dot:nth-child(3) {
                animation-delay: 280ms;
            }

            @keyframes typingPulse {
                0%, 80%, 100% {
                    opacity: 0.35;
                    transform: translateY(0);
                }
                40% {
                    opacity: 1;
                    transform: translateY(-3px);
                }
            }

            .chat-bubble-user {
                max-width: 85%;
                margin-left: auto;
                border-radius: 15px 15px 4px 15px;
                background: #1259a4;
                color: #ffffff;
            }

            .chat-bubble-bot {
                max-width: 90%;
                border-radius: 15px 15px 15px 4px;
                background: #f1f5f9;
                color: #172033;
            }

            .source-tag {
                display: inline-flex;
                align-items: center;
                border-radius: 8px;
                background: #eaf3ff;
                color: #0f4c81;
                padding: 2px 8px;
                margin: 2px 4px 2px 0;
                font-size: 0.72rem;
                max-width: 100%;
            }

            .st-key-chat_input_panel {
                margin-top: 12px;
            }

            .st-key-chat_input_panel [data-testid="stTextInput"] {
                margin-bottom: 0;
            }

            .st-key-chat_input_panel [data-testid="stTextInputRootElement"] {
                min-height: 56px;
                border: 1.5px solid #1972df !important;
                border-radius: 14px;
                background: #ffffff;
                box-shadow:
                    0 0 0 3px rgba(25, 114, 223, 0.08),
                    0 1px 2px rgba(15, 23, 42, 0.04);
            }

            .st-key-chat_input_panel input {
                min-height: 54px;
                color: #0f172a;
                caret-color: #1972df;
            }

            .st-key-chat_input_panel input::placeholder {
                color: #6b7280;
                opacity: 1;
            }

            .st-key-clear_chat_btn button {
                width: 100%;
                border-radius: 10px;
                border-color: #d8e3ef;
                color: #334155;
            }

            @media (max-width: 640px) {
                .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }

                .st-key-chat_toggle_btn button,
                .st-key-chat_toggle_btn_open button {
                    right: 18px;
                    bottom: 18px;
                }

                .st-key-chat_window {
                    right: 16px;
                    bottom: 78px;
                    height: min(640px, calc(100vh - 98px));
                    padding: 14px;
                }

                .sample-topic {
                    margin-bottom: 2px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, ttl=30)
def check_generation_status(
    provider: str,
    ollama_host: str,
    ollama_model: str,
    openrouter_has_key: bool,
    openrouter_model: str,
    gemini_has_key: bool,
    gemini_model: str,
) -> tuple[bool, str]:
    if provider == "gemini":
        if gemini_has_key:
            return True, f"Gemini model configured: {gemini_model}"
        return False, "GEMINI_API_KEY is not configured in Streamlit Secrets."

    if provider == "openrouter":
        if openrouter_has_key:
            return True, f"OpenRouter model configured: {openrouter_model}"
        return False, "OPENROUTER_API_KEY is not configured in Streamlit Secrets."

    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        response.raise_for_status()
        pulled_models = [m.get("name", "") for m in response.json().get("models", [])]
        if not any(ollama_model in pulled_model for pulled_model in pulled_models):
            return False, f"Ollama is running, but '{ollama_model}' is not pulled yet."
        return True, "Connected"
    except requests.exceptions.ConnectionError:
        return False, f"Cannot reach Ollama at {ollama_host}."
    except requests.exceptions.Timeout:
        return False, "Ollama status check timed out."
    except requests.exceptions.RequestException:
        return False, "Ollama returned an unavailable status."
    except ValueError:
        return False, "Ollama returned an invalid status payload."


def queue_chat_query(question: str) -> None:
    st.session_state.chat_open = True
    st.session_state.pending_chat_query = question


def queue_manual_chat_query() -> None:
    query = st.session_state.get("chat_text_field", "").strip()
    if query:
        st.session_state.pending_chat_query = query
        st.session_state.chat_text_field = ""


def render_sample_prompts() -> None:
    st.subheader("Try A Question")
    with st.container(key="sample_prompts_panel"):
        for index, prompt in enumerate(SAMPLE_PROMPTS):
            cols = st.columns([1.05, 5.95], gap="small", vertical_alignment="center")
            with cols[0]:
                st.markdown(
                    f'<span class="sample-topic">{html_text(prompt["category"])}</span>',
                    unsafe_allow_html=True,
                )
            with cols[1]:
                if st.button(
                    prompt["question"],
                    key=f"sample_prompt_{index}",
                    use_container_width=True,
                ):
                    queue_chat_query(prompt["question"])

            if index < len(SAMPLE_PROMPTS) - 1:
                st.markdown('<div class="sample-divider"></div>', unsafe_allow_html=True)


def render_dashboard() -> None:
    st.title(f"{BANK_ICON} AI Bank Assistant")
    st.caption("Banking knowledge dashboard for grounded customer-service and policy answers.")

    render_sample_prompts()

    st.markdown(
        '<div class="assistant-note">AI Bank Assistant covers '
        "accounts, compliance, security, and banking policy.</div>",
        unsafe_allow_html=True,
    )


def render_chat_message(message: dict) -> None:
    bubble_class = "chat-bubble-user" if message.get("role") == "user" else "chat-bubble-bot"
    content = message.get("content", "")
    direction = message.get("direction") or (
        "rtl" if rag.detect_question_language(str(content)) == "ar" else "ltr"
    )
    direction_class = "chat-rtl" if direction == "rtl" else "chat-ltr"
    st.markdown(
        f'<div class="{bubble_class} {direction_class}" dir="{direction}">{html_text(content)}</div>',
        unsafe_allow_html=True,
    )

    sources = message.get("sources") or []
    if sources:
        tags = "".join(
            f'<span class="source-tag">{DOC_ICON}&nbsp;{html_text(source)}</span>'
            for source in sources
        )
        st.markdown(tags, unsafe_allow_html=True)


def render_typing_indicator(direction: str) -> None:
    direction_class = "chat-rtl" if direction == "rtl" else "chat-ltr"
    label = "جاري إعداد الإجابة" if direction == "rtl" else "Preparing answer"
    st.markdown(
        f"""
        <div class="chat-bubble-bot typing-bubble {direction_class}" dir="{direction}" aria-live="polite">
            <span>{html_text(label)}</span>
            <span class="typing-dots" aria-hidden="true">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
def prepare_chat_query(user_query: str) -> dict | None:
    clean_query = (user_query or "").strip()
    if not clean_query:
        return None

    answer_language = rag.detect_question_language(clean_query)
    direction = "rtl" if answer_language == "ar" else "ltr"
    st.session_state.chat_history.append(
        {"role": "user", "content": clean_query, "direction": direction}
    )
    return {
        "clean_query": clean_query,
        "answer_language": answer_language,
        "direction": direction,
    }


def complete_chat_query(pending_payload: dict) -> None:
    answer, sources = rag.answer_question(
        question=pending_payload["clean_query"],
        language=pending_payload["answer_language"],
    )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": [source["title"] for source in sources],
            "direction": pending_payload["direction"],
        }
    )
    st.rerun()


def render_chat_launcher() -> None:
    container_key = "chat_toggle_btn_open" if st.session_state.chat_open else "chat_toggle_btn"
    with st.container(key=container_key):
        icon = CLOSE_ICON if st.session_state.chat_open else BOT_ICON
        if st.button(icon, key="chat_toggle_inner", help="AI Bank Assistant"):
            st.session_state.chat_open = not st.session_state.chat_open
            st.rerun()
def render_chat_window() -> None:
    if not st.session_state.chat_open:
        return

    labels = STRINGS[st.session_state.ui_lang]
    with st.container(key="chat_window"):
        header_col, toggle_col = st.columns([6, 1], vertical_alignment="center")
        with header_col:
            st.markdown(
                f'<div class="chat-header">{BANK_ICON} {html_text(labels["chat_title"])}</div>',
                unsafe_allow_html=True,
            )
        with toggle_col:
           with st.container(key="lang_toggle_btn"):
                    
                   if st.button(labels["toggle_label"], key="lang_toggle_inner"):
                    st.session_state.ui_lang = "ar" if st.session_state.ui_lang == "en" else "en"
                    st.rerun()

        pending_query = st.session_state.pop("pending_chat_query", None)
        pending_payload = prepare_chat_query(pending_query) if pending_query else None

        messages_container = st.container(height=480, key="chat_messages_panel")
        with messages_container:
            if not st.session_state.chat_history:
                render_chat_message({"role": "assistant", "content": labels["greeting"]})

            for message in st.session_state.chat_history:
                render_chat_message(message)

            if pending_payload:
                render_typing_indicator(pending_payload["direction"])

        with st.container(key="chat_input_panel"):
            st.text_input(
                "Message",
                placeholder=labels["placeholder"],
                key="chat_text_field",
                label_visibility="collapsed",
                on_change=queue_manual_chat_query,
                disabled=bool(pending_payload),
            )

        if pending_payload:
            complete_chat_query(pending_payload)

        if st.button(labels["clear"], key="clear_chat_btn"):
            st.session_state.chat_history = []
            st.session_state.pending_chat_query = None
            st.rerun()


def main() -> None:
    init_session_state()
    inject_css()

    try:
        ensure_vector_store()
    except Exception:
        logger.exception("Knowledge base setup failed")
        st.error("Knowledge base setup failed. Please check the server logs.")
        st.stop()

    render_dashboard()
    render_chat_launcher()
    render_chat_window()


main()
