# llm_api.py
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
import hashlib
import os
import re
import json
import time
import logging
import threading
from typing import List, Tuple

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Condition checks decide which branch of the tree a case follows, so they must be
# repeatable: the same summary should always take the same path. Summaries feed
# those checks, so they are generated deterministically too.
DETERMINISTIC_CONFIG = {"temperature": 0.0, "top_p": 1.0}
# Merging profiles and proposing differences are generative tasks; allow a little variety.
GENERATIVE_CONFIG = {"temperature": 0.4, "top_p": 1.0}

# How often to retry a rate-limited (HTTP 429) call, and the longest single wait.
MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "2"))
MAX_RETRY_WAIT = float(os.environ.get("LLM_MAX_RETRY_WAIT", "30"))

# Separator the summary prompt places between the clinical summary and the therapy feedback
FEEDBACK_SEPARATOR = "===THERAPY QUALITY FEEDBACK==="

# Prompt Files
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(CURRENT_DIR, '..', 'prompts')

CHECK_PROMPT_FILE = os.path.join(PROMPTS_DIR, 'prompt_condition.txt')
DIFF_PROMPT_FILE = os.path.join(PROMPTS_DIR, 'prompt_differentiate.txt')
SUMMARY_PROMPT_FILE = os.path.join(PROMPTS_DIR, 'prompt_summary.txt')
MERGE_PROMPT_FILE = os.path.join(PROMPTS_DIR, 'prompt_merge.txt')

# Pre-computed answers shipped with the demo (built by scripts/build_seed_tree.py)
CACHE_FILE = os.path.join(CURRENT_DIR, '..', 'demo', 'llm_cache.json')


class LLMUnavailableError(RuntimeError):
    """Gemini could not answer (rate limit, quota, network). Callers must not guess an answer."""


def _setting(name: str):
    """
    Reads a top-level setting, preferring Streamlit secrets over environment variables.
    Secrets are read on every call, so editing them in Streamlit Cloud takes effect without a redeploy.
    """
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass  # Not running under Streamlit, or no secrets file
    value = os.environ.get(name, "").strip()
    return value or None


def model_name() -> str:
    """The Gemini model to use, set by the top-level GEMINI_MODEL secret. There is no built-in default."""
    name = _setting("GEMINI_MODEL")
    if not name:
        raise LLMUnavailableError(
            'No Gemini model is configured. Add a top-level line like GEMINI_MODEL = "<model name>" '
            "to the app's secrets."
        )
    return name


# --- Setup ---
api_key = _setting("API_KEY") or _setting("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError(
        "Gemini API key not found. Set a top-level API_KEY in .streamlit/secrets.toml "
        "(or the API_KEY environment variable)."
    )
client = genai.Client(api_key=api_key)


# --- Load Prompts ---
def load_prompt(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error loading {filename}: {e}")
        return ""


CHECK_TEMPLATE = load_prompt(CHECK_PROMPT_FILE)
DIFF_TEMPLATE = load_prompt(DIFF_PROMPT_FILE)
SUMMARY_TEMPLATE = load_prompt(SUMMARY_PROMPT_FILE)
MERGE_TEMPLATE = load_prompt(MERGE_PROMPT_FILE)


# --- Response cache ---
# Identical prompts get identical answers, so answers are cached by prompt. The module is
# imported once per server process, so this cache is shared by every session: each sample
# case only costs API calls the first time anyone runs it. It is pre-filled from CACHE_FILE.
_cache: dict = {}
_cache_lock = threading.Lock()


def _load_cache_file():
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            _cache.update(json.load(f))
    except FileNotFoundError:
        pass
    except Exception as e:
        logging.error(f"Could not load LLM cache: {e}")


_load_cache_file()


_used_keys: set = set()


def export_cache(used_only: bool = False) -> dict:
    """Returns cached answers; with used_only, just those read or written since startup."""
    with _cache_lock:
        if used_only:
            return {k: v for k, v in _cache.items() if k in _used_keys}
        return dict(_cache)


def _cache_key(kind: str, prompt: str) -> str:
    return kind + ":" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _retry_delay(err: Exception) -> float:
    text = str(err)
    match = re.search(r"retry in ([\d.]+)s", text) or re.search(r"retryDelay['\"]?:\s*['\"]([\d.]+)s", text)
    return float(match.group(1)) + 1 if match else 15.0


def _generate(kind: str, prompt: str, config: dict) -> str:
    """Calls Gemini with caching and rate-limit retries. Raises LLMUnavailableError on failure."""
    key = _cache_key(kind, prompt)
    with _cache_lock:
        _used_keys.add(key)
        if key in _cache:
            return _cache[key]

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model_name(),
                contents=prompt,
                config=types.GenerateContentConfig(**config),
            )
            text = (response.text or "").strip()
            if not text:
                raise LLMUnavailableError("The AI service returned an empty answer. Please try again.")
            with _cache_lock:
                _cache[key] = text
            return text
        except LLMUnavailableError:
            raise
        except genai_errors.APIError as e:
            if e.code not in (429, 503):
                raise LLMUnavailableError(f"The AI service returned an error: {e}") from e
            if "PerDay" in str(e):
                # Waiting won't help with a daily quota
                raise LLMUnavailableError(
                    "This demo has used up today's AI allowance. The sample cases still work; "
                    "new AI calls will be available again tomorrow."
                ) from e
            wait = _retry_delay(e)
            if attempt < MAX_RETRIES and wait <= MAX_RETRY_WAIT:
                logging.warning(f"Gemini busy ({kind}); retrying in {wait:.0f}s")
                time.sleep(wait)
                continue
            raise LLMUnavailableError(
                "The AI service is at its request limit right now. Please wait a minute and try again."
            ) from e
        except Exception as e:
            raise LLMUnavailableError(f"The AI service returned an error: {e}") from e
    raise LLMUnavailableError("The AI service did not respond.")


def _is_true(response_text: str) -> bool:
    """
    Interprets a TRUE/FALSE model answer. Some models explain their reasoning before answering,
    so the verdict is the last standalone TRUE or FALSE in the response.
    """
    verdicts = re.findall(r"\b(TRUE|FALSE)\b", response_text.upper())
    return bool(verdicts) and verdicts[-1] == "TRUE"


def split_summary(text: str) -> Tuple[str, str]:
    """Splits the summary model's output into (clinical summary, therapy quality feedback)."""
    if FEEDBACK_SEPARATOR in text:
        clinical, feedback = text.split(FEEDBACK_SEPARATOR, 1)
        return clinical.strip(), feedback.strip()
    return text.strip(), ""


# --- Core Functions ---

def llm_generate_summary(transcript_text: str) -> str:
    """
    Generates the full summary output for a transcript: the clinical summary, then
    FEEDBACK_SEPARATOR, then the therapy quality feedback. Use split_summary() on it.
    """
    prompt = SUMMARY_TEMPLATE.format(transcript_content=transcript_text)
    return _generate("summary", prompt, DETERMINISTIC_CONFIG)


def llm_check_condition(summary_text: str, condition_string: str) -> bool:
    """Checks if a SUMMARY satisfies a condition."""
    prompt = CHECK_TEMPLATE.format(
        summary_content=summary_text,
        condition_string=condition_string
    )
    verdict = _is_true(_generate("check", prompt, DETERMINISTIC_CONFIG))
    logging.info(f"LLM Check: '{condition_string}' -> {'TRUE' if verdict else 'FALSE'}")
    return verdict


def llm_merge_summaries(old_summary: str, new_summary: str) -> str:
    """Merges a new patient profile into an existing consolidated group profile."""
    prompt = MERGE_TEMPLATE.format(
        old_summary=old_summary,
        new_summary=new_summary
    )
    return _generate("merge", prompt, GENERATIVE_CONFIG)


def llm_get_differentiating_conditions(new_summary: str, ref_summary: str) -> List[str]:
    """Compares NEW summary vs REFERENCE summary to find differences."""
    prompt = DIFF_TEMPLATE.format(
        summary_new=new_summary,
        summary_ref=ref_summary if ref_summary else "None"
    )
    text = _generate("diff", prompt, {**GENERATIVE_CONFIG, "response_mime_type": "application/json"})
    try:
        # Clean markdown code blocks if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
        conditions = json.loads(text)
    except ValueError as e:
        logging.error(f"Diff extraction returned invalid JSON: {e}")
        return []
    if not isinstance(conditions, list):
        return []
    # De-duplicate while keeping order; drop anything that isn't a non-empty string
    return list(dict.fromkeys(str(c).strip() for c in conditions if str(c).strip()))


def llm_verify_manual_condition(summary_text: str, condition_string: str) -> bool:
    """
    Verifies if a manually entered condition is TRUE for a given summary.
    Used to check the condition holds for the new case and NOT for the reference case.
    """
    return llm_check_condition(summary_text, condition_string)
