"""
Builds the demo's seed knowledge base and pre-computes the AI answers the sample walkthrough needs.

    python scripts/build_seed_tree.py

The Gemini key and model are read from app/.streamlit/secrets.toml (top-level API_KEY and GEMINI_MODEL)
unless those environment variables are already set.
It waits out free-tier rate limits, so a full run takes a few minutes.

Outputs:
  demo/seed_tree.json   the seed SCRDR tree; each rule's cornerstone is the clinical summary of the case that created it
  demo/llm_cache.json   cached Gemini answers (summaries, condition checks, suggested differences, merges) for the
                        bundled samples, so visitors' first steps don't depend on the API quota
"""
import json
import os
import sys
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(ROOT, "demo")
SAMPLES_DIR = os.path.join(DEMO_DIR, "samples")
SEED_FILE = os.path.join(DEMO_DIR, "seed_tree.json")
CACHE_FILE = os.path.join(DEMO_DIR, "llm_cache.json")

# API key and model come from the same secrets file the app uses (environment variables win if set)
with open(os.path.join(ROOT, "app", ".streamlit", "secrets.toml"), "rb") as f:
    _secrets = tomllib.load(f)
for _name in ("API_KEY", "GEMINI_MODEL"):
    if not os.environ.get(_name) and _secrets.get(_name):
        os.environ[_name] = str(_secrets[_name])
if not os.environ.get("GEMINI_MODEL"):
    sys.exit('Add a top-level GEMINI_MODEL = "<model name>" line to app/.streamlit/secrets.toml first.')
# Free tier allows ~5 requests/minute: be patient rather than fail
os.environ.setdefault("LLM_MAX_RETRIES", "10")
os.environ.setdefault("LLM_MAX_RETRY_WAIT", "70")

sys.path.insert(0, os.path.join(ROOT, "app"))
import llm_api  # noqa: E402
from llm_api import (  # noqa: E402
    llm_generate_summary, llm_check_condition, llm_get_differentiating_conditions,
    llm_merge_summaries, split_summary,
)
from transcripts import json_to_text  # noqa: E402
from rdr_engine import engine_from_dict  # noqa: E402

# The seed tree. "case" names the transcript whose clinical summary becomes the rule's cornerstone.
SEED_SPEC = {
    "conditions": ["Client drinks above low-risk guidelines, through heavy weekly intake or binge-drinking episodes"],
    "conclusion": "Hazardous drinking: brief motivational intervention",
    "case": "transcript_1_reducing_alcohol_consumption.json",
    "true_branch": {
        "conditions": ["Client has had an injury or a physical health problem linked to their drinking"],
        "conclusion": "Harmful drinking: brief intervention plus medical follow-up",
        "case": "transcript_3_reducing_alcohol_consumption.json",
    },
    "false_branch": {
        "conditions": ["Client is under the legal drinking age"],
        "conclusion": "Low-risk underage drinking: prevention and education",
        "case": "transcript_6_reducing_alcohol_consumption.json",
    },
}

# Cases visitors can pick in the app (keep in sync with SAMPLE_CASES in app/app.py)
DEMO_SAMPLES = [
    "transcript_2_reducing_alcohol_consumption.json",
    "transcript_0_reducing_alcohol_consumption.json",
    "transcript_7_reducing_alcohol_consumption.json",
    "transcript_5_reducing_alcohol_consumption.json",
]

# Conditions the app suggests as examples; pre-checking them keeps manual-rule saving fast
EXAMPLE_MANUAL_CONDITIONS = [
    "Client is pregnant",
    "Client drinks to cope with low mood or depression",
]


def clinical_summary(name: str) -> str:
    with open(os.path.join(SAMPLES_DIR, name), encoding="utf-8") as f:
        text = json_to_text(json.load(f))
    print(f"  summary: {name}", flush=True)
    clinical, _ = split_summary(llm_generate_summary(text))
    if not clinical:
        raise RuntimeError(f"Empty summary for {name}")
    return clinical


def build_node(spec: dict) -> dict:
    return {
        "conditions": spec["conditions"],
        "conclusion": spec["conclusion"],
        "cornerstone_summary": clinical_summary(spec["case"]),
        "source": spec["case"],
        "true_branch": build_node(spec["true_branch"]) if spec.get("true_branch") else None,
        "false_branch": build_node(spec["false_branch"]) if spec.get("false_branch") else None,
    }


def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(llm_api.export_cache(), f, indent=1, ensure_ascii=False, sort_keys=True)


def main():
    print("Building seed tree")
    seed = {"root": build_node(SEED_SPEC)}
    with open(SEED_FILE, "w", encoding="utf-8") as f:
        json.dump(seed, f, indent=2, ensure_ascii=False)
    save_cache()

    print("\nRunning samples through the tree")
    for name in sorted(n for n in os.listdir(SAMPLES_DIR) if n.endswith(".json")):
        engine = engine_from_dict(seed)  # fresh tree per case, as a new visitor would have
        summary = clinical_summary(name)
        n1, n2 = engine.interpret(summary)
        path = " -> ".join(f"{n.vertex.rule.conclusions[:28]}={'T' if ok else 'F'}" for n, ok in engine.last_trace)
        verdict = n2.vertex.rule.conclusions if n2 else "(no conclusion)"
        print(f"{name}\n  path: {path}\n  conclusion: {verdict}", flush=True)

        if name in DEMO_SAMPLES:
            ref = n2.vertex.summary if n2 else ""
            diffs = llm_get_differentiating_conditions(summary, ref)
            print(f"  suggested differences: {diffs}")
            if n2:
                llm_merge_summaries(ref, summary)
            for cond in EXAMPLE_MANUAL_CONDITIONS:
                on_case = llm_check_condition(summary, cond)
                on_ref = llm_check_condition(ref, cond) if ref else None
                print(f"  '{cond}': case={on_case} cornerstone={on_ref}")
        save_cache()

    print(f"\nWrote {SEED_FILE} and {CACHE_FILE} ({len(llm_api.export_cache())} cached answers)")


if __name__ == "__main__":
    try:
        main()
    finally:
        save_cache()  # keep every answer already paid for, so a re-run resumes where this one stopped
