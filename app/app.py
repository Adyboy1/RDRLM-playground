#app.py
import streamlit as st
import os
import json
import sys
import pickle
import re
import time
import graphviz
import textwrap

# --- PATH CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# Define storage paths locally for UI access
STORAGE_DIR = os.path.join(CURRENT_DIR, '..', 'storage')
TREE_STORAGE_FILE = os.path.join(STORAGE_DIR, "rdr_tree_summary.pkl")
LOG_FILE = os.path.join(STORAGE_DIR, "rdr_event_log.csv")

# Demo assets: seed tree and sample transcripts (cached AI answers are loaded by llm_api)
DEMO_DIR = os.path.join(CURRENT_DIR, '..', 'demo')
SEED_FILE = os.path.join(DEMO_DIR, "seed_tree.json")
SAMPLES_DIR = os.path.join(DEMO_DIR, "samples")

if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

from rdr_engine import RDREngine, Node, Rule, Vertex, engine_from_dict
from transcripts import file_to_text, json_to_text, SUPPORTED_TYPES
from llm_api import (
    llm_get_differentiating_conditions,
    llm_generate_summary,
    llm_merge_summaries,
    llm_verify_manual_condition,
    split_summary,
    LLMUnavailableError,
)

# --- PAGE SETUP ---
st.set_page_config(page_title="RDR Clinical Engine", page_icon="🧠", layout="wide")


def read_secret(section, key, default):
    try:
        return st.secrets[section][key]
    except Exception:
        return default


# Demo mode (the default) gives every visitor a private copy of the seed tree and
# never writes to Firestore or the shared files on disk.
# Set `demo_mode = false` under [general] in secrets.toml for the persistent, multi-tenant mode.
DEMO_MODE = bool(read_secret("general", "demo_mode", True))
ORG_ID = read_secret("general", "app_id", "default_rdr_app")

# Per-session cap on actions that call Gemini, so the shared API quota survives a busy day
MAX_LLM_ACTIONS = 25
MAX_TRANSCRIPT_CHARS = 60_000

# Sample cases offered in demo mode, in display order
SAMPLE_CASES = [
    {
        "file": "transcript_2_reducing_alcohol_consumption.json",
        "label": "University student who sprained her ankle on a pub crawl",
        "hint": "The tree should already get this one right. Try **Agree**, which merges the case into the rule's reference profile.",
    },
    {
        "file": "transcript_0_reducing_alcohol_consumption.json",
        "label": "Evening wine drinking to cope with low mood",
        "hint": "The general harm rule fires, but this client drinks to cope with depression. Try **Disagree** and add a more specific rule, e.g. from the AI's suggested differences.",
    },
    {
        "file": "transcript_7_reducing_alcohol_consumption.json",
        "label": "Drinking while pregnant",
        "hint": "The harm rule fires again, but pregnancy calls for a different plan. **Disagree** and type the condition *Client is pregnant*; it becomes an exception to the harm rule.",
    },
    {
        "file": "transcript_5_reducing_alcohol_consumption.json",
        "label": "Around 20 drinks a week, injured leaving a bar",
        "hint": "Another case for the harm rule. Analyze it after you've added your own rules to see whether they interfere.",
    },
]

if DEMO_MODE:
    st.title("🧠 Clinical RDR Knowledge Base")
    st.caption("Interactive demo: an interpretable decision tree that clinicians grow by correcting it, powered by Ripple Down Rules and Gemini.")
else:
    import backend
    st.title(f"🧠 Clinical RDR Knowledge Base ({ORG_ID})")


# --- DEMO ASSETS ---

@st.cache_data
def load_seed_tree():
    with open(SEED_FILE, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_sample(file_name):
    """Returns (raw JSON text, dialogue text) for a bundled sample."""
    with open(os.path.join(SAMPLES_DIR, file_name), encoding="utf-8") as f:
        raw = f.read()
    return raw, json_to_text(json.loads(raw))


def new_demo_engine():
    return engine_from_dict(load_seed_tree(), in_memory_log=True)


# --- PERSISTENCE FUNCTIONS (full mode only) ---

def sync_from_cloud():
    """Attempts to pull latest state from Cloud to Local."""
    if 'cloud_synced' not in st.session_state:
        cloud_engine = backend.load_engine_from_cloud(app_id=ORG_ID)
        if cloud_engine:
            st.session_state.engine = cloud_engine
            with open(TREE_STORAGE_FILE, "wb") as f:
                pickle.dump(cloud_engine, f)
            st.toast(f"✅ Knowledge Base synced from Cloud ({ORG_ID})")

        cloud_logs = backend.load_logs_from_cloud(app_id=ORG_ID)
        if cloud_logs:
            with open(LOG_FILE, "w", encoding='utf-8') as f:
                f.write(cloud_logs)
            st.toast("✅ Event Logs synced from Cloud")

        st.session_state.cloud_synced = True


def sync_to_cloud():
    """Pushes current local state to Cloud."""
    if 'engine' in st.session_state:
        backend.save_engine_to_cloud(st.session_state.engine, app_id=ORG_ID)

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding='utf-8') as f:
            backend.save_logs_to_cloud(f.read(), app_id=ORG_ID)

    st.toast("☁️ State persisted to Cloud", icon="💾")


def persist():
    """Saves the knowledge base. In demo mode the session state is the only store."""
    if DEMO_MODE:
        return
    with open(TREE_STORAGE_FILE, "wb") as f:
        pickle.dump(st.session_state.engine, f)
    sync_to_cloud()


# --- SESSION STATE INITIALIZATION ---

if DEMO_MODE:
    if 'engine' not in st.session_state:
        st.session_state.engine = new_demo_engine()
else:
    sync_from_cloud()
    if 'engine' not in st.session_state:
        if os.path.exists(TREE_STORAGE_FILE):
            try:
                with open(TREE_STORAGE_FILE, "rb") as f:
                    st.session_state.engine = pickle.load(f)
            except Exception:
                st.session_state.engine = RDREngine()
        else:
            st.session_state.engine = RDREngine()

if not hasattr(st.session_state.engine, 'history'):
    st.session_state.engine.history = []

for key, default in {
    "current_summary": None,
    "current_feedback": "",
    "interpretation_result": None,
    "trace": [],
    "diff_conditions": [],
    "current_source_file": "Unknown",
    "show_revision_form": False,
    "llm_actions": 0,
    "case_id": 0,
    "flash": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def reset_case_state():
    """Clears everything tied to the case currently on screen."""
    st.session_state.current_summary = None
    st.session_state.current_feedback = ""
    st.session_state.interpretation_result = None
    st.session_state.trace = []
    st.session_state.diff_conditions = []
    st.session_state.show_revision_form = False


def use_llm_budget():
    """Counts one Gemini-backed action. Returns False (and warns) once the session cap is hit."""
    if DEMO_MODE and st.session_state.llm_actions >= MAX_LLM_ACTIONS:
        st.warning("This demo session has reached its limit of AI actions. Reload the page to start a fresh session.")
        return False
    st.session_state.llm_actions += 1
    return True


def flash(message, kind="success"):
    """Shows a message after the next rerun."""
    st.session_state.flash = (kind, message)


# --- VISUALIZATION LOGIC ---

def number_rules(root):
    """Assigns rule numbers R1, R2, ... in the order rules appear (pre-order)."""
    numbers = {}

    def walk(node):
        if node is None:
            return
        numbers[id(node)] = len(numbers) + 1
        walk(node.right)
        walk(node.left)

    walk(root)
    return numbers


def pretty(text):
    """Turns the model's '[Paragraph 2: Symptomatology]' markers into bold headings for display."""
    return re.sub(r"^\s*\[Paragraph \d+:\s*(.+?)\]\s*$", r"**\1**\n", text or "", flags=re.MULTILINE)


def parse_conditions(raw):
    try:
        conds = json.loads(raw)
        return conds if isinstance(conds, list) else [str(conds)]
    except (ValueError, TypeError):
        return raw if isinstance(raw, list) else [raw]


def build_graph(root, trace=None, conclusion_node=None):
    numbers = number_rules(root)
    outcome = {id(n): ok for n, ok in (trace or [])}

    dot = graphviz.Digraph()
    dot.attr(rankdir='TB', nodesep='0.5', ranksep='0.8', bgcolor='transparent')
    dot.attr('node', shape='Mrecord', style='filled', fillcolor='#F0F2F6', color='#9AA0A6',
             fontname='Arial', fontsize='10', margin='0.1')
    dot.attr('edge', fontname='Arial', fontsize='9')

    if not root:
        dot.node("empty", "Empty Tree", style='dashed')
        return dot

    wrapper = textwrap.TextWrapper(width=28, break_long_words=False, replace_whitespace=False)

    def esc(text):
        for ch in '{}|<>"':
            text = text.replace(ch, '\\' + ch)
        return text

    def format_label(node):
        conds = parse_conditions(node.vertex.rule.conditions)
        cond_lines = []
        for c in conds:
            cond_lines.extend(wrapper.wrap(f"• {c}"))
        concl_lines = wrapper.wrap(node.vertex.rule.conclusions)
        rule_no = numbers[id(node)]
        return (f"{{ R{rule_no}  IF:\\l" + "\\l".join(esc(l) for l in cond_lines) + "\\l"
                f" | THEN:\\l" + "\\l".join(esc(l) for l in concl_lines) + "\\l }}")

    def add_nodes_recursive(node, node_id):
        attrs = {}
        if id(node) in outcome:
            attrs["fillcolor"] = "#C8E6C9" if outcome[id(node)] else "#FFCDD2"
        if conclusion_node is not None and node is conclusion_node:
            attrs["penwidth"] = "3"
            attrs["color"] = "#2E7D32"
        dot.node(node_id, format_label(node), **attrs)

        if node.left:
            left_id = f"{node_id}L"
            add_nodes_recursive(node.left, left_id)
            dot.edge(node_id, left_id, label=" False", color="#D32F2F", fontcolor="#D32F2F")

        if node.right:
            right_id = f"{node_id}R"
            add_nodes_recursive(node.right, right_id)
            dot.edge(node_id, right_id, label=" True", color="#388E3C", fontcolor="#388E3C")

    add_nodes_recursive(root, "root")
    return dot


engine = st.session_state.engine

# --- SIDEBAR ---
with st.sidebar:
    if DEMO_MODE:
        st.info("**Demo mode.** You have your own copy of the knowledge base. "
                "Nothing you do is saved or seen by anyone else; reloading the page resets it.")

    st.header("Tree Controls")

    if st.button("↩️ Undo Last Rule"):
        success, msg = engine.undo_last_addition()
        if success:
            persist()
            reset_case_state()
            flash(msg)
            st.rerun()
        else:
            st.error("No rules of yours to undo." if DEMO_MODE else msg)

    if DEMO_MODE:
        if st.button("🔄 Reset demo tree"):
            st.session_state.engine = new_demo_engine()
            reset_case_state()
            flash("Knowledge base reset to the starting tree.")
            st.rerun()
    else:
        if "confirm_flush" not in st.session_state:
            st.session_state.confirm_flush = False

        if st.button("🗑️ Flush Entire Tree"):
            st.session_state.confirm_flush = True

        if st.session_state.confirm_flush:
            st.error(f"⚠️ **CONFIRM DELETION: {ORG_ID}** ⚠️")
            st.write("This will permanently delete this organization's knowledge base.")

            col_conf1, col_conf2 = st.columns(2)
            with col_conf1:
                if st.button("✅ YES, DELETE"):
                    engine.flush_tree()
                    try:
                        db = backend.init_firebase()
                        db.collection("artifacts").document(ORG_ID).collection("public").document("tree_data").delete()
                        db.collection("artifacts").document(ORG_ID).collection("public").document("event_logs").delete()
                    except Exception:
                        pass
                    st.session_state.confirm_flush = False
                    reset_case_state()
                    flash("Tree flushed and file deleted.")
                    st.rerun()
            with col_conf2:
                if st.button("❌ NO, CANCEL"):
                    st.session_state.confirm_flush = False
                    st.rerun()

    st.markdown("---")
    st.header("Logs")
    download_data, source_label = None, ""
    if DEMO_MODE:
        if engine.event_rows:
            download_data, source_label = engine.events_csv(), "(this session)"
    else:
        cloud_log_content = backend.load_logs_from_cloud(app_id=ORG_ID)
        if cloud_log_content:
            download_data, source_label = cloud_log_content, "(Cloud)"
        elif os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding='utf-8') as f:
                download_data = f.read()
            source_label = "(Local Cache)"

    if download_data:
        st.download_button(
            label=f"📥 Download Event Log {source_label}",
            data=download_data,
            file_name=f"{'demo' if DEMO_MODE else ORG_ID}_log_{int(time.time())}.csv",
            mime="text/csv"
        )
    else:
        st.caption("No events yet. Agreeing with a conclusion or adding a rule logs an event.")

    if DEMO_MODE:
        st.markdown("---")
        # Filled in at the end of the script, after this run's actions have been counted
        usage_slot = st.empty()
        st.caption("Transcripts: AnnoMI, a public dataset of counselling demonstration videos. "
                   "No real patient data. Educational demo, not for clinical use.")

# --- FLASH MESSAGE ---
if st.session_state.flash:
    kind, message = st.session_state.flash
    getattr(st, kind)(message)
    st.session_state.flash = None

# --- INTRO ---
if DEMO_MODE:
    with st.expander("ℹ️ How it works (30-second read)", expanded=st.session_state.current_summary is None):
        st.markdown(
            """
**The problem.** AI tools that assess therapy sessions are usually black boxes. A clinician can't see *why* a case was
classified a certain way, or fix a wrong answer without retraining a model.

**The approach.** This app keeps the reasoning in a **Single Classification Ripple Down Rules (SCRDR)** tree that
experts can read and correct:

1. **Summarise.** Gemini turns a raw counselling transcript into a structured clinical summary.
2. **Classify.** The summary walks down the tree. Each rule's condition is plain English, and Gemini judges whether it holds:
   go down the **True** branch if it does, the **False** branch if not. The answer is the conclusion of the *last rule that was true*,
   so every result traces back to specific, human-written rules.
3. **Correct.** If the clinician disagrees, they give the right conclusion and pick conditions that separate this case from
   the case that created the rule that misfired (its *cornerstone*). Gemini suggests candidates. The new rule is attached
   exactly where the reasoning went wrong, so it can't change answers for earlier cases that reached other branches.

**Try it.** Pick a sample case below, analyze it, then agree or disagree. Watch the tree grow at the bottom of the page.
"""
        )

# --- 1. INPUT ---
st.subheader("1. Choose a transcript")

transcript_text = None
source_name = "Unknown"
selected_hint = None

if DEMO_MODE:
    input_mode = st.radio("Transcript source", ["Use a sample case", "Upload your own"],
                          horizontal=True, label_visibility="collapsed")
else:
    input_mode = "Upload your own"

if input_mode == "Use a sample case":
    labels = [c["label"] for c in SAMPLE_CASES]
    choice = st.selectbox("Sample case", labels)
    case = SAMPLE_CASES[labels.index(choice)]
    raw_json, transcript_text = load_sample(case["file"])
    source_name = case["file"]
    selected_hint = case["hint"]
    st.caption(f"💡 {case['hint']}")

    col_dl, col_prev = st.columns([1, 3])
    with col_dl:
        st.download_button("📥 Download transcript (JSON)", data=raw_json,
                           file_name=case["file"], mime="application/json")
    with col_prev:
        with st.expander("Preview the dialogue"):
            st.text(transcript_text)
else:
    uploaded_file = st.file_uploader(
        "Upload a transcript (.json in AnnoMI format, .docx, or .txt)", type=SUPPORTED_TYPES)
    if DEMO_MODE:
        st.caption("Tip: download a sample above, edit it, and upload it to see how the tree reacts. "
                   "Please don't upload real patient data.")
    if uploaded_file:
        source_name = uploaded_file.name
        try:
            transcript_text = file_to_text(uploaded_file.name, uploaded_file.getvalue())
        except Exception as e:
            st.error(f"Error reading file: {e}")
        if transcript_text and len(transcript_text) > MAX_TRANSCRIPT_CHARS:
            st.error(f"That transcript is too long for the demo ({len(transcript_text):,} characters; "
                     f"limit {MAX_TRANSCRIPT_CHARS:,}).")
            transcript_text = None

if st.button("▶️ Analyze Case", type="primary", disabled=not transcript_text):
    if use_llm_budget():
        reset_case_state()
        st.session_state.current_source_file = source_name
        st.session_state.case_id += 1

        try:
            with st.spinner("Generating clinical summary (up to a minute for a new transcript)..."):
                summary, feedback = split_summary(llm_generate_summary(transcript_text))
            if not summary:
                st.error("The summary came back empty. Please try again.")
            else:
                with st.spinner("Walking the summary down the rule tree..."):
                    n1, n2 = engine.interpret(summary)
                st.session_state.current_summary = summary
                st.session_state.current_feedback = feedback
                st.session_state.interpretation_result = (n1, n2)
                st.session_state.trace = list(getattr(engine, "last_trace", []))
        except LLMUnavailableError as e:
            st.error(f"⏳ {e}")

# --- 2. RESULTS AREA ---
if st.session_state.get("current_summary") and st.session_state.get("interpretation_result"):
    st.divider()
    st.subheader("2. Review the system's conclusion")
    n1, n2 = st.session_state.interpretation_result
    numbers = number_rules(engine.root)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📄 Clinical Summary")
        st.caption("Generated by Gemini from the transcript. This is what the rules are checked against.")
        with st.container(height=420):
            st.markdown(pretty(st.session_state.current_summary))
        if st.session_state.get("current_feedback"):
            with st.expander("🎓 Bonus: feedback on the counsellor's skills"):
                st.caption("Generated alongside the summary using a counselling-skills rubric. Not used by the rules.")
                st.markdown(pretty(st.session_state.current_feedback))

    with col_b:
        st.markdown("#### 🤖 System Conclusion")
        if n2:
            st.success(f"**{n2.vertex.rule.conclusions}**")
            st.markdown(f"Given by rule **R{numbers.get(id(n2), '?')}**, which fired because:")
            for c in parse_conditions(n2.vertex.rule.conditions):
                st.write(f"• {c}")
        else:
            st.warning("**No conclusion.** No rule's conditions matched this case.")

        if st.session_state.trace:
            st.markdown("**How the tree reasoned:**")
            steps = []
            for node, ok in st.session_state.trace:
                rule_no = numbers.get(id(node), "?")
                steps.append(f"R{rule_no} {'✅ true' if ok else '❌ false'}")
            st.markdown(" → ".join(steps))
            st.caption("Rules evaluated on this case are shaded in the knowledge base below.")

        st.markdown("#### Clinician Review")
        col_agree, col_disagree = st.columns(2)
        with col_agree:
            if st.button("✅ Agree", width="stretch", help="The conclusion is correct"):
                if not n2:
                    st.info("There's no conclusion to agree with. Use **Disagree** to teach the tree a rule for this case.")
                elif use_llm_budget():
                    try:
                        with st.spinner("Merging this case into the rule's reference profile..."):
                            merged = llm_merge_summaries(n2.vertex.summary, st.session_state.current_summary)
                    except LLMUnavailableError as e:
                        st.error(f"⏳ {e}")
                    else:
                        n2.vertex.summary = merged
                        engine.log_merge(st.session_state.current_source_file, n1, n2)
                        persist()
                        reset_case_state()
                        flash(f"Agreed. This case was merged into the reference profile of rule "
                              f"R{numbers.get(id(n2), '?')} (see Rule details below).")
                        st.rerun()
        with col_disagree:
            if st.button("❌ Disagree", width="stretch", help="The conclusion is wrong: teach the tree a new rule"):
                st.session_state.show_revision_form = True

# --- 3. REVISION AREA ---
if st.session_state.get("show_revision_form") and st.session_state.get("interpretation_result"):
    st.divider()
    st.subheader("3. Teach the tree a new rule")
    n1, n2 = st.session_state.interpretation_result
    numbers = number_rules(engine.root)
    ref_sum = n2.vertex.summary if n2 else ""

    if engine.root is None:
        placement = "This will become the first rule (the root) of the tree."
    elif n1 is n2:
        placement = (f"The new rule will be an **exception** to R{numbers.get(id(n1), '?')} (its True branch): "
                     f"it is checked only for cases where R{numbers.get(id(n1), '?')} fired.")
    else:
        placement = (f"The new rule will go on the **False branch** of R{numbers.get(id(n1), '?')}, the last rule checked, "
                     f"so it is tried only for cases that reached that point.")
    st.markdown(placement)
    if n2:
        st.caption(f"Its conditions must be true for this case and false for the cornerstone case of "
                   f"R{numbers.get(id(n2), '?')}, the rule that gave the conclusion you disagree with.")

    new_conclusion = st.text_input("What is the CORRECT conclusion?")

    if st.button("🔍 Find Differences (Ask AI)"):
        if use_llm_budget():
            try:
                with st.spinner("Comparing this case with the cornerstone case..."):
                    st.session_state.diff_conditions = llm_get_differentiating_conditions(
                        st.session_state.current_summary, ref_sum)
            except LLMUnavailableError as e:
                st.error(f"⏳ {e}")
            else:
                if not st.session_state.diff_conditions:
                    st.warning("The AI didn't return any conditions. Try again or enter your own below.")

    final_conditions = []
    if st.session_state.get("diff_conditions"):
        st.write("Select distinguishing conditions:")
        for i, cond in enumerate(st.session_state.diff_conditions):
            if st.checkbox(cond, key=f"diff_{st.session_state.case_id}_{i}"):
                final_conditions.append(cond)

    manual_conds_text = st.text_area(
        "Or enter your own conditions (one per line):",
        height=100,
        placeholder="Example:\nClient is pregnant\nClient drinks to cope with low mood or depression",
        key=f"manual_{st.session_state.case_id}",
    )

    manual_conditions = []
    if manual_conds_text.strip():
        manual_conditions = list(dict.fromkeys(
            c.strip() for c in manual_conds_text.split("\n") if c.strip()))
        final_conditions.extend(manual_conditions)

    if st.button("💾 Save New Rule", type="primary"):
        if not new_conclusion or not final_conditions:
            st.error("Please provide a conclusion and at least one condition.")
        elif use_llm_budget():
            condition_type = "AI_GENERATED"
            verified = True
            if manual_conditions:
                condition_type = "MANUAL" if len(manual_conditions) == len(final_conditions) else "MIXED"
                try:
                    with st.spinner("Checking your conditions against this case and the cornerstone case..."):
                        for cond in manual_conditions:
                            if not llm_verify_manual_condition(st.session_state.current_summary, cond):
                                st.error(f"⚠️ The condition '*{cond}*' is **FALSE** for the current case.")
                                verified = False
                                break
                            if ref_sum and llm_verify_manual_condition(ref_sum, cond):
                                st.error(f"⚠️ **Doesn't differentiate:** '*{cond}*' is ALSO TRUE for the cornerstone "
                                         f"case, so it can't separate the two.")
                                verified = False
                                break
                except LLMUnavailableError as e:
                    st.error(f"⏳ {e}")
                    verified = False

            if verified:
                new_node = Node(Vertex(Rule(json.dumps(final_conditions), new_conclusion),
                                       st.session_state.current_summary))
                source_file = st.session_state.current_source_file

                if engine.root is None:
                    engine.add_node_to_tree(None, new_node, "ROOT", source_file, n1, n2, condition_type)
                elif n1 is n2:
                    engine.add_node_to_tree(n1, new_node, "RIGHT", source_file, n1, n2, condition_type)
                else:
                    engine.add_node_to_tree(n1, new_node, "LEFT", source_file, n1, n2, condition_type)

                persist()
                reset_case_state()
                flash(f"New rule R{number_rules(engine.root).get(id(new_node), '?')} added. "
                      "Analyze the same case again to watch it fire.")
                st.rerun()

# --- 4. KNOWLEDGE BASE ---
st.divider()
st.subheader("🌳 Knowledge base")
if engine.root:
    trace = st.session_state.trace if st.session_state.get("interpretation_result") else []
    conclusion_node = st.session_state.interpretation_result[1] if st.session_state.get("interpretation_result") else None
    st.graphviz_chart(build_graph(engine.root, trace, conclusion_node), width="stretch")
    st.caption("Read top-down: if a rule's conditions hold, follow **True**, otherwise **False**. "
               "Green/red shading shows the rules checked for the current case; the bold outline marks the rule that gave the conclusion.")

    with st.expander("Rule details and cornerstone cases"):
        numbers = number_rules(engine.root)

        def collect(node, out):
            if node:
                out.append(node)
                collect(node.right, out)
                collect(node.left, out)
            return out

        for node in collect(engine.root, []):
            st.markdown(f"**R{numbers[id(node)]}: {node.vertex.rule.conclusions}**")
            st.markdown("IF " + " AND ".join(f"*{c}*" for c in parse_conditions(node.vertex.rule.conditions)))
            source = getattr(node, "source", None)
            st.caption("Cornerstone case reference profile" + (f" (built from {source})" if source else "") + ":")
            with st.container(height=220):
                st.markdown(pretty(node.vertex.summary))
else:
    st.info("Tree is empty. Process a case to start.")

if DEMO_MODE:
    usage_slot.caption(f"AI actions used this session: {st.session_state.llm_actions} / {MAX_LLM_ACTIONS}")
