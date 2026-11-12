# 🧠 Clinical RDR Knowledge Base

**▶️ Live demo: [rdrlm-playground.streamlit.app](https://rdrlm-playground.streamlit.app/)** (no sign-in needed)

An interpretable clinical decision-support system that combines **Single Classification Ripple Down Rules (SCRDR)** with a large language model (Google Gemini).

AI tools that assess therapy sessions are usually black boxes: a clinician can't see *why* a case was classified a certain way, or fix a wrong answer without retraining a model. This project keeps the reasoning in a transparent, auditable binary decision tree. Domain experts correct it incrementally by adding exception rules in plain English, so the knowledge base grows case by case and stays clinically grounded, without any programming.

---

## How it works

1. **Summarise.** Gemini turns a raw counselling transcript into a structured clinical summary.
2. **Classify.** The summary walks down the rule tree. Each rule's condition is plain English and Gemini judges whether it holds: follow the **True** branch if it does, the **False** branch if not. The answer is the conclusion of the *last rule that was true*, so every result traces back to specific, human-written rules.
3. **Correct.** When the clinician disagrees, they give the right conclusion and choose conditions that separate this case from the *cornerstone case* (the case that created the rule that misfired). Gemini suggests candidate conditions and checks typed ones. The new rule is attached exactly where the reasoning went wrong, so it can't change answers for cases that reach other branches.

The demo starts from a five-rule tree built from real counselling transcripts:

```
R1  Drinks above low-risk guidelines → Hazardous drinking: brief motivational intervention
├─ True:  R2  Injury or physical health problem linked to drinking → Harmful drinking: brief intervention + medical follow-up
│         └─ False: R4  Under the legal drinking age → Underage heavy drinking: involve family/school, safety plan
└─ False: R3  Under the legal drinking age → Low-risk underage drinking: prevention and education
          └─ True:  R5  Has vomited or been sick from drinking → Underage drinking with past intoxication
```

R3 and R4 test the same condition but reach different conclusions, because a rule's meaning depends on where it sits in the tree. That context-dependence is the core idea of Ripple Down Rules.

## Try the demo

The [live demo](https://rdrlm-playground.streamlit.app/) gives every visitor a private copy of the tree, so nothing you change is saved or seen by anyone else. Six guided sample cases walk through the whole loop:

| # | Case | What it shows |
|---|---|---|
| 1 | University student who sprained her ankle on a pub crawl | A correct conclusion → **Agree** merges the case into the rule's reference profile |
| 2 | Teenager who has tried alcohol once or twice | The other side of the tree → **Agree** |
| 3 | Evening wine drinking to cope with low mood | **Disagree** → build an exception rule from the AI's suggested differences |
| 4 | Drinking while pregnant | **Disagree** → type your own condition, which is checked against both cases |
| 5 | Another pregnant client | No rule fires → the new rule goes on a **False** branch, and step 4's rule doesn't apply here (rules only work in their own context) |
| 6 | Single working mum who drinks to unwind from stress | Your call: agree, or teach the tree a new rule |

You can also download a sample, edit it, and upload it, or upload your own transcript (`.json` in AnnoMI format, `.docx`, or `.txt`). Please don't upload real patient data.

## Features

- **Transcript processing:** parses AnnoMI-style JSON, `.docx` and `.txt` transcripts into structured clinical summaries.
- **Interpretable classification:** every conclusion shows the rule that gave it and the path the case took (shaded in the tree diagram).
- **Guided knowledge acquisition:** AI-suggested differentiating conditions; typed conditions are verified to be true for the new case and false for the cornerstone case.
- **Repeatable results:** condition checks run at temperature 0, so the same case takes the same path every time.
- **Quota-aware:** a shared answer cache (pre-filled for the sample cases), rate-limit retries, clear messages when the daily quota runs out, and a per-session cap on AI actions.
- **Bonus counsellor feedback:** the summary step also rates the counsellor's skills against a rubric; this is shown separately and never used by the rules.
- **Two modes:** a public **demo mode** (default; per-visitor tree, nothing persisted, no database) and a **persistent mode** with Firebase Firestore storage and multi-tenant isolation via `app_id`.

## Tech stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| LLM | Google Gemini via the [`google-genai`](https://pypi.org/project/google-genai/) SDK (model set in secrets) |
| Rule engine | Custom Python SCRDR implementation |
| Visualisation | Graphviz (rendered in the browser by Streamlit) |
| Storage (persistent mode only) | Google Firebase Firestore |

## Running locally

**Prerequisites:** Python 3.11+, a [Google AI Studio](https://aistudio.google.com/) API key, and (for persistent mode only) a Firebase project with Firestore and a service-account key.

```bash
git clone https://github.com/Adyboy1/RDRLM-playground.git
cd RDRLM-playground
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create **`app/.streamlit/secrets.toml`** (Streamlit reads secrets from the folder you run it in):

```toml
# These two must be at the very top, above any [section]
API_KEY = "your-gemini-api-key"
GEMINI_MODEL = "gemini-3.5-flash-lite"   # switch models here; no code change or redeploy needed

[general]
demo_mode = true                 # false = persistent mode with Firebase
app_id = "Development_Environment"   # persistent mode: separates each organisation's data in Firestore

# Persistent mode only: paste the values from your Firebase Admin SDK JSON key
[firebase]
type = "service_account"
project_id = "your-firebase-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "firebase-adminsdk-xxxxx@your-project.iam.gserviceaccount.com"
client_id = "1234567890"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account"
```

Then start the app:

```bash
cd app
streamlit run app.py
```

It opens at <http://localhost:8501>. On Streamlit Community Cloud, paste the same TOML into the app's **Secrets** settings and set the main file to `app/app.py`.

### Rebuilding the demo tree

The seed tree (`demo/seed_tree.json`) and the cached answers for the samples (`demo/llm_cache.json`) are generated by a script. Re-run it after changing the tree in `scripts/build_seed_tree.py` or editing any prompt:

```bash
python scripts/build_seed_tree.py
```

It reads the API key and model from the same secrets file, waits out free-tier rate limits, and keeps the answers it already has, so an interrupted run resumes where it stopped.

## Project structure

```
app/
  app.py            Streamlit UI: demo/persistent modes, analysis, review, rule editing, tree diagram
  rdr_engine.py     SCRDR engine: Rule / Vertex / Node, interpretation with a reasoning trace, undo, event log, JSON (de)serialisation
  llm_api.py        Gemini calls (summary, condition check, differences, merge), response cache, rate-limit handling
  transcripts.py    Converts .json / .docx / .txt uploads into dialogue text
  backend.py        Firestore persistence for persistent mode
prompts/            Prompt templates for each Gemini task
demo/
  seed_tree.json    The starting knowledge base
  llm_cache.json    Pre-computed Gemini answers for the sample cases
  samples/          AnnoMI transcripts used as cornerstones and sample cases
scripts/
  build_seed_tree.py  Builds the seed tree and the answer cache
```

## Limitations

- **This is an educational demo, not a clinical tool.** The rules and conclusions in the seed tree are illustrative.
- Condition checks are LLM judgements. They are repeatable but not infallible, and the wording of a condition affects how strictly it is applied.
- "Agree" merges a case into the rule's reference profile, so cornerstones become generalised prototypes rather than fixed single cases (a deliberate departure from classic SCRDR).
- The public demo runs on a free-tier Gemini key. Sample cases mostly use cached answers; uploads and newly added rules call Gemini live and count against a shared daily quota.

## Data and citation

Sample transcripts come from **AnnoMI**, a public dataset of expert-annotated counselling (motivational interviewing) demonstration videos. No real patient data is used.

> Z. Wu, S. Balloccu, V. Kumar, R. Helaoui, E. Reiter, D. Reforgiato Recupero and D. Riboni, "Anno-MI: A Dataset of Expert-Annotated Counselling Dialogues," *ICASSP 2022*, pp. 6177–6181. [doi:10.1109/ICASSP43922.2022.9746035](https://doi.org/10.1109/ICASSP43922.2022.9746035) · [Dataset on GitHub](https://github.com/uccollab/AnnoMI)

---

Developed as part of the Datalab Project for Clinical Psychology.
