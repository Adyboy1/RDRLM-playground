🧠 Clinical RDR Knowledge Base

An interpretable clinical decision support system using a Single Classification Ripple Down Rules (SCRDR) architecture integrated with Large Language Models (Google Gemini).

This project addresses the "black box" problem in AI-based psychological assessment by creating a transparent, auditable binary decision tree. It features a "human-in-the-loop" workflow where domain experts (psychologists) can incrementally correct the model's logic by adding specific exception rules, ensuring the knowledge base grows organically and remains clinically valid without requiring programming expertise.

✨ Key Features

Automated Transcript Processing: Parses raw therapy transcripts (.json in AnnoMI format, .docx, .txt) into structured clinical summaries using LLMs.

Interpretable AI (SCRDR): Classifies summaries using a binary decision tree where every conclusion is traceable to specific human-verified rules.

Knowledge Acquisition: Clinicians can easily add exception rules when the system makes an incorrect diagnosis by leveraging AI to find differentiating conditions between cases.

Cloud Persistence: Fully cloud-native backend using Firebase Firestore, eliminating local file synchronization issues and ensuring data survives container restarts.

Multi-Tenant Architecture: Securely supports multiple organizations using the same codebase by isolating databases via configurable app_id secrets.

Live Visualization: Real-time visualization of the decision tree using Graphviz.

🛠️ Tech Stack

Frontend/UI: Streamlit

Backend Database: Google Firebase (Firestore)

LLM API: Google Gemini API (google-genai)

Logic Engine: Custom Python Ripple Down Rules implementation

Visualization: Graphviz

⚙️ Prerequisites

Before installing, ensure you have the following:

Python 3.8+ installed on your system.

Graphviz OS Binaries: The Python graphviz package requires the system-level Graphviz binaries to be installed.

Windows: Download from Graphviz website and add to PATH.

Mac: brew install graphviz

Linux: sudo apt-get install graphviz

Google Gemini API Key: Get one from Google AI Studio.

Firebase Service Account: A Firebase project with Firestore enabled, and a generated Service Account JSON key.

🚀 Installation

1. Clone the repository

git clone [https://github.com/yourusername/clinical-rdr-engine.git](https://github.com/yourusername/clinical-rdr-engine.git)
cd clinical-rdr-engine


2. Create a virtual environment (Recommended)

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate


3. Install Python dependencies

pip install -r requirements.txt


(Ensure streamlit, google-genai, firebase-admin, python-docx, and graphviz are in your requirements.txt)

🔐 Configuration (Secrets Setup)

Because this application uses a multi-tenant cloud architecture, it relies entirely on Streamlit's secrets management. You must create a .streamlit/secrets.toml file in the root of your project.

1. Create the secrets file

mkdir .streamlit
touch .streamlit/secrets.toml


2. Add your credentials to secrets.toml

Copy the following template and fill in your specific details:

app/.streamlit/secrets.toml (Streamlit reads secrets from the directory you run it in)

API_KEY = "your-google-gemini-api-key-here"   # must be top-level, above any [section]

GEMINI_MODEL = "gemini-2.5-flash-lite"        # required, top-level; change it here to switch models (no redeploy needed)

[general]

demo_mode = false   # omit or set true for the public demo: per-visitor seed tree, nothing saved, no Firebase needed

Change this ID for different deployments/organizations!

This determines the Firestore collection where data is stored.

e.g., "Org_A_Data", "Test_Environment", etc.

app_id = "Development_Environment"

[firebase]

Paste the values from your downloaded Firebase Admin SDK JSON file here

type = "service_account"

project_id = "your-firebase-project-id"

private_key_id = "your-private-key-id"

private_key = "-----BEGIN PRIVATE KEY-----\nYOUR...LONG...KEY...HERE\n-----END PRIVATE KEY-----\n"

client_email = "firebase-adminsdk-xxxxx@your-project.iam.gserviceaccount.com"

client_id = "1234567890"

auth_uri = "[https://accounts.google.com/o/oauth2/auth](https://accounts.google.com/o/oauth2/auth)"

token_uri = "[https://oauth2.googleapis.com/token](https://oauth2.googleapis.com/token)"

auth_provider_x509_cert_url = "[https://www.googleapis.com/oauth2/v1/certs](https://www.googleapis.com/oauth2/v1/certs)"

client_x509_cert_url = "[https://www.googleapis.com/robot/v1/metadata/x509/your-service-account](https://www.googleapis.com/robot/v1/metadata/x509/your-service-account)"


🏃‍♂️ Running the Application

Once your environment is set up and secrets are configured, start the Streamlit server from the app folder:

cd app
streamlit run app.py


The application should now be accessible in your web browser at http://localhost:8501.

🎬 Demo Mode (the default)

Unless `demo_mode = false` is set, every visitor gets a private copy of a seed tree (demo/seed_tree.json) built from
alcohol-use counselling transcripts in the public AnnoMI dataset (demo/samples/). Nothing is saved and Firebase is not used.
To keep the walkthrough fast on a free-tier Gemini key, demo/llm_cache.json ships pre-computed answers for the samples.
Rebuild both after changing the seed tree or the prompts:

python scripts/build_seed_tree.py

📖 How to Use the App

Upload a Transcript: Navigate to the "Case Processing" tab and upload a patient transcript (.docx or .txt).

Analyze Case: Click "Analyze Case". The LLM will generate a clinical summary and run it through the current RDR decision tree.

Review Diagnosis:

Agree: If the system's conclusion is correct, click "Agree". The system will merge the current patient's summary with the reference summary for that rule and log the event.

Disagree (Knowledge Acquisition): If the conclusion is incorrect or missing, click "Disagree".

Revise the Tree:

Enter the correct conclusion.

Click "Find Differences" to have the LLM automatically suggest differentiating conditions between the current patient and the previous reference patient.

Select the appropriate conditions or type a manual one (the system will automatically verify manual conditions against the transcripts).

Click "Save Rule". The tree is instantly updated in the Cloud.

Admin Controls: Use the Sidebar to Undo rules, Flush the tree, or download the Tree (.pkl) and Event Logs (.csv) directly from Firebase.

📂 Project Structure

app.py: Main Streamlit application, handles UI, session state, and cloud sync logic.

rdr_engine.py: Contains the core logic for the SCRDR binary tree (Node, Rule, Vertex, RDREngine). Strictly in-memory; no local file I/O.

backend.py: Handles all Firebase Admin SDK connections. Serializes/deserializes the engine to Base64 and pushes/pulls data to Firestore.

llm_api.py: Manages interactions with the Google Gemini API (Summarization, Condition Checking, Finding Differentiators).

requirements.txt: Python package dependencies.

README.md: This documentation file.

Developed as part of the Datalab Project for Clinical Psychology.