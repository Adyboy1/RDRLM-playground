# transcripts.py
"""Turns uploaded transcript files (.docx, .txt, .json) into plain dialogue text."""
import io
import json
import os

SUPPORTED_TYPES = ["docx", "txt", "json"]


def json_to_text(data) -> str:
    """
    Converts a JSON transcript into "Speaker: utterance" lines.
    Expects a list of turns like {"interlocutor": "therapist", "utterance_text": "..."}
    (the AnnoMI format), or an object holding such a list under "utterances" / "transcript".
    """
    if isinstance(data, dict):
        data = data.get("utterances") or data.get("transcript") or []
    if not isinstance(data, list):
        raise ValueError("JSON transcript must be a list of utterances.")

    lines = []
    for turn in data:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("interlocutor") or turn.get("speaker") or "Speaker").strip().capitalize()
        text = str(turn.get("utterance_text") or turn.get("text") or "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    if not lines:
        raise ValueError("No utterances found in the JSON transcript.")
    return "\n".join(lines)


def file_to_text(filename: str, raw: bytes) -> str:
    """Extracts dialogue text from an uploaded file's name and bytes."""
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    if ext == "docx":
        import docx
        doc = docx.Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs)
    if ext == "json":
        return json_to_text(json.loads(raw.decode("utf-8-sig")))
    if ext == "txt":
        return raw.decode("utf-8-sig", errors="replace")
    raise ValueError(f"Unsupported file type: .{ext}")
