# rdr_engine.py
import sys
import os
import pickle
import logging
import json
import csv
import datetime
from typing import Optional, List, Tuple

# Import updated API functions
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from llm_api import (
        llm_check_condition, 
        llm_get_differentiating_conditions, 
        llm_generate_summary, 
        llm_merge_summaries
    )
except ImportError:
    from app.llm_api import (
        llm_check_condition, 
        llm_get_differentiating_conditions, 
        llm_generate_summary, 
        llm_merge_summaries
    )
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
#Path configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(CURRENT_DIR, '..','storage')
TREE_STORAGE_FILE = os.path.join(STORAGE_DIR, "rdr_tree_summary.pkl")
LOG_FILE = os.path.join(STORAGE_DIR, "rdr_event_log.csv")

class Rule:
    def __init__(self, conditions: str, conclusions: str):
        self.conditions = conditions
        self.conclusions = conclusions

class Vertex:
    def __init__(self, rule: Rule, summary: str):
        self.rule = rule
        self.summary: str = summary 

class Node:
    def __init__(self, vertex: Vertex):
        self.vertex = vertex
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None

LOG_HEADER = [
    "Date",
    "Time",
    "Action",
    "File_Path",
    "Last_True_Node",
    "Last_Eval_Node",
    "Revision_Triggered",
    "Condition_Type",
]


class RDREngine:
    def __init__(self, in_memory_log: bool = False):
        self.root: Optional[Node] = None
        # Stack: [(parent_node, side, new_node, source_file)]
        self.history: List[Tuple[Optional[Node], str, Node, str]] = []
        # In-memory logging keeps each demo visitor's events private to their session
        # instead of appending to the shared CSV file on disk.
        self.in_memory_log = in_memory_log
        self.event_rows: List[list] = []
        if not in_memory_log:
            self._init_log()

    def events_csv(self) -> str:
        """Returns the in-memory event log as CSV text (header included)."""
        import io
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(LOG_HEADER)
        writer.writerows(getattr(self, "event_rows", []))
        return buf.getvalue()

    def _init_log(self):
        """Initialize the CSV log with headers."""
        if not os.path.exists(STORAGE_DIR):
            os.makedirs(STORAGE_DIR)
            
        file_exists = os.path.exists(LOG_FILE)
        is_empty = False
        if file_exists:
            is_empty = os.stat(LOG_FILE).st_size == 0
            
        if not file_exists or is_empty:
            try:
                with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # Added 'Condition_Type' column
                    writer.writerow([
                        "Date",
                        "Time",
                        "Action", 
                        "File_Path", 
                        "Last_True_Node", 
                        "Last_Eval_Node", 
                        "Revision_Triggered",
                        "Condition_Type" # New Column
                    ])
            except Exception as e:
                logging.error(f"Failed to init log file: {e}")

    def _log_event(self, action, file_path, n_true: Optional[Node], n_eval: Optional[Node], rev_triggered, cond_type="N/A"):
        """Internal helper to append to the CSV log."""
        try:
            now = datetime.datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            
            def node_str(n):
                if not n: return "None"
                return f"[{id(n)}] {n.vertex.rule.conclusions[:30]}..."

            row = [
                date_str,
                time_str,
                action,
                file_path,
                node_str(n_true),
                node_str(n_eval),
                str(rev_triggered),
                cond_type # Write condition type
            ]
            if getattr(self, "in_memory_log", False):
                self.event_rows.append(row)
                return
            with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            logging.error(f"Logging failed: {e}")

    def interpret(self, current_patient_summary: str) -> Tuple[Optional[Node], Optional[Node]]:
        last_tried_node: Optional[Node] = None
        last_true_node: Optional[Node] = None
        current_node: Optional[Node] = self.root
        # Every rule evaluated on the way down, with its outcome, for explaining the result
        self.last_trace: List[Tuple[Node, bool]] = []

        while current_node:
            last_tried_node = current_node
            condition = current_node.vertex.rule.conditions
            is_true = llm_check_condition(current_patient_summary,condition)
            self.last_trace.append((current_node, is_true))

            if is_true:
                last_true_node = current_node
                current_node = current_node.right
            else:
                current_node = current_node.left

        return (last_tried_node, last_true_node)

    # --- MANAGEMENT METHODS ---

    def add_node_to_tree(self, parent_node: Optional[Node], new_node: Node, side: str, source_file: str = "Unknown", last_eval_node: Optional[Node] = None, last_true_node: Optional[Node] = None, condition_type: str = "Unknown"):
        """
        Adds a node, tracks it for undo, and logs the event with condition type.
        """
        if not hasattr(self, 'history'):
            self.history = []

        if side == "ROOT":
            self.root = new_node
            self.history.append((None, "ROOT", new_node, source_file))
            logging.info("ROOT node added.")
        elif side == "RIGHT" and parent_node:
            parent_node.right = new_node
            self.history.append((parent_node, "RIGHT", new_node, source_file))
            logging.info(f"Node added to RIGHT of {parent_node.vertex.rule.conclusions[:15]}...")
        elif side == "LEFT" and parent_node:
            parent_node.left = new_node
            self.history.append((parent_node, "LEFT", new_node, source_file))
            logging.info(f"Node added to LEFT of {parent_node.vertex.rule.conclusions[:15]}...")

        # Log with Condition Type
        self._log_event("ADD_RULE", source_file, last_true_node, last_eval_node, True, condition_type)

    def log_merge(self, source_file: str, last_eval_node: Optional[Node], last_true_node: Optional[Node]):
        """Logs a Merge event (Revision Triggered = False)."""
        self._log_event("MERGE_AGREEMENT", source_file, last_true_node, last_eval_node, False, "N/A")

    def undo_last_addition(self) -> Tuple[bool, str]:
        if not hasattr(self, 'history'):
            self.history = []
        
        if not self.history:
            return False, "No history to undo."

        item = self.history.pop()
        # Handle tuple size variation
        if len(item) == 4:
            parent, direction, node_to_remove, source_file = item
        else:
            parent, direction, node_to_remove = item
            source_file = "Unknown"

        if direction == "ROOT":
            self.root = None
            self._log_event("UNDO", source_file, None, None, "Reversed", "N/A")
            return True, "Root node removed. Tree is empty."
        
        if direction == "RIGHT":
            if parent.right == node_to_remove:
                parent.right = None
                self._log_event("UNDO", source_file, None, None, "Reversed", "N/A")
                return True, "Last specific rule (Right) removed."
            else:
                return False, "Tree structure mismatch during undo."
        
        if direction == "LEFT":
            if parent.left == node_to_remove:
                parent.left = None
                self._log_event("UNDO", source_file, None, None, "Reversed", "N/A")
                return True, "Last alternative rule (Left) removed."
            else:
                return False, "Tree structure mismatch during undo."
        
        return False, "Unknown error."

    def flush_tree(self):
        self._log_event("FLUSH_TREE", "N/A", None, None, "N/A", "N/A")
        self.root = None
        self.history = []
        if getattr(self, "in_memory_log", False):
            return  # Demo/session engines never touch the shared tree file
        if os.path.exists(TREE_STORAGE_FILE):
            try:
                os.remove(TREE_STORAGE_FILE)
                logging.warning(f"Deleted tree storage file: {TREE_STORAGE_FILE}")
            except Exception as e:
                logging.error(f"Failed to delete tree storage file: {e}")
        else:
            logging.info("Flush requested but no storage file found.")

    def revise(self, transcript_path: str):
        # Legacy CLI method - kept for compatibility
        pass

# --- JSON (de)serialization: used for the demo seed tree ---

def node_to_dict(node: Optional[Node]) -> Optional[dict]:
    if node is None:
        return None
    return {
        "conditions": json.loads(node.vertex.rule.conditions) if _is_json_list(node.vertex.rule.conditions) else node.vertex.rule.conditions,
        "conclusion": node.vertex.rule.conclusions,
        "cornerstone_summary": node.vertex.summary,
        "source": getattr(node, "source", None),
        "true_branch": node_to_dict(node.right),
        "false_branch": node_to_dict(node.left),
    }


def node_from_dict(data: Optional[dict]) -> Optional[Node]:
    if not data:
        return None
    conds = data["conditions"]
    cond_str = json.dumps(conds) if isinstance(conds, list) else str(conds)
    node = Node(Vertex(Rule(cond_str, data["conclusion"]), data.get("cornerstone_summary", "")))
    node.source = data.get("source")
    node.right = node_from_dict(data.get("true_branch"))
    node.left = node_from_dict(data.get("false_branch"))
    return node


def engine_from_dict(data: dict, in_memory_log: bool = True) -> "RDREngine":
    """Builds a fresh engine from a JSON tree. Seed nodes are not in the undo history."""
    engine = RDREngine(in_memory_log=in_memory_log)
    engine.root = node_from_dict(data.get("root"))
    engine.history = []
    return engine


def engine_to_dict(engine: "RDREngine") -> dict:
    return {"root": node_to_dict(engine.root)}


def _is_json_list(text) -> bool:
    try:
        return isinstance(json.loads(text), list)
    except (TypeError, ValueError):
        return False

# --- Persistence Helpers ---

def load_tree() -> RDREngine:
    if os.path.exists(TREE_STORAGE_FILE):
        try:
            with open(TREE_STORAGE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return RDREngine()

def save_tree(engine):
    with open(TREE_STORAGE_FILE, "wb") as f:
        pickle.dump(engine, f)