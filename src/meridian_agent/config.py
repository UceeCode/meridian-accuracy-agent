"""Central configuration. Every threshold that affects a consequential output lives here."""
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
except ImportError:
    pass
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("MERIDIAN_DATA_DIR", ROOT / "data"))
OUT_DIR = Path(os.getenv("MERIDIAN_OUT_DIR", ROOT / "outputs"))

# --- model -----------------------------------------------------------------
PROVIDER = os.getenv("MERIDIAN_PROVIDER", "ollama")                # "ollama" (free, local) or "anthropic" (paid API)
ANTHROPIC_MODEL = os.getenv("MERIDIAN_MODEL") or "claude-sonnet-5-5"
OLLAMA_MODEL = os.getenv("MERIDIAN_MODEL") or "qwen2.5:7b"          # any Ollama model that supports tool calling
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))          # Ollama's default window is small; tool output needs room
MAX_STEPS = int(os.getenv("MERIDIAN_MAX_STEPS", "40"))      # hard cap on agent loop turns
MAX_TOKENS = int(os.getenv("MERIDIAN_MAX_TOKENS", "4096"))
TOOL_RESULT_CHAR_LIMIT = 14000                                # keeps tool output bounded

# --- policy (from policy_error_rate_deduction.md) --------------------------
ERROR_RATE_THRESHOLD = 0.15

# --- attribution vocabulary ------------------------------------------------
CATEGORIES = ("EMPLOYEE", "SYSTEM", "PROCEDURE", "SHARED", "UNDETERMINED")
CONFIDENCES = ("low", "medium", "high")
CONF_RANK = {"low": 1, "medium": 2, "high": 3}

# Burden-of-proof ordering: moving a verdict UP this scale (towards blaming a person)
# requires more evidence than moving it down. See safeguards.validate_review.
BLAME_RANK = {"SYSTEM": 0, "PROCEDURE": 0, "UNDETERMINED": 1, "SHARED": 2, "EMPLOYEE": 3}

# Error types an automated system could plausibly cause vs. ones that are typically human slips.
SYSTEM_PLAUSIBLE = {"fx_rate_variance", "wrong_account_code", "duplicate_entry", "consolidation_variance"}
HUMAN_TYPE = {"transposition", "missing_attachment", "wrong_period"}

# --- person-level safeguards -----------------------------------------------
MIN_EMPLOYEE_ERRORS_FOR_REVIEW = 3     # medium+ confidence EMPLOYEE-attributed errors needed to appear in review list
MAX_UNDETERMINED_SHARE = 0.10          # above this, individual-level use of the output is not supported
