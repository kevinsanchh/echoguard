from pathlib import Path
import torch
import os
from dotenv import load_dotenv

# -----------------------
# Global configuration
# -----------------------

# Model + class file paths
load_dotenv()

MODEL_PATH = Path("assets/multi-label_model.pth")
CSV_PATH = Path("assets/dataset_train.csv")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_prompt_file = os.getenv("GEMINI_STATIC_PROMPT")
_prompt_path = Path(__file__).resolve().parent.parent.parent / _prompt_file

try:
    GEMINI_STATIC_PROMPT = _prompt_path.read_text(encoding="utf-8")
except Exception as e:
    raise RuntimeError(f"Failed to load Gemini static prompt file: {_prompt_path}. Error: {e}")
print("Loaded Static Prompt Length:", len(GEMINI_STATIC_PROMPT))

# Audio processing parameters
SAMPLE_RATE = 16000
MAX_DURATION = 10
MAX_SAMPLES = SAMPLE_RATE * MAX_DURATION  # seconds

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
