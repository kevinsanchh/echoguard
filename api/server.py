from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from pathlib import Path

# ML Model Imports & Configuration - Copied from model.py
import torch.nn.functional as F

from utils.config import DEVICE
from utils.model_loader import load_ml_assets
from utils.vad_utils import load_vad_model

from routes.home import home_bp
from routes.classify import classify_bp
from routes.vad_primary import vad_bp
from routes.validate_non_speech import validate_bp
from routes.transcribe import transcribe_bp
from routes.gemini_analysis import gemini_bp

# app instance
app = Flask(__name__)
CORS(app)

if not os.path.exists(app.instance_path):
    os.makedirs(app.instance_path)
    print(f"DEBUG: Created instance folder: {app.instance_path}")

# Load ML model and classes at startup once
# GLOBAL MODEL AND CLASSES
global_model, global_classes = load_ml_assets()
app.config["model"] = global_model
app.config["classes"] = global_classes
app.config["device"] = DEVICE
print("DEBUG: ML model and classes loaded successfully.")

# Load RNNoise VAD model at startup (Silero-compatible interface)
try:
    vad_model, vad_helpers = load_vad_model()
    app.config["vad_model"] = vad_model
    app.config["vad_helpers"] = vad_helpers
    print("DEBUG: VAD model loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load VAD model: {e}")
    raise e

# Register Blueprints
app.register_blueprint(home_bp)
app.register_blueprint(classify_bp)
app.register_blueprint(vad_bp)
app.register_blueprint(validate_bp)
app.register_blueprint(transcribe_bp)
app.register_blueprint(gemini_bp)


# To run locally use 'python server.py'
if __name__ == "__main__":
    # Vercel sets the 'VERCEL' environment variable to '1' during deployment.
    # We check if this variable is NOT set before starting the development server.
    
    if os.environ.get('VERCEL') != '1':
        app.run(debug=True, port=8080)