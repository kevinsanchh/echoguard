# EchoGuard

EchoGuard is a real-time audio monitoring and analysis system designed to detect, classify, and analyze environmental sounds and speech. It utilizes a hybrid AI pipeline that separates audio into speech and non-speech components, processing them in parallel to provide a comprehensive context-aware risk analysis using Google's Gemini 2.5 Flash.

## Overview

The system captures audio input (either via real-time microphone recording or file upload) and processes it through a sophisticated pipeline:

1.  **Voice Activity Detection (VAD):** Uses RNNoise to distinguish between human speech and environmental sounds (non-speech).
2.  **Speech Transcription:** Speech segments are processed using `faster-whisper` to generate accurate transcripts.
3.  **Environmental Classification:** Non-speech segments are validated and classified using a custom PyTorch-based CNN (`ImprovedResNetAudio`) to detect specific sounds (e.g., alarms, aggressive noises).
4.  **Contextual Analysis:** The transcription and classification results are aggregated and sent to Google Gemini.
5.  **Risk Assessment:** Gemini generates a "Risk Score," "Benefit Score," and detailed reasoning, which is displayed on the frontend dashboard.

## Features

- **Real-time Monitoring:** Captures audio directly from the browser with visual feedback.
- **Dual-Pipeline Processing:** Simultaneous handling of speech (transcription) and non-speech (event detection).
- **AI-Powered Analysis:** Integrates Google Gemini 2.5 Flash to synthesize audio data into actionable insights.
- **Custom CNN Model:** Uses a ResNet-based architecture with Squeeze-and-Excitation blocks for multi-label audio classification.
- **Audio Validation:** Filters out low-quality or irrelevant non-speech audio to reduce false positives.
- **Interactive Dashboard:** A Next.js-based UI that displays recording history, detected events, confidence scores, and risk analysis.
- **File Upload Support:** Allows users to upload pre-recorded WAV files for analysis.

## Tech Stack

### Frontend

- **Framework:** Next.js (React)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **Animations:** Framer Motion
- **Icons:** Lucide React

### Backend (API)

- **Server:** Flask (Python 3.12)
- **Machine Learning:** PyTorch, Torchaudio
- **Speech-to-Text:** Faster-Whisper
- **VAD:** RNNoise (Python wrapper)
- **LLM Integration:** Google Generative AI (Gemini)

## Architecture

The backend is organized into a modular pipeline orchestrated by a central router:

- **/api/process/vad**: The entry point for audio chunks. Splits audio into speech and non-speech.
- **/api/process/validate-non-speech**: Checks non-speech audio for duration, RMS amplitude, and loudness before classification.
- **/api/process/model**: Runs the custom CNN on valid non-speech audio to detect event labels.
- **/api/process/transcribe**: Stitches speech segments and runs the Whisper model.
- **/api/process/gemini**: Aggregates all data and requests a final analysis from the LLM.

## Prerequisites

- Python 3.12+
- Node.js 18+
- FFmpeg (Required for Torchaudio and Whisper)
- Google Gemini API Key

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd echoguard
```

### 2. Backend Setup

Navigate to the root directory (or API folder if separated) and install the Python dependencies.

```bash
# It is recommended to create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

**Note:** Ensure you have the model assets placed in the correct directory:

- `api/assets/multi-label_model.pth`
- `api/assets/dataset_train.csv`

### 3. Frontend Setup

Install the Node.js dependencies.

```bash
npm install
```

## Configuration

Create a `.env` file in the root directory (or where your Flask server reads environment variables) with the following keys:

```ini
GEMINI_API_KEY=your_google_gemini_api_key
GEMINI_STATIC_PROMPT=prompt.txt
```

Ensure `prompt.txt` exists at the path resolved in `api/utils/config.py`.

## Running the Application

### Start the Backend Server

```bash
python api/server.py
```

The server will start on `http://localhost:8080`.

### Start the Frontend Client

```bash
npm run dev
```

The application will be available at `http://localhost:3000`.

## Usage

1.  Open the application in your browser.
2.  Click the **Start Monitoring** button to begin real-time recording, or use the **Upload** feature for existing files.
3.  The application will visualize the audio input.
4.  Once recording stops, the backend processes the audio chunks.
5.  Click the **Dashboard** button to view the results, including transcripts, detected sounds, and the AI-generated risk assessment.

## Project Structure

- **app/**: Next.js frontend pages and layouts.
- **components/**: Reusable UI components (Dashboard, Audio Recorder).
- **api/**: Flask backend.
  - **routes/**: API endpoints for VAD, Transcription, Model, and Gemini.
  - **utils/**: Helper logic for audio processing, validation, and session management.
  - **assets/**: ML model weights and dataset references.
