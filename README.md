# 🎧 EchoGuard

An AI-powered audio safety analysis platform that evaluates the **risk**, **benefit**, and **overall appropriateness** of audio content by analyzing both **speech** and **environmental sounds**.  
EchoGuard combines deep learning, speech-to-text transcription, and an intelligent Gemini-based scoring layer to help parents and guardians understand the media their children consume.

---

## 🧭 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [Backend Pipeline](#-backend-pipeline)
- [Frontend Application](#-frontend-application)
- [Data Flow Lifecycle](#-data-flow-lifecycle)
- [Project Structure](#-project-structure)
- [Required Configuration Setup](#required-configuration-setup)
- [Installation Guide](#installation-guide)
- [UI Screenshots](#-ui-screenshots)
- [Project Videos](#-project-videos)
- [Contact](#-contact)

---

## 🚀 Features

- Dual-pipeline audio evaluation (speech + environmental sounds)
- **Faster-Whisper transcription** for high-accuracy speech detection  
- **Multi-label PyTorch CNN classifier** for hazardous/environmental sound recognition  
- **Gemini-based scoring via Gemini 2.5 Pro Model** generating:
  - `risk_score`  
  - `benefit_score`  
  - `confidence_score`  
  - `risk_reasoning`  
  - `benefit_reasoning`  
  - `confidence_reasoning`  
- Real-time dashboard for viewing detailed breakdowns  
- VAD (Voice Activity Detection) to split raw WAV files into non-speech and speech segments
- Fully modular backend for cleaner extensions and updates  
- Structured directories for documentation, slides, posters, and videos  

---

## 🛠 Tech Stack

| Component | Tools / Frameworks |
|----------|---------------------|
| **Frontend** | Next.js, React, Tailwind CSS |
| **Backend API** | Flask (Python) |
| **Speech Transcription** | Faster-Whisper |
| **Violent Environmental Sound Classification** | CNN Model, PyTorch, TorchAudio |
| **AI Scoring** | Gemini 2.5 Pro Model |
| **Processing Tools** | FFmpeg, VAD |
| **Other** | npm, pip, REST APIs |

---

## 🏗 System Architecture

EchoGuard’s architecture is divided into three major layers:

1. **Frontend (Next.js)**  
   - Audio upload  
   - User interface  
   - API request handling  
   - Risk/benefit/confidence visualization  

2. **Backend (Flask)**  
   - VAD segmentation  
   - Speech transcription  
   - Sound classification  
   - Scoring logic and Gemini evaluation  

3. **Gemini 2.5 Pro**  
   - Interprets transcript + sound events  
   - Produces structured JSON scoring output  

---

## 🔌 Backend Pipeline

### **1️⃣ Session Manager**
- Creates per-run directories  
- Stores uploaded audio, VAD chunks, transcripts, and output JSON  

### **2️⃣ Voice Activity Detection (VAD)**
- Splits raw WAV files into non-speech and speech segments
- Sends non-speech segments to validate/non-speech endpoint

### **3️⃣ Faster-Whisper Transcription**
- Generates transcript fopr full recording
- Merges text into a structured final transcript  

### **4️⃣ Environmental Sound Classification**
- Multi-label CNN model (PyTorch)  
- Predicts probability of 14+ sound event classes  
- Aggregates predictions across chunks  

### **5️⃣ Gemini Risk/Benefit Scoring**
- Combines transcript + sound events  
- Sends structured prompt to Gemini  
- Gemini returns:
  ```json
  {
    "risk_score": 0,
    "benefit_score": 0,
    "confidence_score": 0.0,
    "risk_reasoning": "...",
    "benefit_reasoning": "...",
    "confidence_reasoning": "...",
  }
  ```

### **6️⃣ Final Output Assembly**
- Combines all results  
- Sends JSON response back to the frontend dashboard  

---

## 🖥 Frontend Application

The frontend is built with **Next.js + Tailwind CSS** and includes:

- **Home Page**  
- **Audio Upload Page**  
- **Real-time Processing Indicator**  
- **Risk/Benefit Score Display**   
- **Detected Sound Events**  
- **Gemini Explanation Modal**  

Data is retrieved from Flask via REST endpoints and displayed through dynamic React components.

---

## 🔄 Data Flow Lifecycle

EchoGuard supports **two distinct processing workflows** depending on how the user provides audio input:  
1. A **Live Recording Workflow** for real-time detection  
2. An **Upload Audio File Workflow** for batch processing of pre-existing audio files

Both workflows ultimately converge into the Gemini scoring pipeline but differ in how and when audio is processed.

---

### 🎙️ Live Recording Workflow  
When using the live recording feature, EchoGuard processes audio **in real time**. The recording is split into 5-second chunks that are sent to the backend as soon as they are captured, enabling immediate environmental sound detection while the user is still recording.

### Live Recording Diagram
> ![Live Recording Workflow Diagram](diagrams/echoguard_live_recording_workflow.png)

1. User records live audio  
2. Backend stores raw WAV file  
3. VAD splits into speech and non-speech segments  
4. Non-speech segments sent to validation/non-speech endpoint in real time  
5. All validated non-speech segments reach the CNN classifier  
6. CNN analyzes segments and sends **live environmental sound detections** back to the frontend  
   - *(Steps 1–6 repeat for each 5-second subclip)*  
7. Once the user finishes recording, **all stored 5-second WAV files** are sent for transcription  
8. Full transcript + CNN model results from all subclips are sent to **Gemini 2.5 Pro**  
9. Gemini produces risk, benefit, and confidence scores along with reasoning  
10. Backend returns structured JSON output  
11. Frontend renders all final results in the dashboard  

---

### 📁 Upload Audio File Workflow  
When uploading an existing audio file, the entire clip is processed **at once**. VAD identifies all speech vs. non-speech segments, which are then fed into the transcription module and sound classification model simultaneously.

### Upload Audio File Diagram
> ![Upload Audio File Diagram](diagrams/echoguard_upload_audio_file_workflow.png)

1. User uploads an existing audio file  
2. Backend stores the full raw WAV file in a session directory  
3. VAD analyzes the entire file and splits it into **speech** and **non-speech** segments  
4. **All non-speech segments** are sent to the validation/non-speech endpoint
5. **All validated non-speech segments** are sent to the CNN Model Classifier
5. **All raw WAV files** are sent to Faster-Whisper for full transcription  
6. Backend aggregates:  
   - Complete transcript  
   - CNN predictions across all non-speech segments  
7. Combined transcript + sound classification results are sent to **Gemini 2.5 Pro** in one scoring request  
8. Gemini returns structured JSON containing risk, benefit, confidence, and reasoning for each
9. Backend assembles unified output
10. Frontend displays the complete evaluation in the dashboard  

---


## 📁 Project Structure

### **Frontend (Next.js)**

| Path | Purpose |
|------|---------|
| `app/` | Main application routes |
| `components/` | Reusable React components |
| `styles/` | Global CSS & Tailwind layers |
| `public/` | Static assets |
| `lib/` | API utilities & helpers |
| `hooks/` | Custom React hooks |

### **Backend (Flask)**

| Path | Purpose |
|------|---------|
| `api/server.py` | Flask entrypoint |
| `api/routes/` | API endpoints |
| `api/utils/` | Utilities and helper functions used inside routes |
| `api/routes/transcribe/` | Faster-Whisper Transcription logic |
| `api/routes/cnn_model/` | CNN model + inference |
| `api/routes/gemini_analysis` | Scoring wrapper logic |

### **Documentation/Media Directories**

| Directory | Content |
|-----------|---------|
| `docs/` | Architecture diagrams, documentation |
| `posters/` | Capstone poster files |
| `slides/` | Presentation slide decks |
| `videos/` | index.html with links to YT videos |
| `screenshots/` | UI screenshots for README |

---

## ⚙️ Required Configuration Setup

Before starting the backend server, EchoGuard requires two configuration files.

Inside the project root, you will find an `example/` directory containing:

- `example_.env`
- `example_static_gemini_prompt.txt`

These files must be moved and renamed before running the backend.

---

#### 1️⃣ Prepare the Configuration Files

From the project root:

1. Move both files out of the `example/` directory into the root directory.
2. Rename them as follows:

| Example File | Rename To |
|-------------|----------|
| `example_.env` | `.env` |
| `example_static_gemini_prompt.txt` | `static_gemini_prompt.txt` |

After this step, your root directory should include:
- `.env`
- `static_gemini_prompt.txt`

You may delete the `example/` directory once this is done.

---

#### 2️⃣ Update the `.env` File

Open the `.env` file and provide **your own value** for the GEMINI_API_KEY

Example:
```.env
GEMINI_API_KEY=your_own_gemini_api_key_here
GEMINI_STATIC_PROMPT=static_gemini_prompt.txt
```

- **`GEMINI_API_KEY`**  
  Must be your own valid Gemini API key. The example file intentionally leaves this blank.

- **`GEMINI_STATIC_PROMPT`**  
  Points to the static prompt file used by EchoGuard.  
  The actual prompt content is **not included** in the repository and must be provided manually.

---

#### 3️⃣ Update the Static Prompt File

Open `static_gemini_prompt.txt` and insert your own prompt text.

> ⚠️ The original project prompt is intentionally not exposed.  
> You must define your own Gemini prompt logic in this file for the system to function.

---

Once these steps are complete, you can proceed with the backend setup and start the server normally.


## Installation Guide

Before cloning, create your own copy of the project by **forking** the repository.

### 1️⃣ Fork the Repository
Visit the GitHub repository and click **Fork** to create your own copy under your account:
```
https://github.com/your-username/echoguard
```

### 2️⃣ Clone Your Fork
```bash
git clone https://github.com/your-username/echoguard.git
cd echoguard
```

---

### 3️⃣ Backend Setup (Flask)
```bash
cd api
pip install -r requirements.txt
python3 server.py
```

- Backend will run at:  
  ```
  http://localhost:8080
  ```

---

### 4️⃣ Frontend Setup (Next.js)
```bash
Inside the root directory:
npm install
npm run dev
```

- Frontend will run at:  
  ```
  http://localhost:3000
  ```

---

### ✔ Summary
- Fork → Clone → Install backend dependencies → Install frontend dependencies  
- Run backend and frontend in parallel  
- Access the UI through `localhost:3000`  

---

## 🖼 UI Screenshots

### Home Page  
![Home Page](screenshots/home_page.png)

### Buttons to record/upload Audio
![Buttons to record/upload Audio](screenshots/recording_buttons.png)

### Analyzing Audio Animation
![Analyzing Audio Animation](screenshots/analyzing_audio_animation.png)

### Detecting Sounds in Real time (Live Record workflow)
![Detecting Sounds in Real time (Live Record workflow)](screenshots/detected_sounds_in_real_time.png)

### Playback feature + View Results Button
![Playback feature + View Results Button](screenshots/playback_and_view_results.png)

### Not Enough Context Message
![Not Enough Context Message](screenshots/not_enough_context_message.png)

### Dashboard (High Confidence score)
![Dashboard (High Confidence score)](screenshots/dashboard_high_confidence.png)

### Dashboard (Low Confidence score)
![Dashboard (Low Confidence score)](screenshots/dashboard_low_confidence.png)`

### Dashboard (No Environmental Sounds)
![Dashboard (No Environmental Sounds)](screenshots/dashboard_no_environmental_sounds.png)

---

## 🎬 Project Videos

Click any video thumbnail below to watch it on YouTube.

---

### ▶️ 🎧 Intro Video (Click to Watch on YouTube)
Introduces the EchoGuard software solution, motivation, and overall system goals.

[![Watch on YouTube](https://img.shields.io/badge/Watch%20on-YouTube-red?logo=youtube)](https://www.youtube.com/watch?v=jNRVgff8B6c)

[![EchoGuard Intro Video](https://img.youtube.com/vi/jNRVgff8B6c/0.jpg)](https://www.youtube.com/watch?v=jNRVgff8B6c)

---

### ▶️ 📘 User Guide (Click to Watch on YouTube)
Demonstrates how each human actor interacts with EchoGuard, including live recording and upload workflows.

[![Watch on YouTube](https://img.shields.io/badge/Watch%20on-YouTube-red?logo=youtube)](https://www.youtube.com/watch?v=ZLmIjZMEtM4)

[![EchoGuard User Guide](https://img.youtube.com/vi/ZLmIjZMEtM4/0.jpg)](https://www.youtube.com/watch?v=ZLmIjZMEtM4)

---

### ▶️ ⚙️ Installation & Maintenance Guide (Click to Watch on YouTube)
Covers project setup and configuration.

[![Watch on YouTube](https://img.shields.io/badge/Watch%20on-YouTube-red?logo=youtube)](https://www.youtube.com/watch?v=CDl6xSZhbOw)

[![EchoGuard Installation Guide](https://img.youtube.com/vi/CDl6xSZhbOw/0.jpg)](https://www.youtube.com/watch?v=CDl6xSZhbOw)

---

### ▶️ 🚀 Future Improvements (Click to Watch on YouTube)
Discusses proposed enhancements, extensions, and future directions for EchoGuard.

[![Watch on YouTube](https://img.shields.io/badge/Watch%20on-YouTube-red?logo=youtube)](https://www.youtube.com/watch?v=meSsb-Ases4)

[![EchoGuard Future Improvements](https://img.youtube.com/vi/meSsb-Ases4/0.jpg)](https://www.youtube.com/watch?v=meSsb-Ases4)

---

## 🙏 Acknowledgements

This project was developed as part of a **Capstone course** and benefited greatly from the guidance, feedback, and support of the following individuals. Their expertise and mentorship played a key role in shaping EchoGuard’s technical direction and overall quality.

### Faculty Advisors
- **Professor Christian Poellabauer**  
  Capstone Instructor / Faculty Advisor  
  GitHub: https://github.com/cpoellab
  Email: [cpoellab@fiu.edu]

- **Shayl Griffith**  
  Assistant Psychology Professor  
  Email: [shagriff@fiu.edu]

### Graduate Mentors
- **Enshi Zhang**  
  Graduate Mentor  
  GitHub: https://github.com/coolsoda
  Email: [ezhan004@fiu.edu]

- **Rahmina Rubaiat**  
  Graduate Mentor  
  GitHub: https://github.com/RahminaRubaiat
  Email: [erruba005@fiu.edu]

Their insights into system design, software architecture, and project execution were invaluable throughout the development process.

## 📬 Contact

For questions, contributions, or general discussion about EchoGuard:

- **GitHub:** https://github.com/abner577  
- **Email:** abner07282005@gmail.com  
- **LinkedIn:** https://www.linkedin.com/in/abner-rodriguez-/ 

### 🔐 About the Gemini Prompt

The Gemini prompt included in this repository and demonstrated in the project videos is a **placeholder version**.  
The full production prompt is intentionally not exposed in the codebase.

If you are interested in the **specific prompt logic**, design decisions, or how it was structured, feel free to reach out. I would be happy to share it with you.

---
