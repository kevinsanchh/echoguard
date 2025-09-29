from flask import Flask, jsonify
from flask_cors import CORS
import os
from flask import Flask, jsonify, request

# app instance
app = Flask(__name__)
CORS(app)

# /api/home
@app.route("/api/home", methods=['GET'])
def return_home():
    return jsonify({
        'message': "Example Message",
    })

@app.route("/api/audio-upload", methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part in the request'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if audio_file:
        # You can save the file or process it here
        filename = audio_file.filename
        print(f"DEBUG: Received audio file: {filename}")
        print(f"DEBUG: File content type: {audio_file.content_type}")

        # Optionally, you can save the file temporarily for inspection
        # For a real application, you'd process this audio (e.g., send to an ML model)
        # save_dir = "uploaded_audio_clips"
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, filename)
        # audio_file.save(save_path)
        # print(f"DEBUG: Saved audio clip to {save_path}")

        return jsonify({'message': f'Audio file {filename} received successfully!'}), 200
    
    return jsonify({'error': 'Something went wrong processing the audio file'}), 500


# To run locally use 'python server.py'
if __name__ == "__main__":
    # Vercel sets the 'VERCEL' environment variable to '1' during deployment.
    # We check if this variable is NOT set before starting the development server.
    if os.environ.get('VERCEL') != '1':
        app.run(debug=True, port=8080)