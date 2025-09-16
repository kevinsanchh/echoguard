from flask import Flask, jsonify
from flask_cors import CORS

# app instance
app = Flask(__name__)
CORS(app)

# /api/home
@app.route("/api/home", methods=['GET'])
def return_home():
    return jsonify({
        'message': "Example Message",
    })


# The block below is used for Development ONLY, comment it out when pushing to GitHub
# To run locally use 'python server.py' and make sure the below block is un-commented
# if __name__ == "__main__":
#     app.run(debug=True, port=8080)