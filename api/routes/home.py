from flask import Blueprint, jsonify

home_bp = Blueprint("home", __name__, url_prefix="/api")

# /api/home - Example endpoint
@home_bp.route("/home", methods=['GET'])
def return_home():
    return jsonify({
        'message': "Example Message",
    })