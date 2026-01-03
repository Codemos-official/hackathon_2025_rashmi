from flask import Blueprint, request, jsonify

resume_bp = Blueprint("resume_bp", __name__)

RESUME_SUGGESTIONS = {
    "Data Scientist": [
        "Add Python, Pandas, NumPy",
        "Mention ML projects",
        "Include statistics"
    ],
    "Data Analyst": [
        "Add SQL, Excel, Power BI",
        "Mention dashboards",
        "Quantify insights"
    ],
    "Full Stack Web Developer": [
        "Add React, Node.js, REST APIs",
        "Include GitHub links",
        "Mention deployment experience"
    ],
    "App Developer": [
        "Add Flutter / Android skills",
        "Mention Play Store apps",
        "Include UI screenshots"
    ]
}

@resume_bp.route("/suggestions", methods=["POST"])
def suggestions():
    data = request.get_json()
    role = data.get("role")
    return jsonify({
        "role": role,
        "suggestions": RESUME_SUGGESTIONS.get(role, [])
    })
