from flask import Blueprint, request, jsonify

mentors_bp = Blueprint("mentors_bp", __name__)

# Sample mentors data
MENTORS = [
    {
        "id": 1,
        "name": "Priya Sharma",
        "designation": "Senior Software Engineer",
        "experience": "8 years",
        "expertise": ["DSA", "System Design", "Java"],
        "session_duration": "30 minutes",
        "purpose": "Fundamentals clarification",
        "availability": "Weekends"
    },
    {
        "id": 2,
        "name": "Rahul Verma",
        "designation": "Tech Lead",
        "experience": "6 years",
        "expertise": ["Full Stack", "React", "APIs"],
        "session_duration": "30 minutes",
        "purpose": "Career guidance & doubts",
        "availability": "Evenings"
    }
]

@mentors_bp.route("/", methods=["GET"])
def get_mentors():
    return jsonify({
        "mentors": MENTORS
    })

@mentors_bp.route("/book", methods=["POST"])
def book_session():
    data = request.get_json()

    return jsonify({
        "message": "Session booked successfully",
        "mentor_id": data.get("mentor_id"),
        "session_duration": "30 minutes",
        "status": "Confirmed"
    })
