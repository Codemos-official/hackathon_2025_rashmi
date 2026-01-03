from flask import Blueprint, jsonify, request
from database import get_db_connection

courses_bp = Blueprint("courses", __name__)

@courses_bp.route("/api/courses", methods=["GET"])
def get_courses():
    domain = request.args.get("domain")
    conn = get_db_connection()
    cursor = conn.cursor()

    if domain:
        cursor.execute(
            "SELECT * FROM courses WHERE domain=%s", (domain,)
        )
    else:
        cursor.execute("SELECT * FROM courses")

    data = cursor.fetchall()
    conn.close()

    return jsonify(data)
