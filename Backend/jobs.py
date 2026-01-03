from flask import Blueprint, request, jsonify

jobs_bp = Blueprint("jobs_bp", __name__)

# Sample off-campus job drives
JOBS = [
    {
        "company": "TCS",
        "city": "Indore",
        "location": "Vijay Nagar",
        "date": "15 Jan 2026",
        "time": "10:00 AM",
        "role": "Data Analyst",
        "package": "4.5 LPA",
        "description": "Work on data analysis, dashboards, and business insights."
    },
    {
        "company": "Infosys",
        "city": "Indore",
        "location": "Vijay Nagar",
        "date": "18 Jan 2026",
        "time": "11:30 AM",
        "role": "Software Engineer",
        "package": "3.6 LPA",
        "description": "Backend development and enterprise application support."
    },
    {
        "company": "Wipro",
        "city": "Bhopal",
        "location": "MP Nagar",
        "date": "20 Jan 2026",
        "time": "10:30 AM",
        "role": "Project Engineer",
        "package": "3.5 LPA",
        "description": "Application development and client project delivery."
    }
]

@jobs_bp.route("/", methods=["GET"])
def get_jobs():
    city = request.args.get("city")

    if city:
        filtered_jobs = [job for job in JOBS if job["city"].lower() == city.lower()]
        return jsonify({
            "city": city,
            "jobs": filtered_jobs
        })

    return jsonify({
        "jobs": JOBS
    })
