from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return {"status": "Career Platform Backend Running (No ORM)"}

if __name__ == "__main__":
    app.run(debug=True)
