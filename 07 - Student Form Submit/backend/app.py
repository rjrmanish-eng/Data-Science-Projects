from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

FILE_NAME = "students.xlsx"

# Create Excel file if not exists
if not os.path.exists(FILE_NAME):
    df = pd.DataFrame(columns=[
        "Sr No",
        "Student Name",
        "Father Name",
        "Mother Name",
        "Address",
        "Mobile",
        "Email",
        "Age",
        "Passout Year",
        "Interest"
    ])
    df.to_excel(FILE_NAME, index=False)


@app.route("/")
def home():
    return "Backend Running Successfully 🚀"


@app.route("/submit", methods=["POST"])
def submit():

    data = request.json

    df = pd.read_excel(FILE_NAME)

    # Duplicate Check
    duplicate = df[
        (df["Mobile"].astype(str) == str(data["mobile"])) |
        (df["Email"] == data["email"])
    ]

    if not duplicate.empty:
        return jsonify({
            "status": "error",
            "message": "Duplicate Mobile or Email Found"
        }), 400

    sr_no = len(df) + 1

    new_row = {
        "Sr No": sr_no,
        "Student Name": data["student_name"],
        "Father Name": data["father_name"],
        "Mother Name": data["mother_name"],
        "Address": data["address"],
        "Mobile": data["mobile"],
        "Email": data["email"],
        "Age": data["age"],
        "Passout Year": data["passout_year"],
        "Interest": data["interest"]
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_excel(FILE_NAME, index=False)

    return jsonify({
        "status": "success",
        "message": "Data Saved Successfully"
    })


if __name__ == "__main__":
    app.run(debug=True)