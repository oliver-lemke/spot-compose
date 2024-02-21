import os

import numpy as np

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def upload_files():
    # Check if the files are present in the request
    if "points" not in request.files or "colors" not in request.files:
        return jsonify({"message": "Both files (points and colors) are required"}), 400

    points_file = request.files["points"]
    colors_file = request.files["colors"]

    points = np.load(points_file)
    colors = np.load(colors_file)

    print(points.shape)
    print(colors.shape)

    print("Received dict:", type(request.args))

    output = np.ones((5, 2))
    os.makedirs("tmp", exist_ok=True)
    with open("tmp/output.npy", "wb") as f:
        np.save(f, output)

    return send_file("tmp/output.npy", as_attachment=True, download_name="output.npy")


if __name__ == "__main__":
    app.run(port=5001, debug=True)
