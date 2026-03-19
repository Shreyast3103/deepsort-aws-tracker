import os
import subprocess
import uuid
from flask import Flask, render_template, request, redirect, send_from_directory

app = Flask(__name__)

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
MODEL_YOLO = os.path.join(BASE_DIR, "models", "yolov10n.onnx")
MODEL_REID = os.path.join(BASE_DIR, "models", "reid.onnx")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------------
# Home Page
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# -------------------------------
# Upload & Process
# -------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return redirect("/")

    file = request.files["video"]
    if file.filename == "":
        return redirect("/")

    unique_name = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(filepath)

    print("Processing:", filepath)

    subprocess.run([
        "python3",
        "src/yolo_deepsort.py",
        filepath,
        MODEL_YOLO,
        MODEL_REID
    ])

    # Get newest output video
    output_files = sorted(
        [f for f in os.listdir(OUTPUT_FOLDER) if f.startswith("serial_out_")],
        key=lambda x: os.path.getmtime(os.path.join(OUTPUT_FOLDER, x)),
        reverse=True
    )

    timings_files = sorted(
        [f for f in os.listdir(OUTPUT_FOLDER) if f.startswith("timings_")],
        key=lambda x: os.path.getmtime(os.path.join(OUTPUT_FOLDER, x)),
        reverse=True
    )

    if not output_files:
        return "No output video generated."

    output_video = output_files[0]
    metrics = {}

    if timings_files:
        timings_path = os.path.join(OUTPUT_FOLDER, timings_files[0])
        if os.path.exists(timings_path):
            with open(timings_path, "r") as f:
                for line in f:
                    if ":" in line:
                        key, value = line.strip().split(":", 1)
                        metrics[key.strip()] = value.strip()

    return render_template(
        "result.html",
        video_file=output_video,
        metrics=metrics
    )

# -------------------------------
# Serve Output Files
# -------------------------------
@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)