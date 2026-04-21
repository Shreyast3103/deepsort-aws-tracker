import os
import subprocess
import uuid
import logging
import boto3
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

# -------------------------------
# Logging (for CloudWatch later)
# -------------------------------
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logging.info("App started successfully") 
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
# S3 CONFIG
# -------------------------------
BUCKET_NAME = "deepsort-tracker-bucket"   # ⚠️ change if needed
s3 = boto3.client('s3')

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

    logging.info(f"Uploaded file: {unique_name}")

    # -------------------------------
    # Upload INPUT to S3
    # -------------------------------
    try:
        s3.upload_file(filepath, BUCKET_NAME, f"input/{unique_name}")
        logging.info("Uploaded input video to S3")
    except Exception as e:
        logging.error(f"S3 upload failed: {e}")

    # -------------------------------
    # Run DeepSORT
    # -------------------------------
    logging.info("Starting processing...")

    subprocess.run([
        "python3",
        "src/yolo_deepsort.py",
        filepath,
        MODEL_YOLO,
        MODEL_REID
    ])

    logging.info("Processing completed")

    # -------------------------------
    # Get latest output
    # -------------------------------
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
    output_path = os.path.join(OUTPUT_FOLDER, output_video)

    # -------------------------------
    # Upload OUTPUT to S3
    # -------------------------------
    try:
        s3.upload_file(output_path, BUCKET_NAME, f"output/{output_video}")
        logging.info("Uploaded output video to S3")
    except Exception as e:
        logging.error(f"S3 output upload failed: {e}")

    # -------------------------------
    # Create S3 URL
    # -------------------------------
    video_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/output/{output_video}"

    # -------------------------------
    # Read metrics
    # -------------------------------
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
        video_file=video_url,
        metrics=metrics
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)