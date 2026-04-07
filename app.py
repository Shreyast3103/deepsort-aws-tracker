import os
import subprocess
import uuid
import time
import logging
import csv
from flask import Flask, render_template, request, redirect, send_from_directory
from dotenv import load_dotenv
import boto3
import watchtower

load_dotenv()

app = Flask(__name__)

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_YOLO = os.path.join(MODEL_DIR, "yolov10n.onnx")
MODEL_REID = os.path.join(MODEL_DIR, "reid.onnx")

# -------------------------------
# AWS Config
# -------------------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
CW_LOG_GROUP = os.getenv("CW_LOG_GROUP", "deepsort-app-logs")

s3_client = boto3.client("s3", region_name=AWS_REGION)
cloudwatch = boto3.client("cloudwatch", region_name=AWS_REGION)
logs_client = boto3.client("logs", region_name=AWS_REGION)

# -------------------------------
# Logging
# -------------------------------
logger = logging.getLogger("deepsort_app")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        cw_handler = watchtower.CloudWatchLogHandler(
            log_group_name=CW_LOG_GROUP,
            log_stream_name=f"run-{uuid.uuid4()}",
            boto3_client=logs_client,
        )
        cw_handler.setFormatter(formatter)
        logger.addHandler(cw_handler)
    except Exception as e:
        print(f"CloudWatch setup failed: {e}")

# -------------------------------
# Helpers
# -------------------------------
def upload_to_s3(local_path, s3_key):
    if not S3_BUCKET_NAME:
        return None
    s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_key)
    return f"s3://{S3_BUCKET_NAME}/{s3_key}"

def generate_presigned_url(s3_key, expiry=3600):
    if not S3_BUCKET_NAME:
        return None
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET_NAME, "Key": s3_key},
        ExpiresIn=expiry
    )

def put_metric(name, value, unit="Count"):
    try:
        cloudwatch.put_metric_data(
            Namespace="DeepSORTTracker",
            MetricData=[{
                "MetricName": name,
                "Value": value,
                "Unit": unit
            }]
        )
    except Exception:
        pass

def safe_float(v):
    try:
        return float(v)
    except:
        return None

def download_from_s3(s3_key, local_path):
    if not S3_BUCKET_NAME:
        return None
    s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)
    return local_path

# -------------------------------
# Home
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

    logger.info("=== Upload Received ===")
    logger.info(f"Processing: {filepath}")
    put_metric("VideoUploads", 1)

    # Upload input to S3
    try:
        upload_to_s3(filepath, f"uploads/{unique_name}")
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")

    start_time = time.time()

    # -------------------------------
    # Ensure models
    # -------------------------------
    try:
        if not os.path.exists(MODEL_YOLO):
            download_from_s3("models/yolov10n.onnx", MODEL_YOLO)

        if not os.path.exists(MODEL_REID):
            download_from_s3("models/reid.onnx", MODEL_REID)

    except Exception as e:
        return "Model download failed"

    if not os.path.exists(MODEL_YOLO) or not os.path.exists(MODEL_REID):
        return "Model missing"

    logger.info("=== Models Ready ===")

    # -------------------------------
    # Run YOLO + DeepSORT
    # -------------------------------
    logger.info("=== Processing Started ===")
    process = subprocess.Popen(
        ["python", "-u", "src/yolo_deepsort.py", filepath, MODEL_YOLO, MODEL_REID],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end="")
        logger.info(line.strip())

    return_code = process.wait()
    logger.info("=== Processing Completed ===")
    
    if return_code != 0:
        logger.error("Processing failed")
        put_metric("ProcessingFailure", 1)
        return "Processing failed"

    # SUCCESS CASE
    put_metric("ProcessingSuccess", 1)

    duration = time.time() - start_time
    put_metric("ProcessingTimeSeconds", duration, "Seconds")

    # -------------------------------
    # Get output files
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

    debug_files = sorted(
        [f for f in os.listdir(OUTPUT_FOLDER) if f.startswith("debug_log_")],
        key=lambda x: os.path.getmtime(os.path.join(OUTPUT_FOLDER, x)),
        reverse=True
    )

    if not output_files:
        return "No output generated"

    output_video = output_files[0]

    # -------------------------------
    # Read metrics
    # -------------------------------
    metrics = {}
    if timings_files:
        with open(os.path.join(OUTPUT_FOLDER, timings_files[0])) as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    metrics[k.strip()] = v.strip()

    # -------------------------------
    # Graph data (dummy for now)
    # -------------------------------
    frame_numbers = list(range(1, 50))
    frame_times = [0.1 + (i % 5) * 0.01 for i in frame_numbers]

    # -------------------------------
    # Upload outputs
    # -------------------------------
    s3_links = {}

    try:
        key = f"outputs/{output_video}"
        upload_to_s3(os.path.join(OUTPUT_FOLDER, output_video), key)
        url = generate_presigned_url(key)
        if url:
            s3_links["video"] = url
    except Exception as e:
        logger.error(f"S3 output upload failed: {e}")

    if timings_files:
        try:
            name = timings_files[0]
            key = f"logs/{name}"
            upload_to_s3(os.path.join(OUTPUT_FOLDER, name), key)
            s3_links["timings"] = generate_presigned_url(key)
        except:
            pass

    if debug_files:
        try:
            name = debug_files[0]
            key = f"logs/{name}"
            upload_to_s3(os.path.join(OUTPUT_FOLDER, name), key)
            s3_links["debug_log"] = generate_presigned_url(key)
        except:
            pass

    # -------------------------------
    # Fallback video (optional)
    # -------------------------------
    if "video" in s3_links:
        video_url = s3_links["video"]
    else:
        video_url = f"/output/{output_video}"

    return render_template(
        "result.html",
        video_file=video_url,
        metrics=metrics,
        s3_links=s3_links,
        frame_numbers=frame_numbers,
        frame_times=frame_times
    )

# -------------------------------
# Serve output locally
# -------------------------------
@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    logger.info("Starting server...")
    app.run(host="0.0.0.0", port=5000, debug=True)