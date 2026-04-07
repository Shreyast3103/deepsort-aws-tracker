import os
import subprocess
import uuid
import time
import logging
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
MODEL_YOLO = os.path.join(BASE_DIR, "models", "yolov10n.onnx")
MODEL_REID = os.path.join(BASE_DIR, "models", "reid.onnx")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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
            log_stream_name="flask-app",
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
    cloudwatch.put_metric_data(
        Namespace="DeepSORTTracker",
        MetricData=[{
            "MetricName": name,
            "Value": value,
            "Unit": unit
        }]
    )

def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

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

    logger.info(f"Processing: {filepath}")
    put_metric("VideoUploads", 1)

    # Upload input video to S3
    try:
        upload_to_s3(filepath, f"uploads/{unique_name}")
    except Exception as e:
        logger.error(f"S3 input upload failed: {e}")

    start_time = time.time()

    result = subprocess.run(
        [
            "python",
            "src/yolo_deepsort.py",
            filepath,
            MODEL_YOLO,
            MODEL_REID
        ],
        capture_output=True,
        text=True
    )

    logger.info(result.stdout)
    if result.stderr:
        logger.error(result.stderr)

    if result.returncode != 0:
        put_metric("ProcessingFailure", 1)
        return "Processing failed."

    duration = time.time() - start_time
    put_metric("ProcessingSuccess", 1)
    put_metric("ProcessingTimeSeconds", duration, "Seconds")

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

    debug_files = sorted(
        [f for f in os.listdir(OUTPUT_FOLDER) if f.startswith("debug_log_")],
        key=lambda x: os.path.getmtime(os.path.join(OUTPUT_FOLDER, x)),
        reverse=True
    )

    if not output_files:
        return "No output video generated."

    output_video = output_files[0]
    metrics = {}

    # -------------------------------
    # Performance metrics
    # -------------------------------
    if timings_files:
        timings_path = os.path.join(OUTPUT_FOLDER, timings_files[0])
        if os.path.exists(timings_path):
            with open(timings_path, "r") as f:
                for line in f:
                    if ":" in line:
                        key, value = line.strip().split(":", 1)
                        metrics[key.strip()] = value.strip()

    # Send numeric metrics to CloudWatch
    for key, value in metrics.items():
        numeric_value = safe_float(value)
        if numeric_value is not None:
            metric_name = key.replace(" ", "_").replace("-", "_")
            try:
                put_metric(metric_name, numeric_value, unit="Count")
            except Exception as e:
                logger.warning(f"Could not send metric {metric_name}: {e}")

    # Upload output files to S3
    s3_links = {}

    try:
        output_key = f"outputs/{output_video}"
        upload_to_s3(os.path.join(OUTPUT_FOLDER, output_video), output_key)
        s3_links["video"] = generate_presigned_url(output_key)
    except Exception as e:
        logger.error(f"Output video upload failed: {e}")

    if timings_files:
        try:
            timings_name = timings_files[0]
            timings_key = f"logs/{timings_name}"
            upload_to_s3(os.path.join(OUTPUT_FOLDER, timings_name), timings_key)
            s3_links["timings"] = generate_presigned_url(timings_key)
        except Exception as e:
            logger.error(f"Timings upload failed: {e}")

    if debug_files:
        try:
            debug_name = debug_files[0]
            debug_key = f"logs/{debug_name}"
            upload_to_s3(os.path.join(OUTPUT_FOLDER, debug_name), debug_key)
            s3_links["debug_log"] = generate_presigned_url(debug_key)
        except Exception as e:
            logger.error(f"Debug log upload failed: {e}")

    video_url = s3_links.get("video")

    return render_template(
        "result.html",
         video_file=video_url,
         metrics=metrics,
         s3_links=s3_links
         )

# -------------------------------
# Serve Output Files
# -------------------------------
@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    logger.info("Starting server...")
    app.run(host="0.0.0.0", port=5000, debug=True)