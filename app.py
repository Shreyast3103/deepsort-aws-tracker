import os
import json
import uuid
import logging
from flask import Flask, render_template, request, redirect, send_from_directory, jsonify
import boto3
import watchtower

app = Flask(__name__)

# -------------------------------
# Parameter Store
# -------------------------------
PARAM_PREFIX = "/deepsort"
ssm = boto3.client("ssm", region_name="us-east-1")

def get_param(name):
    return ssm.get_parameter(Name=name)["Parameter"]["Value"]

AWS_REGION = get_param(f"{PARAM_PREFIX}/AWS_REGION")
S3_BUCKET_NAME = get_param(f"{PARAM_PREFIX}/S3_BUCKET_NAME")
CW_LOG_GROUP = get_param(f"{PARAM_PREFIX}/CW_LOG_GROUP")
SQS_QUEUE_URL = get_param(f"{PARAM_PREFIX}/SQS_QUEUE_URL")

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
JOBS_FOLDER = os.path.join(BASE_DIR, "jobs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(JOBS_FOLDER, exist_ok=True)

# -------------------------------
# AWS Clients
# -------------------------------
s3 = boto3.client("s3", region_name=AWS_REGION)
sqs = boto3.client("sqs", region_name=AWS_REGION)
logs = boto3.client("logs", region_name=AWS_REGION)

# -------------------------------
# Logging
# -------------------------------
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    try:
        cw = watchtower.CloudWatchLogHandler(
            log_group_name=CW_LOG_GROUP,
            log_stream_name="flask",
            boto3_client=logs
        )
        cw.setFormatter(formatter)
        logger.addHandler(cw)
    except Exception as e:
        print("CloudWatch error:", e)

# -------------------------------
# Helpers
# -------------------------------
def job_file(job_id):
    return os.path.join(JOBS_FOLDER, f"{job_id}.json")

def save_job(job_id, status, msg="", output=None):
    with open(job_file(job_id), "w") as f:
        json.dump({
            "job_id": job_id,
            "status": status,
            "message": msg,
            "progress": 0,   # NEW
            "output_video": output
        }, f)

def load_job(job_id):
    f = job_file(job_id)
    if not os.path.exists(f):
        return None
    return json.load(open(f))

def queue_count():
    try:
        r = sqs.get_queue_attributes(
            QueueUrl=SQS_QUEUE_URL,
            AttributeNames=[
                "ApproximateNumberOfMessages",
                "ApproximateNumberOfMessagesNotVisible"
            ]
        )
        a = r["Attributes"]
        available = int(a.get("ApproximateNumberOfMessages", 0))
        in_flight = int(a.get("ApproximateNumberOfMessagesNotVisible", 0))

        return {
            "available": available,
            "in_flight": in_flight,
            "total": available + in_flight
        }
    except:
        return {"available": 0, "in_flight": 0, "total": 0}

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():

    files = request.files.getlist("video")

    if not files or files[0].filename == "":
        return redirect("/")

    first_job_id = None
    first_filename = None

    for file in files:
        job_id = str(uuid.uuid4())
        unique_name = f"{job_id}_{file.filename}"
        path = os.path.join(UPLOAD_FOLDER, unique_name)

        file.save(path)

        save_job(job_id, "queued", "Waiting in queue")

        # Upload to S3
        s3.upload_file(path, S3_BUCKET_NAME, f"uploads/{unique_name}")

        # Send to SQS
        sqs.send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=json.dumps({
                "job_id": job_id,
                "filename": unique_name,
                "input_s3_key": f"uploads/{unique_name}"
            })
        )

        # store first job for UI
        if first_job_id is None:
            first_job_id = job_id
            first_filename = unique_name

    queue_status = queue_count()

    return render_template(
        "result.html",
        job_id=first_job_id,
        filename=first_filename,
        status="Queued",
        queue=queue_status,
        s3_links=None
    )

@app.route("/status/<job_id>")
def status(job_id):
    d = load_job(job_id)

    if not d:
        return jsonify({"status": "unknown"})

    d["queue"] = queue_count()

    # 🔥 ADD THIS: find next processing job
    next_job = None
    for f in os.listdir(JOBS_FOLDER):
        j = json.load(open(os.path.join(JOBS_FOLDER, f)))
        if j.get("status") == "Processing":
            next_job = j["job_id"]
            break

    d["next_job"] = next_job

    return jsonify(d)

@app.route("/output/<f>")
def out(f):
    return send_from_directory(OUTPUT_FOLDER, f)

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)