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

def save_job(job_id, status, msg="", progress=0, output=None, filename=""):
    with open(job_file(job_id), "w") as f:
        json.dump({
            "job_id": job_id,
            "status": status,
            "message": msg,
            "progress": progress,
            "output_video": output,
            "filename": filename
        }, f)

def load_job(job_id):
    f = job_file(job_id)
    if not os.path.exists(f):
        return None
    return json.load(open(f))


# -------------------------------
# Queue status (SQS)
# -------------------------------
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

    except Exception as e:
        print("Queue error:", e)
        return {"available": 0, "in_flight": 0, "total": 0}


# -------------------------------
# Get all jobs
# -------------------------------
def get_all_jobs():
    jobs = []

    if not os.path.exists(JOBS_FOLDER):
        return jobs

    for f in os.listdir(JOBS_FOLDER):
        path = os.path.join(JOBS_FOLDER, f)

        try:
            with open(path, "r") as file:
                jobs.append(json.load(file))
        except:
            pass

    return jobs


# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("video")

    if not files:
        return redirect("/")

    job_ids = []

    for file in files:
        if file.filename == "":
            continue

        job_id = str(uuid.uuid4())
        name = f"{job_id}_{file.filename}"

        path = os.path.join(UPLOAD_FOLDER, name)
        file.save(path)

        # ✅ ALWAYS lowercase status
        save_job(job_id, "queued", filename=name)

        # Upload to S3
        s3.upload_file(path, S3_BUCKET_NAME, f"uploads/{name}")

        # Send to SQS
        sqs.send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=json.dumps({
                "job_id": job_id,
                "filename": name,
                "input_s3_key": f"uploads/{name}"
            })
        )

        job_ids.append(job_id)

    return redirect(f"/results?jobs={','.join(job_ids)}")

@app.route("/results")
def results():
    job_ids = request.args.get("jobs", "").split(",")

    jobs = []
    all_completed = True

    for job_id in job_ids:
        d = load_job(job_id)
        if not d:
            continue

        jobs.append(d)

        if d.get("status") != "completed":
            all_completed = False

    return render_template(
        "results.html",
        jobs=jobs,
        all_completed=all_completed
    )

@app.route("/status/<job_id>")
def status(job_id):
    job = load_job(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    all_jobs = get_all_jobs()

    next_job = None

    for j in all_jobs:
        if j.get("status") != "completed":
            next_job = j.get("job_id")
            break

    return jsonify({
        "job_id": job_id,
        "filename": job.get("filename", ""),
        "status": job.get("status", "queued"),
        "queue": queue_count()
    })


@app.route("/output/<f>")
def out(f):
    return send_from_directory(OUTPUT_FOLDER, f)


@app.route("/track/<job_id>")
def track(job_id):
    d = load_job(job_id)

    if not d:
        return "Job not found"

    return render_template(
        "result.html",
        job_id=job_id,
        filename=d.get("filename", ""),
        status=d.get("status", "queued"),
        queue=queue_count(),
        s3_links=None,
        metrics=None
    )


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)