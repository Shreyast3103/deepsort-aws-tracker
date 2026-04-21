import os
import json
import time
import subprocess
import logging
import boto3
import watchtower

# -------------------------------
# Parameter Store
# -------------------------------
ssm = boto3.client("ssm", region_name="us-east-1")

def get(n):
    return ssm.get_parameter(Name=n)["Parameter"]["Value"]

AWS_REGION = get("/deepsort/AWS_REGION")
S3_BUCKET = get("/deepsort/S3_BUCKET_NAME")
SQS_URL = get("/deepsort/SQS_QUEUE_URL")
CW_LOG = get("/deepsort/CW_LOG_GROUP")
SNS_ARN = get("/deepsort/SNS_TOPIC_ARN")

# -------------------------------
# Paths
# -------------------------------
BASE = os.getcwd()
UP = os.path.join(BASE, "uploads")
OUT = os.path.join(BASE, "output")
JOB = os.path.join(BASE, "jobs")

MODEL_YOLO = os.path.join(BASE, "models", "yolov10n.onnx")
MODEL_REID = os.path.join(BASE, "models", "reid.onnx")

os.makedirs(UP, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
os.makedirs(JOB, exist_ok=True)

# -------------------------------
# AWS
# -------------------------------
s3 = boto3.client("s3", region_name=AWS_REGION)
sqs = boto3.client("sqs", region_name=AWS_REGION)
sns = boto3.client("sns", region_name=AWS_REGION)
logs = boto3.client("logs", region_name=AWS_REGION)

# -------------------------------
# Logging
# -------------------------------
logger = logging.getLogger("worker")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    h = logging.StreamHandler()
    h.setFormatter(formatter)
    logger.addHandler(h)

    try:
        cw_handler = watchtower.CloudWatchLogHandler(
            log_group_name=CW_LOG,
            log_stream_name="worker",
            boto3_client=logs
        )
        cw_handler.setFormatter(formatter)
        logger.addHandler(cw_handler)
    except Exception as e:
        print(f"CloudWatch setup failed: {e}")

# -------------------------------
# Helpers
# -------------------------------
def job_path(job_id):
    return os.path.join(JOB, f"{job_id}.json")

def update_job(job_id, status, filename="", output=None):
    data = {
        "job_id": job_id,
        "status": status,
        "filename": filename,
        "output_video": output
    }
    with open(job_path(job_id), "w") as f:
        json.dump(data, f)

def email(job_id, file, summary_text, output_video=None):
    try:
        msg = f"""Job {job_id} completed

File: {file}

=== SUMMARY ===
{summary_text}
"""
        if output_video:
            msg += f"\nOutput file: {output_video}\n"

        sns.publish(
            TopicArn=SNS_ARN,
            Subject="Video Processing Completed",
            Message=msg
        )
        logger.info(f"SNS email sent for {job_id}")
    except Exception as e:
        logger.error(f"SNS failed: {e}")

def find_files(job_id):
    output_video = None
    timings_file = None

    for f in os.listdir(OUT):
        if job_id not in f:
            continue

        if f.startswith("serial_out_"):
            output_video = f
        elif f.startswith("timings_"):
            timings_file = f

    return output_video, timings_file

def read_summary(timings_file):
    if not timings_file:
        return "No summary available"
    path = os.path.join(OUT, timings_file)
    if not os.path.exists(path):
        return "No summary available"
    return open(path).read()

# -------------------------------
# Worker Loop
# -------------------------------
while True:
    try:
        r = sqs.receive_message(
            QueueUrl=SQS_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20
        )

        if "Messages" not in r:
            continue

        m = r["Messages"][0]
        body = json.loads(m["Body"])

        jid = body["job_id"]
        fname = body["filename"]

        logger.info(f"Processing {jid}")

        update_job(jid, "Processing", fname)

        local_input = os.path.join(UP, fname)
        s3.download_file(S3_BUCKET, body["input_s3_key"], local_input)

        process = subprocess.Popen(
            [
                "python3",
                "src/yolo_deepsort.py",
                local_input,
                MODEL_YOLO,
                MODEL_REID
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in process.stdout:
            logger.info(line.strip())

        process.wait()

        if process.returncode != 0:
            update_job(jid, "failed", fname)

            # ✅ IMPORTANT: delete even if failed
            sqs.delete_message(
                QueueUrl=SQS_URL,
                ReceiptHandle=m["ReceiptHandle"]
            )
            continue

        output_video, timings_file = find_files(jid)

        if output_video:
            s3.upload_file(
            os.path.join(OUT, output_video),
            S3_BUCKET,
            f"outputs/{jid}/{output_video}"
        )

        summary = read_summary(timings_file)

        # ✅ FINAL FIX
        update_job(jid, "Completed", fname, output_video)

        email(jid, fname, summary, output_video)

        sqs.delete_message(
            QueueUrl=SQS_URL,
            ReceiptHandle=m["ReceiptHandle"]
        )

        logger.info(f"Completed {jid}")

    except Exception as e:
        logger.exception(f"Worker error: {e}")
        time.sleep(3)