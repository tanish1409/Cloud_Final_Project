"""
Pub/Sub utility functions shared across all microservices.
"""
import os, json, logging, base64
from google.cloud import pubsub_v1
logger = logging.getLogger(__name__)
PROJECT_ID = os.environ.get("GCP_PROJECT", "ngsim-scenarios-proj")

def publish(topic_name, run_id, **extra):
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, topic_name)
    payload = {"run_id": run_id, **extra}
    data = json.dumps(payload).encode("utf-8")
    msg_id = publisher.publish(topic_path, data).result()
    logger.info(f"Published to {topic_name}: {payload} (msg_id={msg_id})")
    return msg_id

def parse_pubsub_message(request):
    envelope = request.get_json(silent=True) or {}
    if "message" not in envelope:
        raise ValueError("No Pub/Sub message in request body")
    msg_data = envelope["message"].get("data", "")
    decoded = base64.b64decode(msg_data).decode("utf-8")
    payload = json.loads(decoded)
    logger.info(f"Received Pub/Sub message: {payload}")
    return payload
