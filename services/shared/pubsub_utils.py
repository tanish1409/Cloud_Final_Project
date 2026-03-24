"""
Pub/Sub utility functions shared across all microservices.
Publishes JSON messages containing run_id and GCS paths.
"""
import os
import json
import logging
from google.cloud import pubsub_v1

logger = logging.getLogger(__name__)

PROJECT_ID = os.environ.get("GCP_PROJECT", "ngsim-scenarios-proj")


def publish(topic_name: str, run_id: str, **extra):
    """
    Publish a message to a Pub/Sub topic.
    Message payload is JSON with run_id and any extra key-value pairs.
    """
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, topic_name)

    payload = {"run_id": run_id, **extra}
    data = json.dumps(payload).encode("utf-8")

    future = publisher.publish(topic_path, data)
    msg_id = future.result()
    logger.info(f"Published to {topic_name}: {payload} (msg_id={msg_id})")
    return msg_id


def parse_pubsub_message(request):
    """
    Parse incoming Pub/Sub push message from a Flask request.
    Returns the decoded JSON payload dict.
    """
    envelope = request.get_json(silent=True) or {}
    if "message" not in envelope:
        raise ValueError("No Pub/Sub message in request body")

    import base64
    msg_data = envelope["message"].get("data", "")
    decoded = base64.b64decode(msg_data).decode("utf-8")
    payload = json.loads(decoded)
    logger.info(f"Received Pub/Sub message: {payload}")
    return payload
