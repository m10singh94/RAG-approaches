from __future__ import annotations

from functools import lru_cache

import boto3
from botocore.config import Config

from rag_approaches.config import get_settings

_CLIENT_CONFIG = Config(retries={"mode": "adaptive", "max_attempts": 8}, read_timeout=300)


@lru_cache
def session() -> boto3.Session:
    s = get_settings()
    return boto3.Session(profile_name=s.profile, region_name=s.region)


@lru_cache
def client(service: str):
    return session().client(service, config=_CLIENT_CONFIG)


@lru_cache
def account_id() -> str:
    return client("sts").get_caller_identity()["Account"]


def model_arn(model_id: str) -> str:
    """RetrieveAndGenerate wants an ARN; inference profiles (au./apac./global. prefixes) have a different shape."""
    if model_id.startswith("arn:"):
        return model_id
    region = get_settings().region
    if model_id.split(".")[0] in {"au", "apac", "us", "eu", "jp", "global"}:
        return f"arn:aws:bedrock:{region}:{account_id()}:inference-profile/{model_id}"
    return f"arn:aws:bedrock:{region}::foundation-model/{model_id}"
