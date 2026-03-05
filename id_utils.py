import uuid

def new_uuid(prefix: str | None = None) -> str:
    u = str(uuid.uuid4())
    return f"{prefix}_{u}" if prefix else u
