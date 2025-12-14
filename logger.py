import csv
import os
from datetime import datetime, timezone

HEADER = ["timestamp_iso", "timestamp_unix", "ear", "mar", "reason", "fps"]

def append_event(path, ear, mar, reason, fps):
    ts = datetime.now(timezone.utc)
    row = [
        ts.isoformat(),
        f"{ts.timestamp():.3f}",
        f"{ear:.4f}" if ear is not None else "",
        f"{mar:.4f}" if mar is not None else "",
        reason,
        f"{fps:.2f}",
    ]

    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(HEADER)
        w.writerow(row)

def append_marker(path, label):
    ts = datetime.now(timezone.utc)
    row = [
        ts.isoformat(),
        f"{ts.timestamp():.3f}",
        "",
        "",
        f"marker:{label}",
        "",
    ]

    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(HEADER)
        w.writerow(row)
