import csv
import os
from collections import Counter
from statistics import mean

LOG_PATH = "logs.csv"

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def read_rows(path):
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        return list(r)

def split_sessions(rows):
    sessions = []
    current = []

    for row in rows:
        reason = (row.get("reason") or "").strip()
        if reason == "marker:session_start":
            if current:
                sessions.append(current)
                current = []
            current.append(row)
        elif reason == "marker:session_end":
            current.append(row)
            sessions.append(current)
            current = []
        else:
            current.append(row)

    if current:
        sessions.append(current)

    return sessions

def summarize_session(rows):
    reasons = []
    fps_vals = []
    ear_vals = []
    mar_vals = []

    for row in rows:
        reason = (row.get("reason") or "").strip()
        if reason.startswith("marker:"):
            continue
        if reason:
            reasons.append(reason)

        fps = safe_float(row.get("fps"))
        if fps is not None:
            fps_vals.append(fps)

        ear = safe_float(row.get("ear"))
        if ear is not None:
            ear_vals.append(ear)

        mar = safe_float(row.get("mar"))
        if mar is not None:
            mar_vals.append(mar)

    counts = Counter(reasons)

    out = {
        "events_total": sum(counts.values()),
        "events_by_reason": dict(counts),
        "avg_fps": mean(fps_vals) if fps_vals else None,
        "avg_ear_on_events": mean(ear_vals) if ear_vals else None,
        "avg_mar_on_events": mean(mar_vals) if mar_vals else None,
    }
    return out

def main():
    rows = read_rows(LOG_PATH)
    if not rows:
        print("No rows found.")
        return

    sessions = split_sessions(rows)

    print(f"Rows: {len(rows)}")
    print(f"Sessions found: {len(sessions)}")
    print()

    all_counts = Counter()
    all_fps = []

    for i, sess in enumerate(sessions, start=1):
        s = summarize_session(sess)
        print(f"Session {i}")
        print("  total events:", s["events_total"])
        print("  events by reason:", s["events_by_reason"])
        if s["avg_fps"] is not None:
            print("  avg fps:", f"{s['avg_fps']:.2f}")
        if s["avg_ear_on_events"] is not None:
            print("  avg ear on events:", f"{s['avg_ear_on_events']:.4f}")
        if s["avg_mar_on_events"] is not None:
            print("  avg mar on events:", f"{s['avg_mar_on_events']:.4f}")
        print()

        all_counts.update(s["events_by_reason"])
        if s["avg_fps"] is not None:
            all_fps.append(s["avg_fps"])

    print("Overall")
    print("  total events:", sum(all_counts.values()))
    print("  events by reason:", dict(all_counts))
    if all_fps:
        print("  avg fps across sessions:", f"{mean(all_fps):.2f}")

    if not os.path.exists(LOG_PATH):
        print("logs.csv not found. Run main.py and generate at least one event.")
        exit(0)

if __name__ == "__main__":
    main()
