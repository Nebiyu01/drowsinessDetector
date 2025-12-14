import csv
from collections import Counter
from statistics import mean

def main():
    with open("logs.csv", "r") as f:
        rows = list(csv.DictReader(f))

    reasons = [r["reason"] for r in rows]
    ears = [float(r["ear"]) for r in rows if r["ear"]]
    mars = [float(r["mar"]) for r in rows if r["mar"]]
    fps_vals = [float(r["fps"]) for r in rows if r["fps"]]

    print("Total events:", len(rows))
    print("Events by reason:", Counter(reasons))
    print("Avg EAR on events:", round(mean(ears), 4))
    print("Avg MAR on events:", round(mean(mars), 4))
    print("Avg FPS:", round(mean(fps_vals), 2))

if __name__ == "__main__":
    main()
