import csv
import matplotlib.pyplot as plt

times = []
ears = []

with open("logs.csv") as f:
    for r in csv.DictReader(f):
        times.append(float(r["timestamp_unix"]))
        ears.append(float(r["ear"]))

plt.plot(times, ears)
plt.xlabel("Time")
plt.ylabel("EAR")
plt.title("EAR at detection events")
plt.show()
