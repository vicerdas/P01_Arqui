#!/usr/bin/env python3
import subprocess, sys, csv
import matplotlib.pyplot as plt

sizes = [100, 1000, 10000, 100000, 1000000, 10000000]
reps = 30
rows = []

for n in sizes:
    inp = f"data/bench_{n}.dat"
    subprocess.run(["python3", "tools/gen_input.py", str(n), inp, "random", "1"], check=True)

    out_s = f"data/out_scalar_{n}.dat"
    out_v = f"data/out_vector_{n}.dat"
    subprocess.run(["./bin/norm_scalar", inp, out_s, str(reps)], check=True)
    subprocess.run(["./bin/norm_vector", inp, out_v, str(reps)], check=True)

    def read_ms(stats_path):
        with open(stats_path) as f:
            for line in f:
                if line.startswith("kernel_ms="):
                    return float(line.strip().split("=")[1])
        return None

    ms_s = read_ms(out_s + ".stats.txt")
    ms_v = read_ms(out_v + ".stats.txt")
    speedup = ms_s / ms_v if ms_v > 0 else float("nan")
    rows.append((n, ms_s, ms_v, speedup))
    print(f"N={n:>10}  escalar={ms_s:.4f}ms  vectorial={ms_v:.4f}ms  speedup={speedup:.2f}x")

with open("data/benchmark_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["N", "scalar_ms", "vector_ms", "speedup"])
    w.writerows(rows)

ns = [r[0] for r in rows]
speedups = [r[3] for r in rows]

plt.figure(figsize=(8, 5))
plt.plot(ns, speedups, marker="o")
plt.xscale("log")
plt.xlabel("N (tamano del arreglo, escala log)")
plt.ylabel("Speedup (t_escalar / t_vectorial)")
plt.title("Speedup de la version vectorial (AVX2) vs escalar")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.savefig("data/speedup.png", dpi=150)
print("\nGuardado: data/benchmark_results.csv y data/speedup.png")
