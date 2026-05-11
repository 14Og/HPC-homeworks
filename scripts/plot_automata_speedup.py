import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import sys

plt.style.use(["grid", "notebook", "science"])

def plot_speedup(csv_file):
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    procs = data[:, 0]
    times = data[:, 1]
    
    t1 = times[0]
    speedup = t1 / times
    
    plt.figure(figsize=(10, 8))
    plt.plot(procs, speedup, marker='o', c='g', linewidth=2, label='Actual Speedup')
    
    plt.plot(procs, procs, linestyle='--', color='gray', label='Ideal Speedup')
    
    plt.title("Cellular Automata MPI Speedup")
    plt.xlabel("Number of Processes")
    plt.ylabel("Speedup (T1 / Tp)")
    plt.xticks(procs)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
        
    plt.show()

if __name__ == "__main__":
    try:
        file_path = sys.argv[1]
        plot_speedup(file_path)
    except:
        print(f"{sys.argv[0]} $AUTOMATA_BENCHMARK_FILE", file=sys.stderr)
        exit(1)
