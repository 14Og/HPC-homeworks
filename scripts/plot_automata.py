import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import sys

plt.style.use(["grid", "notebook", "science"])

def plot_automata(csv_file, title="1D Cellular Automata"):
    data = np.loadtxt(csv_file, delimiter=',')
    
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='binary', interpolation='nearest')
    plt.title(title)
    plt.xlabel("Cell Index")
    plt.ylabel("Time Step")
            
    plt.show()

if __name__ == "__main__":
    try:
        file_path = sys.argv[1]
        plot_automata(file_path)
    except:
        print(f"{sys.argv[0]} $AUTOMATA_RESULT_FILE", file=sys.stderr)
        exit(1)
