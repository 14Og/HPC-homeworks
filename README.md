# HPC Homework 2 — OpenMP

Report covering the tasks from [docs/OpenMP_hw+tasks.pdf.pdf](docs/OpenMP_hw+tasks.pdf.pdf):
`BugReduction`, `BugParFor`, `Pi`, `Car`, `LinearSolver` (`Axisb`), `LeastSquares`.

## Environment

| | |
|---|---|
| CPU | Intel Core Ultra 9 285H, 16 physical cores, 1 thread/core |
| Compiler | gcc/g++ 13 |
| OpenMP | `libgomp` (GCC) |
| Build | CMake, `-Wall -Wextra -Wpedantic -Werror` |

Build and run:
```sh
cmake -B build && cmake --build build
./build/tasks/Pi 1 2 4 8 16          # most tasks take a thread-count list
```

## Summary

| Task | Speedup (1 → 16 threads) |
|---|---|
| BugReduction | n/a |
| BugParFor | n/a |
| Pi | 7.6× |
| Car | **7.8×** |
| LinearSolver (Axisb) | 5.5× |
| LeastSquares | 6.0× |

---

## 1. BugReduction — dot product (5 pts)

Source: [tasks/BugReduction.c](tasks/BugReduction.c)

**Bugs in the original**:
1. `#pragma omp for reduction(+ : sum)` was placed inside a function called **outside any parallel region** — an orphaned worksharing construct that falls back to sequential execution.
2. `main` discarded the return value of `dotprod` and printed its own (uninitialised) `sum = 0`.

**Fix**: combined into `#pragma omp parallel for reduction(+ : sum)` so the directive both creates the team and distributes the loop. The caller now uses the returned value.

**OpenMP concept demonstrated**: `reduction(+ : sum)` gives each thread a private accumulator, then combines them with `+` at the end of the region. No race, no atomic needed.

## 2. BugParFor — parallel for with thread ID (5 pts)

Source: [tasks/BugParFor.c](tasks/BugParFor.c)

**Bugs in the original**:
1. `#pragma omp parallel` (no `for`) wrapped a plain `for` loop → every thread ran the full loop, producing N × `num_threads` iterations instead of N.
2. `schedule(static, chunk)` attached to a bare `omp parallel` is a semantic error — the clause belongs on a work-sharing construct.
3. `tid` was a shared variable, racing between threads.

**Fix**: changed the directive to `#pragma omp parallel for shared(a, b, c, chunk) private(i, tid) schedule(static, chunk)`, making the loop a real work-sharing construct with `tid` privatised per thread.

## 3. Pi — numerical integration (15 pts)

Source: [tasks/Pi.cc](tasks/Pi.cc)

Computes $\pi$ via midpoint-rule quadrature of

$$\int_0^1 \frac{4}{1 + x^2}dx = \pi$$

with $N = 10^8$ subintervals.

**Parallelization**: a single `#pragma omp parallel for reduction(+ : sum) schedule(static)` wraps the sum. Every iteration is a division and a couple of adds; no dependencies between iterations.

| threads | time (s) | speedup |
|---|---|---|
| 1  | 0.418 | 1.00× |
| 2  | 0.249 | 1.68× |
| 4  | 0.145 | 2.89× |
| 8  | 0.082 | 5.08× |
| 16 | 0.055 | **7.63×** |

## 4. Car — PPM column-shift animation (25 pts)

Source: [tasks/Car.cc](tasks/Car.cc)

Reads `car.ppm`, then produces `width` frames, each with every column cyclically shifted left by one more than the previous — giving the illusion of forward motion. The frames are stitched into `media/car.gif` with `ffmpeg`.

**Algorithmic choice**: rather than physically rotating pixel data each frame, the shift is implemented as an **index offset** on read:
```cpp
auto srcNum = (x + aImg.x - shift % aImg.x) % aImg.x;
```
No data mutation between frames → all frames can be generated independently in parallel.

**Parallelization**: `#pragma omp parallel for` over the outer frame loop. Each thread opens its own `ofstream` (thread-local stack variable), writes its own uniquely-named file, and reads from the shared, immutable `PPMImage`. A `std::atomic_bool` error flag is used to propagate file-open failures out of the parallel region (OpenMP cannot carry exceptions across region boundaries).

| threads | time (s) | speedup |
|---|---|---|
| 1  | 1.734 | 1.00× |
| 2  | 0.909 | 1.91× |
| 4  | 0.498 | 3.48× |
| 8  | 0.305 | 5.69× |
| 16 | 0.222 | **7.81×** |

The workload is dominated by integer-to-string formatting (each pixel emits three decimal values), which is CPU-bound and parallelises well. Disk I/O does not bottleneck at this size (~45 MB total output across 303 frames).

Final GIF assembly (not parallelised, external tool). Run from `media/` to reproduce `car.gif`:
```sh
cd media && \
ffmpeg -f image2 -c:v ppm -i 'frames/frame_%d' -vf "palettegen=stats_mode=diff" -y palette.png && \
ffmpeg -framerate 30 -f image2 -c:v ppm -i 'frames/frame_%d' -i palette.png \
       -lavfi "[0:v]format=rgb24[v];[v][1:v]paletteuse=dither=none" -y car.gif && \
rm palette.png
```

## 5. LinearSolver (Axisb) — Jacobi iterative solver (25 pts)

Source: [tasks/LinearSolver.cc](tasks/LinearSolver.cc)

Solves $A\mathbf{x} = \mathbf{b}$ for a dense strictly diagonally dominant system using the Jacobi method. Matrix and vector storage use Eigen.

**Algorithm**:

$$x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j \ne i} a_{ij} \cdot x_j^{(k)} \right)$$

Each $x_i^{(k+1)}$ depends only on $\mathbf{x}^{(k)}$ (the previous iteration's full vector), so within one iteration the $n$ row updates are fully independent. Ping-pong buffers (`xOld`, `xNew`) prevent accidentally turning this into Gauss-Seidel.

**Parallelization**:
```cpp
#pragma omp parallel for reduction(+ : diffSq) schedule(static)
for (Index i = 0; i < n; ++i) {
    double sum = aA.row(i).dot(xOld) - aA(i, i) * xOld(i);
    xNew(i)    = (aB(i) - sum) / aA(i, i);
    double d   = xNew(i) - xOld(i);
    diffSq    += d * d;
}
```
The per-iteration convergence check ($L^2$ norm of $\mathbf{x}_\text{new} - \mathbf{x}_\text{old}$) is merged into the same parallel loop, reusing the reduction.

`Eigen::setNbThreads(1)` is called at startup so Eigen's internal threading doesn't nest inside our own.

| threads | time (s) | speedup |
|---|---|---|
| 1  | 1.321 | 1.00× |
| 2  | 0.735 | 1.80× |
| 4  | 0.465 | 2.84× |
| 8  | 0.261 | 5.06× |
| 16 | 0.242 | **5.46×** |

Scaling plateaus between 8 and 16 threads. This is **memory-bandwidth-limited** — each iteration reads the full 800 MB matrix plus two vectors, and the arithmetic intensity (FLOPs per byte) is too low to saturate 16 cores worth of compute. The residual $\|A\mathbf{x} - \mathbf{b}\| \approx 1.06 \cdot 10^{-8}$ is identical across all thread counts, confirming numerical determinism.

## 6. LeastSquares — linear regression via gradient descent (25 pts)

Source: [tasks/LeastSquares.cc](tasks/LeastSquares.cc)

Fits $f(x) = a x + b$ to $N = 10^6$ noisy samples generated from $y_i = 2.5 \, x_i - 1 + \mathrm{noise}()$.

**Algorithm**: gradient descent on the MSE loss

$$L(a, b) = \frac{1}{N} \sum_{i=1}^{N} (a x_i + b - y_i)^2$$

whose gradients both reduce to a sum over the residual $r_i = a x_i + b - y_i$:

$$\frac{\partial L}{\partial a} = \frac{2}{N} \sum_{i=1}^{N} x_i r_i \qquad \frac{\partial L}{\partial b} = \frac{2}{N} \sum_{i=1}^{N} r_i$$

**Parallelization**: one pass over the data computes both gradients simultaneously using OpenMP's **multi-variable reduction** clause:
```cpp
#pragma omp parallel for reduction(+ : gradA, gradB) schedule(static)
for (Index i = 0; i < n; ++i) {
    double r = a * aX(i) + b - aY(i);
    gradA += aX(i) * r;
    gradB += r;
}
```
Each thread gets private copies of both accumulators; both are combined at the region exit.

| threads | time (s) | speedup |
|---|---|---|
| 1  | 13.82 | 1.00× |
| 2  | 8.50  | 1.63× |
| 4  | 4.81  | 2.87× |
| 8  | 2.32  | 5.95× |
| 16 | 2.31  | **5.99×** |

Recovered parameters: $a = 2.4999$, $b = -0.999966$ — essentially the true values, with the small offset attributable to finite-sample noise. Iteration count is identical (209) across all thread counts.

Scaling plateaus at 8 threads for the same reason as the Jacobi solver: the compute per element (two multiplies, two adds) is too light to stay CPU-bound once DRAM bandwidth is saturated.