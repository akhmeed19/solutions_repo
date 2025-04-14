# Problem 1

# Interference Patterns on a Water Surface

## 1. Introduction

When two or more waves overlap, they can combine in ways that reinforce each other (constructive interference) or cancel each other out (destructive interference). On a water surface, these phenomena are visibly manifested by ripples forming complex patterns of crests and troughs. 

In this problem, we focus on **circular waves** emitted by multiple point sources arranged at the vertices of a chosen **regular polygon** (e.g., equilateral triangle, square, pentagon, etc.). We will:
- Derive the equations describing individual circular waves.
- Sum these wave contributions at each point of the surface (superposition principle).
- Observe how coherent waves (same amplitude, wavelength, frequency, and a constant phase difference) interfere to produce stable patterns.
- Visualize and analyze the resulting interference patterns.

By studying these patterns, we gain insight into fundamental wave physics and the importance of phase relationships in forming coherent interference phenomena.

---

## 2. Theoretical Background

### 2.1 Single Disturbance Equation

A **circular wave** on the water surface, emanating from a point source at coordinates \((x_i, y_i)\), can be mathematically described by:

\[
y_i(x, y, t) \;=\; A \,\sin\!\bigl(k\,r_i - \omega\,t + \phi\bigr),
\]

where:
- \(A\) is the amplitude of the wave.
- \(k = \tfrac{2\pi}{\lambda}\) is the wave number, related to the wavelength \(\lambda\).
- \(\omega = 2\pi f\) is the angular frequency, related to the frequency \(f\).
- \(r_i = \sqrt{(x - x_i)^2 + (y - y_i)^2}\) is the radial distance from source \(i\) to the point \((x, y)\).
- \(\phi\) is an initial phase (often taken as 0 for coherent sources).

### 2.2 Principle of Superposition

When multiple wave sources exist, the total displacement at \((x, y)\) is the **algebraic sum** of the individual wave displacements:

\[
y_{\text{tot}}(x, y, t) \;=\; \sum_{i=1}^{N} A \,\sin\!\bigl(k\,r_i - \omega\,t + \phi\bigr),
\]

where \(N\) is the number of sources. If the sources are all coherent (same \(A\), \(\lambda\), and \(\omega\), with a fixed \(\phi\)-difference, which we can assume to be 0), the interference pattern will remain stable over time.

### 2.3 Constructive and Destructive Interference

- **Constructive Interference:** Occurs at points where the path difference between waves from different sources corresponds to an integer multiple of the wavelength, leading to waves arriving **in phase** and producing maximal amplitude.

- **Destructive Interference:** Occurs at points where the path difference is a half-integer multiple of the wavelength, leading to waves arriving **out of phase** and canceling each other, reducing the amplitude (potentially to near zero).

In two dimensions, these constructive and destructive loci form interlaced **hyperbolic curves** (for two sources). With more sources arranged symmetrically, these patterns become more intricate but retain geometric symmetry.

### 2.4 Regular Polygon Configuration

We choose \(N\) point sources located at the vertices of a regular \(N\)-sided polygon (equilateral triangle for \(N=3\), square for \(N=4\), pentagon for \(N=5\), etc.). For a polygon of “radius” \(R\) (the distance from the center to each vertex), the coordinates of the \(i\)-th vertex can be written as:

\[
x_i = R \,\cos\!\bigl(\tfrac{2\pi i}{N}\bigr), 
\quad
y_i = R \,\sin\!\bigl(\tfrac{2\pi i}{N}\bigr),
\quad
\text{for } i = 0, 1, 2, \dots, N-1.
\]

This setup ensures an evenly spaced arrangement of sources, making the interference pattern highly symmetric.

---

## 3. Implementation (Python Script)

Below is a sample Python code that simulates and visualizes the interference pattern from sources located at the vertices of a chosen regular polygon (here, **a square** with four equally spaced sources** is used as an example**). You can change the value of `N` and `R` to experiment with different polygons and spacing.

```python
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Define Wave and Simulation Parameters
# -----------------------------
A = 1.0                # Amplitude
lam = 1.0              # Wavelength
k = 2.0 * np.pi / lam  # Wave number
f = 1.0                # Frequency
omega = 2.0 * np.pi * f
phi = 0.0              # Initial phase

# Choose the number of vertices (N) for the polygon
N = 4  # e.g., 3 for triangle, 4 for square, 5 for pentagon, etc.

# Radius (distance from center to each vertex)
R = 2.0  

# Create a 2D grid for (x, y)
grid_size = 400  # resolution for the plot
x_min, x_max = -5, 5
y_min, y_max = -5, 5
x_vals = np.linspace(x_min, x_max, grid_size)
y_vals = np.linspace(y_min, y_max, grid_size)
X, Y = np.meshgrid(x_vals, y_vals)

# -----------------------------
# 2) Position the Wave Sources
# -----------------------------
# For a regular polygon, each source is at angle 2π i/N from x-axis
sources = []
for i in range(N):
    angle = 2.0 * np.pi * i / N
    xi = R * np.cos(angle)
    yi = R * np.sin(angle)
    sources.append((xi, yi))

# -----------------------------
# 3) Calculate the Total Wave Field
# -----------------------------
# We will compute the displacement at time t=0 for simplicity
t = 0.0
wave_field = np.zeros_like(X)  # same shape as the grid

for (x_i, y_i) in sources:
    # Distance from source to each point on the grid
    R_i = np.sqrt((X - x_i)**2 + (Y - y_i)**2)
    
    # Contribute to the total wave
    # We add a small epsilon to R_i to avoid division by zero at the source location
    wave_field += A * np.sin(k * R_i - omega * t + phi)

# -----------------------------
# 4) Visualization
# -----------------------------
plt.figure(figsize=(8, 6))
levels = 100  # number of contour levels
contour = plt.contourf(X, Y, wave_field, levels=levels, cmap='RdBu')

# Plot the sources as black dots
for (xi, yi) in sources:
    plt.plot(xi, yi, 'ko')  # black circle to mark source

plt.colorbar(contour, label="Wave Displacement")
plt.title(f"Interference Pattern from {N} Coherent Sources at Polygon Vertices")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.show()
```

### Code Explanation

1. **Wave Parameters:**  
   - `A` is the amplitude; `lam` is the wavelength; from these, we get `k = 2π / λ`.  
   - `f` is the frequency, so `omega = 2πf`. The phase is set to zero for simplicity, assuming all sources are in phase.

2. **Regular Polygon Setup:**  
   - `N` determines the polygon type (3 for triangle, 4 for square, 5 for pentagon, etc.).  
   - `R` is the distance from the polygon’s center to each vertex. The `i`-th vertex is placed at \((R \cos(\theta_i), R \sin(\theta_i))\), where \(\theta_i = \tfrac{2\pi i}{N}\).

3. **Grid Definition:**  
   - A 2D grid of points \((X, Y)\) covers the region from `x_min` to `x_max` and `y_min` to `y_max`.  
   - `grid_size` controls the resolution of the visualization.

4. **Wave Field Calculation:**  
   - For each vertex (source), we compute `R_i`, the distance from the source to each grid point. Then we compute the wave contribution \(\sin(k\,R_i - \omega\,t + \phi)\).  
   - These contributions are summed in `wave_field` to reflect superposition.

5. **Visualization:**  
   - We use `plt.contourf` to produce a filled contour plot. The red/blue colormap (`RdBu`) highlights positive (crests) versus negative (troughs) displacements.  
   - Mark each source with a black circle to visualize their locations.  
   - A color bar on the right shows the displacement scale.

---

## 4. Results & Analysis

When you run the code, you will see a **symmetrical interference pattern** whose geometry depends on:
- **Number of sources \(N\):**  A larger \(N\) yields more intricate patterns.  
- **Polygon radius \(R\):** Closer sources create denser interference patterns near the center; more widely spaced sources produce broader patterns.

### Key Observations

1. **Central Interference Zone:**  
   Near the center of the polygon, multiple waves arrive from nearly equal distances, typically creating a region of strong constructive interference. The exact pattern will depend on \(N\).

2. **Circular or Hyperbolic Bands:**  
   When \(N=2\), interference fringes are classic hyperbolae (constant path difference). For higher \(N\), the lines or fringes curve and intersect in a more complex arrangement.

3. **Constructive vs. Destructive Ridges:**  
   Alternating fringes of high (constructive) and low (destructive) displacement appear. The wave amplitude is maximal when path differences among sources meet integer multiples of \(\lambda\), and minimal for half-integer multiples of \(\lambda\).

4. **Effect of Time Variation:**  
   Although the code sets \(t=0\), the pattern remains essentially the same if all sources are truly coherent. Over time, the whole pattern may shift phase, but the **spatial** distribution of peaks and nodes does not fundamentally change.

---

## 5. Conclusion

This simulation and visualization of **interference patterns** formed by multiple coherent point sources on a water surface highlights several key insights:

- **Wave Superposition:**  The total displacement at any point is the sum of the individual waves, illustrating how constructive and destructive interference arises from path differences.
- **Geometric Symmetry:**  Arranging sources in a regular polygon yields highly symmetric interference patterns. Varying the number of vertices \(N\) or the polygon’s size \(R\) changes the observed pattern in predictable ways.
- **Practical Significance:**  These results mirror real-world phenomena, such as the interference of light in optics, the diffraction of sound waves in acoustics, and more.

By experimenting with the number of sources, their spacing, and even the time parameter, one can explore a variety of wave phenomena and gain deeper understanding of interference—a fundamental principle in wave physics.

**Deliverables**:
1. **Markdown Document & Code**: The code above (or Jupyter Notebook equivalent) suffices to replicate and extend the simulation.  
2. **Explanation of Patterns**: The step-by-step discussion helps you connect the mathematical model to the observed interference fringes.  
3. **Graphs & Visualizations**: Contour plots of the resulting wave field vividly demonstrate constructive and destructive interference across the water surface.

Feel free to adapt these scripts further, for instance, by:
- Animating the pattern over time (\(t\neq 0\)).
- Changing the phase \(\phi\) for one or more sources to investigate phase-shifted interference.
- Exploring different polygons (triangle, pentagon, hexagon) or random arrangements of sources to see less symmetric patterns.

With this, you have a complete framework for investigating **wave interference** from multiple coherent sources arranged at the vertices of a regular polygon.