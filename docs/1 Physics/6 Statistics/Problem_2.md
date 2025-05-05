# Estimating π using Monte Carlo Methods

## Introduction

Monte Carlo methods represent a class of computational algorithms that utilize random sampling to obtain numerical results. One of the most elegant and intuitive applications of these methods is estimating the value of π. This document explores two different Monte Carlo approaches to approximating π: the circle-based method and Buffon's Needle experiment.

## Part 1: Estimating π Using a Circle

### 1.1 Theoretical Foundation

The circle-based Monte Carlo method for estimating π relies on the relationship between the area of a circle and the area of its bounding square. Consider a unit circle (radius = 1) centered at the origin, enclosed by a 2×2 square:

- Area of the unit circle: $A_{circle} = \pi r^2 = \pi \cdot 1^2 = \pi$
- Area of the bounding square: $A_{square} = (2r)^2 = 4$

The ratio of these areas is:

$$\frac{A_{circle}}{A_{square}} = \frac{\pi}{4}$$

If we randomly generate points within the square, the probability of a point falling inside the circle equals the ratio of the areas:

$$P(\text{point inside circle}) = \frac{A_{circle}}{A_{square}} = \frac{\pi}{4}$$

By rearranging, we get:

$$\pi \approx 4 \times \frac{\text{points inside circle}}{\text{total points}}$$

This provides us with a method to estimate π: generate random points within the square, count how many fall inside the circle, and apply the formula.

### 1.2 Simulation Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Circle

def estimate_pi_circle(num_points):
    """
    Estimate π using the circle-based Monte Carlo method.
    
    Parameters:
    -----------
    num_points : int
        Number of random points to generate
        
    Returns:
    --------
    float
        Estimated value of π
    ndarray
        Array of points inside the circle
    ndarray
        Array of points outside the circle
    """
    # Generate random points in the square [-1, 1] × [-1, 1]
    x = np.random.uniform(-1, 1, num_points)
    y = np.random.uniform(-1, 1, num_points)
    
    # Compute distances from origin
    distances = x**2 + y**2
    
    # Determine which points are inside the circle (distance < 1)
    inside_circle = distances <= 1
    
    # Count points inside the circle
    count_inside = np.sum(inside_circle)
    
    # Estimate π
    pi_estimate = 4 * count_inside / num_points
    
    # Return the estimate and points for visualization
    return pi_estimate, np.column_stack((x[inside_circle], y[inside_circle])), np.column_stack((x[~inside_circle], y[~inside_circle]))

def visualize_circle_method(points_inside, points_outside, pi_estimate, num_points):
    """
    Create a visualization of the circle-based Monte Carlo method.
    
    Parameters:
    -----------
    points_inside : ndarray
        Points inside the circle
    points_outside : ndarray
        Points outside the circle
    pi_estimate : float
        Estimated value of π
    num_points : int
        Total number of points used in the simulation
    """
    plt.figure(figsize=(10, 10))
    
    # Plot the unit circle
    circle = Circle((0, 0), 1, fill=False, color='r', linewidth=2)
    plt.gca().add_patch(circle)
    
    # Plot the square
    plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'b-', linewidth=2)
    
    # Plot points
    if len(points_inside) > 0:
        plt.scatter(points_inside[:, 0], points_inside[:, 1], color='green', alpha=0.5, s=5, label='Inside Circle')
    if len(points_outside) > 0:
        plt.scatter(points_outside[:, 0], points_outside[:, 1], color='red', alpha=0.5, s=5, label='Outside Circle')
    
    plt.axis('equal')
    plt.grid(True)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.title(f'Estimating π using Monte Carlo Method (Circle)\n'
              f'Points: {num_points}, π ≈ {pi_estimate:.6f}, Error: {abs(pi_estimate - np.pi):.6f}')
    plt.legend()
    plt.savefig('circle_monte_carlo.png')
    plt.show()

def analyze_convergence_circle(max_points=1000000, steps=20):
    """
    Analyze how the estimation of π converges as the number of points increases.
    
    Parameters:
    -----------
    max_points : int
        Maximum number of points to use
    steps : int
        Number of steps to take between 1000 and max_points
    
    Returns:
    --------
    ndarray
        Array of numbers of points used
    ndarray
        Array of π estimates
    ndarray
        Array of execution times
    """
    # Use logarithmic spacing for better visualization
    points_range = np.logspace(3, np.log10(max_points), steps).astype(int)
    pi_estimates = np.zeros(steps)
    exec_times = np.zeros(steps)
    
    for i, n in enumerate(points_range):
        start_time = time.time()
        pi_estimates[i], _, _ = estimate_pi_circle(n)
        exec_times[i] = time.time() - start_time
    
    return points_range, pi_estimates, exec_times

# Example usage
if __name__ == "__main__":
    # Estimate π using different numbers of points
    num_points_visualization = 5000  # For visualization
    pi_estimate, points_inside, points_outside = estimate_pi_circle(num_points_visualization)
    visualize_circle_method(points_inside, points_outside, pi_estimate, num_points_visualization)
    
    # Analyze convergence
    points_range, pi_estimates, exec_times = analyze_convergence_circle(max_points=1000000)
    
    # Plot convergence
    plt.figure(figsize=(12, 6))
    
    # Plot the estimates
    plt.subplot(1, 2, 1)
    plt.semilogx(points_range, pi_estimates, 'b-o')
    plt.axhline(y=np.pi, color='r', linestyle='--', label='True π')
    plt.xlabel('Number of Points')
    plt.ylabel('Estimated π')
    plt.title('Convergence of π Estimate (Circle Method)')
    plt.grid(True)
    plt.legend()
    
    # Plot the errors
    plt.subplot(1, 2, 2)
    plt.loglog(points_range, np.abs(pi_estimates - np.pi), 'g-o')
    plt.xlabel('Number of Points (log scale)')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('Error vs. Number of Points')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('circle_convergence.png')
    plt.show()
    
    # Print results
    print("Circle-based Monte Carlo Method Results:")
    print(f"{'Points':<12} {'π Estimate':<15} {'Error':<15} {'Time (s)':<10}")
    print("-" * 52)
    for n, pi_est, t in zip(points_range, pi_estimates, exec_times):
        print(f"{n:<12} {pi_est:<15.8f} {abs(pi_est - np.pi):<15.8f} {t:<10.6f}")
```

### 1.3 Visualization and Analysis

The circle-based Monte Carlo method provides a straightforward and intuitive way to estimate π. As the number of points increases, the estimate converges to the true value of π. The method's error decreases proportionally to $1/\sqrt{n}$, where $n$ is the number of points.

The visualization shows random points distributed within a square, with points colored based on whether they fall inside or outside the unit circle. As more points are generated, the ratio of points inside the circle to the total number of points converges to $\pi/4$.

## Part 2: Estimating π Using Buffon's Needle

### 2.1 Theoretical Foundation

Buffon's Needle is a classic probability problem formulated by Georges-Louis Leclerc, Comte de Buffon, in the 18th century. The problem involves dropping a needle randomly on a surface with parallel lines and calculating the probability of the needle crossing a line.

Consider a plane with parallel lines spaced at a distance $d$ apart. A needle of length $L$ (where $L \leq d$) is dropped randomly on this plane. The probability that the needle crosses a line is:

$$P(\text{needle crosses a line}) = \frac{2L}{\pi d}$$

Rearranging this formula to solve for π:

$$\pi \approx \frac{2L \times \text{number of throws}}{d \times \text{number of crossings}}$$

When $L = d$, the formula simplifies to:

$$\pi \approx \frac{2 \times \text{number of throws}}{\text{number of crossings}}$$

This provides another method to estimate π: randomly drop needles, count how many cross lines, and apply the formula.

### 2.2 Simulation Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import time

def estimate_pi_buffon(num_needles, needle_length=1.0, line_distance=1.0):
    """
    Estimate π using Buffon's Needle experiment.
    
    Parameters:
    -----------
    num_needles : int
        Number of needles to drop
    needle_length : float
        Length of the needle
    line_distance : float
        Distance between parallel lines
        
    Returns:
    --------
    float
        Estimated value of π
    ndarray
        Array of needle positions (y-coordinate of center)
    ndarray
        Array of needle angles
    ndarray
        Boolean array indicating whether each needle crosses a line
    """
    # Generate random needle positions (y-coordinate of needle center)
    y_positions = np.random.uniform(0, line_distance, num_needles)
    
    # Generate random needle angles (with horizontal)
    angles = np.random.uniform(0, np.pi, num_needles)
    
    # Calculate the distance from each needle's center to the nearest line
    distances_to_nearest_line = np.minimum(y_positions, line_distance - y_positions)
    
    # Calculate the y-projection of each needle's half-length
    y_projections = (needle_length / 2) * np.sin(angles)
    
    # Determine which needles cross a line
    crosses_line = y_projections >= distances_to_nearest_line
    
    # Count needles crossing lines
    count_crosses = np.sum(crosses_line)
    
    # Estimate π
    if count_crosses > 0:
        pi_estimate = (2 * needle_length * num_needles) / (line_distance * count_crosses)
    else:
        pi_estimate = float('inf')  # No crossings
    
    return pi_estimate, y_positions, angles, crosses_line

def visualize_buffon_needle(y_positions, angles, crosses_line, pi_estimate, num_needles):
    """
    Create a visualization of Buffon's Needle experiment.
    
    Parameters:
    -----------
    y_positions : ndarray
        Y-coordinates of needle centers
    angles : ndarray
        Angles of needles
    crosses_line : ndarray
        Boolean array indicating whether each needle crosses a line
    pi_estimate : float
        Estimated value of π
    num_needles : int
        Total number of needles used in the simulation
    """
    plt.figure(figsize=(12, 8))
    
    # Draw horizontal lines
    num_lines = 5
    for i in range(num_lines + 1):
        plt.axhline(y=i, color='black', linewidth=1)
    
    # Calculate x-positions (arbitrary, for visualization only)
    x_positions = np.random.uniform(0.5, num_lines - 0.5, len(y_positions))
    
    # Draw needles
    needle_length = 1.0  # Same as in the simulation
    for i in range(len(y_positions)):
        # Map y-position to the correct line spacing
        y_pos = y_positions[i] % 1 + int(x_positions[i])
        
        # Calculate needle endpoints
        dx = (needle_length / 2) * np.cos(angles[i])
        dy = (needle_length / 2) * np.sin(angles[i])
        
        x1, y1 = x_positions[i] - dx, y_pos - dy
        x2, y2 = x_positions[i] + dx, y_pos + dy
        
        color = 'red' if crosses_line[i] else 'blue'
        plt.plot([x1, x2], [y1, y2], color=color, linewidth=1.5)
    
    plt.xlim(0, num_lines)
    plt.ylim(0, num_lines)
    plt.gca().set_aspect('equal')
    plt.title(f"Buffon's Needle Experiment\n"
              f"Needles: {num_needles}, Crossings: {np.sum(crosses_line)}, π ≈ {pi_estimate:.6f}")
    
    # Add a legend
    plt.plot([], [], 'b-', label='No Crossing')
    plt.plot([], [], 'r-', label='Crossing Line')
    plt.legend()
    
    plt.savefig('buffon_needle.png')
    plt.show()

def analyze_convergence_buffon(max_needles=1000000, steps=20):
    """
    Analyze how the estimation of π converges as the number of needles increases.
    
    Parameters:
    -----------
    max_needles : int
        Maximum number of needles to use
    steps : int
        Number of steps to take between 1000 and max_needles
    
    Returns:
    --------
    ndarray
        Array of numbers of needles used
    ndarray
        Array of π estimates
    ndarray
        Array of execution times
    """
    # Use logarithmic spacing for better visualization
    needles_range = np.logspace(3, np.log10(max_needles), steps).astype(int)
    pi_estimates = np.zeros(steps)
    exec_times = np.zeros(steps)
    
    for i, n in enumerate(needles_range):
        start_time = time.time()
        pi_estimates[i], _, _, _ = estimate_pi_buffon(n)
        exec_times[i] = time.time() - start_time
    
    return needles_range, pi_estimates, exec_times

# Example usage
if __name__ == "__main__":
    # Estimate π using different numbers of needles
    num_needles_visualization = 100  # For visualization
    pi_estimate, y_positions, angles, crosses_line = estimate_pi_buffon(num_needles_visualization)
    visualize_buffon_needle(y_positions, angles, crosses_line, pi_estimate, num_needles_visualization)
    
    # Analyze convergence
    needles_range, pi_estimates, exec_times = analyze_convergence_buffon(max_points=1000000)
    
    # Plot convergence
    plt.figure(figsize=(12, 6))
    
    # Plot the estimates
    plt.subplot(1, 2, 1)
    plt.semilogx(needles_range, pi_estimates, 'b-o')
    plt.axhline(y=np.pi, color='r', linestyle='--', label='True π')
    plt.xlabel('Number of Needles')
    plt.ylabel('Estimated π')
    plt.title('Convergence of π Estimate (Buffon\'s Needle)')
    plt.grid(True)
    plt.legend()
    
    # Plot the errors
    plt.subplot(1, 2, 2)
    plt.loglog(needles_range, np.abs(pi_estimates - np.pi), 'g-o')
    plt.xlabel('Number of Needles (log scale)')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('Error vs. Number of Needles')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('buffon_convergence.png')
    plt.show()
    
    # Print results
    print("\nBuffon's Needle Method Results:")
    print(f"{'Needles':<12} {'π Estimate':<15} {'Error':<15} {'Time (s)':<10}")
    print("-" * 52)
    for n, pi_est, t in zip(needles_range, pi_estimates, exec_times):
        print(f"{n:<12} {pi_est:<15.8f} {abs(pi_est - np.pi):<15.8f} {t:<10.6f}")
```

### 2.3 Visualization and Analysis

Buffon's Needle experiment offers a fascinating geometric approach to estimating π. The visualization shows needles dropped randomly on a surface with parallel lines, with red needles indicating those that cross lines and blue needles indicating those that do not.

While conceptually elegant, Buffon's Needle typically converges more slowly than the circle-based method. This is because the probability of a needle crossing a line is relatively small, leading to higher variance in the estimate. The error also decreases proportionally to $1/\sqrt{n}$, but with a larger constant factor compared to the circle-based method.

## Part 3: Comparison of Methods

### 3.1 Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import time

def compare_methods(max_samples=1000000, steps=15):
    """
    Compare the circle-based and Buffon's Needle methods for estimating π.
    
    Parameters:
    -----------
    max_samples : int
        Maximum number of samples (points/needles) to use
    steps : int
        Number of steps to take between 1000 and max_samples
    """
    # Use logarithmic spacing for better visualization
    samples_range = np.logspace(3, np.log10(max_samples), steps).astype(int)
    
    # Arrays to store results
    circle_estimates = np.zeros(steps)
    circle_errors = np.zeros(steps)
    circle_times = np.zeros(steps)
    
    buffon_estimates = np.zeros(steps)
    buffon_errors = np.zeros(steps)
    buffon_times = np.zeros(steps)
    
    # Run simulations for both methods
    for i, n in enumerate(samples_range):
        # Circle-based method
        start_time = time.time()
        circle_estimates[i], _, _ = estimate_pi_circle(n)
        circle_times[i] = time.time() - start_time
        circle_errors[i] = abs(circle_estimates[i] - np.pi)
        
        # Buffon's Needle method
        start_time = time.time()
        buffon_estimates[i], _, _, _ = estimate_pi_buffon(n)
        buffon_times[i] = time.time() - start_time
        buffon_errors[i] = abs(buffon_estimates[i] - np.pi)
    
    # Plot comparison
    plt.figure(figsize=(18, 12))
    
    # Plot estimated π values
    plt.subplot(2, 2, 1)
    plt.semilogx(samples_range, circle_estimates, 'b-o', label='Circle Method')
    plt.semilogx(samples_range, buffon_estimates, 'g-o', label='Buffon\'s Needle')
    plt.axhline(y=np.pi, color='r', linestyle='--', label='True π')
    plt.xlabel('Number of Samples (log scale)')
    plt.ylabel('Estimated π')
    plt.title('Comparison of π Estimates')
    plt.grid(True)
    plt.legend()
    
    # Plot errors
    plt.subplot(2, 2, 2)
    plt.loglog(samples_range, circle_errors, 'b-o', label='Circle Method')
    plt.loglog(samples_range, buffon_errors, 'g-o', label='Buffon\'s Needle')
    
    # Add a reference line for 1/sqrt(n) convergence
    ref_line = circle_errors[0] * np.sqrt(samples_range[0] / samples_range)
    plt.loglog(samples_range, ref_line, 'k--', label='1/√n Reference')
    
    plt.xlabel('Number of Samples (log scale)')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('Error Comparison')
    plt.grid(True)
    plt.legend()
    
    # Plot execution times
    plt.subplot(2, 2, 3)
    plt.loglog(samples_range, circle_times, 'b-o', label='Circle Method')
    plt.loglog(samples_range, buffon_times, 'g-o', label='Buffon\'s Needle')
    plt.xlabel('Number of Samples (log scale)')
    plt.ylabel('Execution Time (s) (log scale)')
    plt.title('Computational Efficiency')
    plt.grid(True)
    plt.legend()
    
    # Plot error vs. time
    plt.subplot(2, 2, 4)
    plt.loglog(circle_times, circle_errors, 'b-o', label='Circle Method')
    plt.loglog(buffon_times, buffon_errors, 'g-o', label='Buffon\'s Needle')
    plt.xlabel('Execution Time (s) (log scale)')
    plt.ylabel('Absolute Error (log scale)')
    plt.title('Error vs. Computation Time')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('method_comparison.png')
    plt.show()
    
    # Print comparative table
    print("\nComparison of Methods:")
    print(f"{'Samples':<12} {'Circle π':<15} {'Circle Error':<15} {'Buffon π':<15} {'Buffon Error':<15}")
    print("-" * 72)
    for n, c_est, c_err, b_est, b_err in zip(samples_range, circle_estimates, circle_errors, buffon_estimates, buffon_errors):
        print(f"{n:<12} {c_est:<15.8f} {c_err:<15.8f} {b_est:<15.8f} {b_err:<15.8f}")

# Run the comparison
if __name__ == "__main__":
    print("\nComparing Circle and Buffon's Needle Methods:")
    compare_methods(max_samples=500000)
```

### 3.2 Comparative Analysis

Both Monte Carlo methods converge to π as the number of samples increases, but they differ in efficiency and convergence rate:

1. **Accuracy and Convergence**:
   - Both methods exhibit the expected $O(1/\sqrt{n})$ convergence rate, typical of Monte Carlo methods.
   - The circle-based method generally provides more accurate estimates with fewer samples due to its lower variance.
   - Buffon's Needle shows higher variability, especially at lower sample counts.

2. **Computational Efficiency**:
   - The circle-based method is computationally more efficient, requiring simpler calculations per sample.
   - Buffon's Needle involves more complex geometry calculations, leading to slightly longer execution times.

3. **Practical Considerations**:
   - The circle-based method is easier to implement and visualize, making it more suitable for educational purposes.
   - Buffon's Needle provides a fascinating historical connection and demonstrates how physical experiments can be used to estimate mathematical constants.

## Conclusion

Monte Carlo methods offer elegant and intuitive approaches to estimating π, demonstrating the power of probabilistic techniques in numerical computation. The circle-based method provides a more efficient and accurate estimate with fewer samples, while Buffon's Needle offers historical and educational value despite its slower convergence.

These methods highlight the versatility of Monte Carlo simulation across various domains, from mathematics and physics to finance and computer science. They also provide insights into the trade-offs between different simulation approaches and the relationship between sample size and estimation accuracy.

While modern computational methods can calculate π to much higher precision, Monte Carlo approaches remain valuable for educational purposes and as foundational examples of probabilistic problem-solving.