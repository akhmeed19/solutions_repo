# Problem 1

# Investigating the Range as a Function of the Angle of Projection

## 1. Theoretical Foundation

### Derivation of Governing Equations

Projectile motion can be modeled using basic principles of classical mechanics. Under gravity and ignoring air resistance, the equations of motion in the horizontal (x) and vertical (y) directions are given by:

\[
x(t) = v_0 \cos(\theta) t
\]
\[
y(t) = v_0 \sin(\theta) t - \frac{1}{2} g t^2
\]

Where:

- \( v_0 \) is the initial velocity
- \( \theta \) is the angle of projection
- \( g \) is gravitational acceleration

### Family of Solutions

The solution to these equations gives a family of trajectories dependent on initial conditions such as initial velocity, launch angle, and gravitational acceleration. Varying these conditions alters the trajectory shape significantly.

## 2. Analysis of the Range

The horizontal range \( R \) is defined as the horizontal distance traveled when the projectile returns to the initial vertical height (assuming level ground). It is given by:

\[
R = \frac{v_0^2 \sin(2\theta)}{g}
\]

Key Observations:

- Maximum range occurs at \( \theta = 45^\circ \).
- Increasing initial velocity increases the range quadratically.
- Increasing gravitational acceleration decreases the range proportionally.

## 3. Practical Applications

This idealized projectile model serves as the basis for various practical applications, including:

- Sports: determining optimal angles for throwing or kicking.
- Engineering: planning trajectories for missiles or rockets.
- Astrophysics: predicting motion under gravitational fields.

Real-world adaptations include factors like uneven terrain, air resistance, drag, and wind conditions, which can be included using numerical methods and simulations.

## 4. Implementation

Here's a Python implementation to simulate projectile motion and visualize how the range varies with projection angle:

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
g = 9.81  # gravitational acceleration (m/s^2)
v0 = 20   # initial velocity (m/s)
angles = np.linspace(0, np.pi/2, 180)

# Calculate range for each angle
def calculate_range(v0, angle, g):
    return (v0**2) * np.sin(2*angle) / g

ranges = calculate_range(v0, angles, g)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(np.degrees(angles), ranges, label=f'Initial velocity = {v0} m/s')
plt.xlabel('Angle of Projection (degrees)')
plt.ylabel('Range (m)')
plt.title('Projectile Range as a Function of Launch Angle')
plt.grid(True)
plt.legend()
plt.show()
```

### Visual Analysis

The visualization clearly shows the parabolic relationship between angle and range, confirming the maximum range at 45°.

## Limitations and Realistic Enhancements

### Limitations of the Idealized Model:

- Neglects air resistance (drag), wind, and uneven terrain.
- Assumes constant gravitational acceleration.

### Suggestions for Enhancements:

- Incorporate drag using velocity-dependent forces.
- Include elevation differences or uneven ground.
- Consider variable gravitational acceleration for high-altitude or space applications.

Such enhancements will make the model more realistic and applicable to practical engineering scenarios.
