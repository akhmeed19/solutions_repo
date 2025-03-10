# Problem 1

# Investigating Projectile Motion: Range as a Function of Projection Angle

## Theoretical Foundation

Projectile motion is governed by Newton's laws of motion. When a projectile is launched at an initial velocity \( v_0 \) at an angle \( \theta \) from the horizontal, the motion can be decomposed into horizontal and vertical components.

### Governing Equations

The equations of motion, ignoring air resistance and considering gravitational acceleration \( g \) acting downward, are:

- Horizontal motion:
  \[ x(t) = v_0 \cos(\theta) t \]

- Vertical motion:
  \[ y(t) = v_0 \sin(\theta) t - \frac{1}{2} g t^2 \]

The range \( R \) (horizontal distance traveled) occurs when \( y(t) = 0 \) (assuming launch and landing heights are the same):
\[
0 = v_0 \sin(\theta) t - \frac{1}{2} g t^2 \implies t = 0 \quad \text{or} \quad t = \frac{2 v_0 \sin(\theta)}{g}
\]

Thus, the range is given by:
\[ R = v_0 \cos(\theta) \cdot \frac{2 v_0 \sin(\theta)}{g} = \frac{v_0^2 \sin(2\theta)}{g} \]

### Family of Solutions

The range depends on:

- Initial velocity \( v_0 \)
- Angle \( \theta \)
- Gravitational acceleration \( g \)

Changing these parameters results in a variety of projectile trajectories and ranges.

## Analysis of Range

### Range vs. Angle

- Maximum range occurs at \( \theta = 45^\circ \).
- For angles symmetrically placed around \( 45^\circ \), the range is identical due to the \( \sin(2\theta) \) relationship.

### Influence of Parameters

- Increasing initial velocity \( v_0 \) proportionally increases the range.
- Increasing gravitational acceleration \( g \) decreases the range.

## Practical Applications

- **Sports:** Analysis of ball trajectories (e.g., soccer, basketball, golf).
- **Engineering:** Projectile launches (rockets, missiles).
- **Astrophysics:** Orbital insertion angles.

## Computational Implementation

Below is a Python script that simulates projectile motion and plots range versus angle.

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
g = 9.81  # gravitational acceleration (m/s^2)
v0 = 50   # initial velocity (m/s)
angles_deg = np.linspace(0, 90, 180)
angles_rad = np.radians(angles_deg)

# Calculate range
ranges = (v0**2 * np.sin(2 * angles_rad)) / g

# Plot
plt.figure(figsize=(10, 6))
plt.plot(angles_deg, ranges, label=f'v0={v0} m/s')

# Highlight maximum range
max_range_idx = np.argmax(ranges)
plt.plot(angles_deg[max_range_idx], ranges[max_range_idx], 'ro', label='Max Range')

plt.title('Projectile Range vs. Launch Angle')
plt.xlabel('Launch Angle (degrees)')
plt.ylabel('Range (meters)')
plt.legend()
plt.grid(True)
plt.show()
```

## Limitations and Enhancements

- **Idealized Assumptions:** No air resistance, drag, or wind considered.
- **Terrain:** Uneven surfaces alter landing height, modifying the range.
- **Realistic Extensions:**
  - **Air Resistance:** Incorporate drag force proportional to velocity squared.
  - **Wind:** Add horizontal acceleration to the motion equations.
  - **Uneven Terrain:** Adjust final vertical position conditions.

By incorporating these real-world factors, the model becomes increasingly robust and versatile, accurately describing diverse physical scenarios.
