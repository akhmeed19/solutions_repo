# Problem 3

Below is the complete Markdown document with the updated theoretical background section integrated into the previous solution.

---

# Trajectories of a Freely Released Payload Near Earth

## Introduction

When a payload is released from a moving rocket near Earth, its subsequent trajectory is determined by its initial position, velocity, and the gravitational pull of the Earth. Depending on the initial conditions, the payload can follow:

- **Elliptical (or circular) orbits:** if its kinetic energy is such that the total energy is negative.
- **Parabolic trajectories:** the borderline case between bound and unbound orbits (zero total energy).
- **Hyperbolic trajectories:** if the payload has more than enough kinetic energy to escape Earth's gravitational field (positive total energy).

This document explains the underlying physics, describes a numerical method to simulate the payload motion, and shows how to visualize the different trajectories.

## Extended Theoretical Background

Understanding the trajectories of a freely released payload near Earth begins with Newton's Law of Gravitation and the conservation laws of energy and angular momentum. Here, we expand on these principles, derive key equations, and explain how they govern orbital motion.

### Newton’s Law of Gravitation and Equations of Motion

The gravitational force acting on a payload at a distance \( r \) from Earth's center is given by

\[
\vec{F} = -\frac{\mu}{r^3}\vec{r},
\]

where \( \mu = GM \) is Earth’s gravitational parameter (\( G \) is the gravitational constant and \( M \) is the mass of Earth).

Using Newton's second law, the acceleration of the payload is:

\[
\ddot{\vec{r}} = -\frac{\mu}{r^3}\vec{r}.
\]

While these equations can be expressed in Cartesian coordinates, gravity as a central force (always pointing toward the center) is more insightfully described in polar coordinates \((r, \theta)\).

### Polar Coordinates and Conservation Laws

In polar coordinates, the position vector is represented as:

\[
\vec{r} = r \hat{r}.
\]

The acceleration in polar coordinates has two components:

- **Radial acceleration:**
  
  \[
  \ddot{r} - r\dot{\theta}^2 = -\frac{\mu}{r^2},
  \]

- **Transverse (angular) acceleration:**

  \[
  r\ddot{\theta} + 2\dot{r}\dot{\theta} = 0.
  \]

The second equation implies the conservation of angular momentum \( L \):

\[
L = r^2 \dot{\theta} = \text{constant}.
\]

### Deriving the Orbit Equation

Using the conservation of angular momentum, we derive an equation for the shape of the orbit. Introducing the substitution:

\[
u = \frac{1}{r},
\]

and differentiating with respect to \(\theta\):

\[
\frac{dr}{d\theta} = -\frac{1}{u^2} \frac{du}{d\theta},
\]

the radial acceleration term can be expressed in terms of \( u \) and \(\theta \). After some manipulation, the radial equation transforms into the differential equation:

\[
\frac{d^2 u}{d\theta^2} + u = \frac{\mu}{L^2}.
\]

This linear differential equation has the general solution:

\[
u(\theta) = \frac{\mu}{L^2}\left(1 + e\cos(\theta)\right),
\]

or equivalently, the orbital equation in polar form:

\[
r(\theta) = \frac{L^2/\mu}{1 + e\cos(\theta)}.
\]

Here, \( e \) is the eccentricity of the orbit:
- \( 0 \leq e < 1 \): Elliptical (or circular if \( e=0 \))
- \( e = 1 \): Parabolic (the threshold between bound and unbound)
- \( e > 1 \): Hyperbolic (unbound)

### Energy Considerations: The Vis-Viva Equation

The specific orbital energy (energy per unit mass) is given by

\[
\epsilon = \frac{v^2}{2} - \frac{\mu}{r},
\]

where \( v \) is the speed of the payload. The sign of \(\epsilon\) determines the type of orbit:

- **Elliptical orbits:** \(\epsilon < 0\)
- **Parabolic trajectory:** \(\epsilon = 0\)
- **Hyperbolic trajectory:** \(\epsilon > 0\)

For a given orbit, the vis-viva equation relates the velocity \( v \) at any distance \( r \) to the semi-major axis \( a \):

\[
v^2 = \mu\left(\frac{2}{r} - \frac{1}{a}\right).
\]

For a circular orbit (\( r = a \)), this simplifies to:

\[
v_{\text{circ}} = \sqrt{\frac{\mu}{r}},
\]

and the escape velocity (the speed needed for a parabolic trajectory) is:

\[
v_{\text{esc}} = \sqrt{\frac{2\mu}{r}}.
\]

### Effective Potential and Radial Motion

Another way to analyze orbital motion is through the effective potential. The total energy in the radial direction, considering the conservation of angular momentum, can be expressed as:

\[
\epsilon = \frac{1}{2}\dot{r}^2 + V_{\text{eff}}(r),
\]

where the effective potential \( V_{\text{eff}}(r) \) is defined as:

\[
V_{\text{eff}}(r) = -\frac{\mu}{r} + \frac{L^2}{2r^2}.
\]

- The first term represents gravitational potential energy.
- The second term represents the "centrifugal" potential energy due to the payload's angular momentum.

The effective potential provides insight into radial stability. The minimum of \( V_{\text{eff}}(r) \) corresponds to a stable circular orbit, while deviations lead to oscillatory changes in \( r \), characteristic of elliptical orbits.

### Connecting Initial Conditions to Orbit Types

The initial position and velocity determine both the specific energy \( \epsilon \) and the angular momentum \( L \):

- **Circular Orbit:** The velocity is exactly \( v_{\text{circ}} \), balancing gravitational pull and centripetal force for a constant \( r \).
- **Elliptical Orbit:** A velocity lower than \( v_{\text{circ}} \) results in an elliptical path with varying \( r \).
- **Hyperbolic Trajectory:** A velocity exceeding \( v_{\text{esc}} \) results in a positive energy orbit, allowing the payload to escape Earth’s gravitational influence.

### Summary

The trajectory of a payload released near Earth is governed by:
- **Gravitational forces:** As described by Newton’s law.
- **Conservation of angular momentum:** Leading to the derivation of the orbit equation.
- **Energy considerations:** Classifying orbits as elliptical, parabolic, or hyperbolic based on the specific orbital energy.
- **Effective potential:** Providing insight into the stability of orbits.

This theoretical framework forms the basis for designing simulations and computational tools to predict payload trajectories for space missions.

## Numerical Simulation

We use Python along with the `scipy.integrate.solve_ivp` solver to numerically integrate the equations of motion. The payload is assumed to be released at a given altitude above Earth's surface. Three cases are simulated:

1. **Circular Orbit:** Initial speed equal to \(v_{\text{circ}} = \sqrt{\mu/r}\).
2. **Elliptical / Reentry:** A lower initial speed (e.g., \(0.8 \times v_{\text{circ}}\)) causing an elliptical path that may intersect Earth.
3. **Hyperbolic Trajectory:** A speed greater than the escape velocity (\(v_{\text{esc}} = \sqrt{2\mu/r}\)), for example, \(1.1 \times v_{\text{esc}}\).

## Python Script

Below is the complete Python script:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
mu = 3.986004418e14   # Earth's gravitational parameter, m^3/s^2
R_earth = 6.371e6     # Earth's radius, m

def dynamics(t, state):
    """
    Computes the derivatives for the state vector.
    
    state = [x, y, vx, vy]
    """
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    ax = -mu * x / r**3
    ay = -mu * y / r**3
    return [vx, vy, ax, ay]

def simulate_trajectory(r0, v0, t_span, t_eval):
    """
    Simulates the payload trajectory given initial conditions.
    
    Parameters:
    - r0: Initial position vector [x, y]
    - v0: Initial velocity vector [vx, vy]
    - t_span: Tuple for the simulation time span (start, end)
    - t_eval: Array of time points at which to store the solution
    """
    state0 = [r0[0], r0[1], v0[0], v0[1]]
    sol = solve_ivp(dynamics, t_span, state0, t_eval=t_eval, rtol=1e-8)
    return sol

# Initial Conditions
altitude = 200e3                # Altitude above Earth's surface: 200 km
r_mag = R_earth + altitude      # Distance from Earth's center
r0 = [r_mag, 0]                 # Starting along the x-axis

# Case 1: Circular Orbit (v = sqrt(mu/r))
v_circ = np.sqrt(mu / r_mag)
v0_circ = [0, v_circ]

# Case 2: Elliptical (suborbital) trajectory (v = 0.8 * v_circ)
v0_ellipse = [0, 0.8 * v_circ]

# Case 3: Hyperbolic trajectory (v = 1.1 * v_esc, where v_esc = sqrt(2*mu/r))
v_esc = np.sqrt(2 * mu / r_mag)
v0_hyper = [0, 1.1 * v_esc]

# Simulation time parameters
t_span = (0, 6000)  # seconds
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Simulate each trajectory
sol_circ = simulate_trajectory(r0, v0_circ, t_span, t_eval)
sol_ellipse = simulate_trajectory(r0, v0_ellipse, t_span, t_eval)
sol_hyper = simulate_trajectory(r0, v0_hyper, t_span, t_eval)

# Plot the trajectories
plt.figure(figsize=(8, 8))

# Draw Earth
theta = np.linspace(0, 2 * np.pi, 500)
earth_x = R_earth * np.cos(theta)
earth_y = R_earth * np.sin(theta)
plt.fill(earth_x, earth_y, 'b', alpha=0.3, label="Earth")

# Plot each trajectory
plt.plot(sol_circ.y[0], sol_circ.y[1], 'r', label="Circular Orbit")
plt.plot(sol_ellipse.y[0], sol_ellipse.y[1], 'g', label="Elliptical / Reentry")
plt.plot(sol_hyper.y[0], sol_hyper.y[1], 'm', label="Hyperbolic Trajectory")

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Payload Trajectories Near Earth")
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()
```

## Explanation of the Code

1. **Dynamics Function:**  
   The `dynamics` function computes the derivatives of the state vector (position and velocity) using Newton’s law of gravitation.

2. **Simulation Function:**  
   The `simulate_trajectory` function sets up the initial state and integrates the equations over the defined time span using the `solve_ivp` solver.

3. **Initial Conditions:**  
   - The payload is assumed to be released from a position 200 km above Earth's surface.
   - Three initial velocity cases are defined:
     - **Circular orbit:** using the circular velocity.
     - **Elliptical trajectory:** using 80% of the circular velocity.
     - **Hyperbolic trajectory:** using 110% of the escape velocity.

4. **Plotting:**  
   The trajectories are plotted in the \(xy\)-plane, with Earth represented as a filled circle. Each trajectory is labeled for clarity.

## Graphical Representations

The script produces a plot showing:

- **Circular Orbit:** A closed, circular path around Earth.
- **Elliptical / Reentry Trajectory:** An ellipse that may intersect Earth’s surface, indicating a reentry scenario.
- **Hyperbolic Trajectory:** An open curve representing an escape trajectory.

---

This simulation demonstrates how different initial conditions can lead to dramatically different paths—critical for applications like satellite deployment, payload recovery, or interplanetary missions.