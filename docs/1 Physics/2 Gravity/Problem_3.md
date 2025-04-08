# Problem 3

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

The gravitational force acting on a payload at a distance $r$ from Earth's center is given by

$$
\vec{F} = -\frac{\mu}{r^3}\vec{r},
$$

where $\mu = GM$ is Earth’s gravitational parameter ($G$ is the gravitational constant and $M$ is the mass of Earth).

Using Newton's second law, the acceleration of the payload is:

$$
\ddot{\vec{r}} = -\frac{\mu}{r^3}\vec{r}.
$$

While these equations can be expressed in Cartesian coordinates, gravity as a central force (always pointing toward the center) is more insightfully described in polar coordinates $(r, \theta)$.

### Polar Coordinates and Conservation Laws

In polar coordinates, the position vector is represented as:

$$
\vec{r} = r \hat{r}.
$$

The acceleration in polar coordinates has two components:

- **Radial acceleration:**
  
  $$
  \ddot{r} - r\dot{\theta}^2 = -\frac{\mu}{r^2},
  $$

- **Transverse (angular) acceleration:**

  $$
  r\ddot{\theta} + 2\dot{r}\dot{\theta} = 0.
  $$

The second equation implies the conservation of angular momentum $L$:

$$
L = r^2 \dot{\theta} = \text{constant}.
$$

### Deriving the Orbit Equation

Using the conservation of angular momentum, we derive an equation for the shape of the orbit. Introducing the substitution:

$$
u = \frac{1}{r},
$$

and differentiating with respect to $\theta$:

$$
\frac{dr}{d\theta} = -\frac{1}{u^2} \frac{du}{d\theta},
$$

the radial acceleration term can be expressed in terms of $u$ and $\theta$. After some manipulation, the radial equation transforms into the differential equation:

$$
\frac{d^2 u}{d\theta^2} + u = \frac{\mu}{L^2}.
$$

This linear differential equation has the general solution:

$$
u(\theta) = \frac{\mu}{L^2}\left(1 + e\cos(\theta)\right),
$$

or equivalently, the orbital equation in polar form:

$$
r(\theta) = \frac{L^2/\mu}{1 + e\cos(\theta)}.
$$

Here, $e$ is the eccentricity of the orbit:
- $0 \leq e < 1$: Elliptical (or circular if $e=0$)
- $e = 1$: Parabolic (the threshold between bound and unbound)
- $e > 1$: Hyperbolic (unbound)

### Energy Considerations: The Vis-Viva Equation

The specific orbital energy (energy per unit mass) is given by

$$
\epsilon = \frac{v^2}{2} - \frac{\mu}{r},
$$

where $v$ is the speed of the payload. The sign of $\epsilon$ determines the type of orbit:

- **Elliptical orbits:** $\epsilon < 0$
- **Parabolic trajectory:** $\epsilon = 0$
- **Hyperbolic trajectory:** $\epsilon > 0$

For a given orbit, the vis-viva equation relates the velocity $v$ at any distance $r$ to the semi-major axis $a$:

$$
v^2 = \mu\left(\frac{2}{r} - \frac{1}{a}\right).
$$

For a circular orbit ($r = a$), this simplifies to:

$$
v_{\text{circ}} = \sqrt{\frac{\mu}{r}},
$$

and the escape velocity (the speed needed for a parabolic trajectory) is:

$$
v_{\text{esc}} = \sqrt{\frac{2\mu}{r}}.
$$

### Effective Potential and Radial Motion

Another way to analyze orbital motion is through the effective potential. The total energy in the radial direction, considering the conservation of angular momentum, can be expressed as:

$$
\epsilon = \frac{1}{2}\dot{r}^2 + V_{\text{eff}}(r),
$$

where the effective potential $V_{\text{eff}}(r)$ is defined as:

$$
V_{\text{eff}}(r) = -\frac{\mu}{r} + \frac{L^2}{2r^2}.
$$

- The first term represents gravitational potential energy.
- The second term represents the "centrifugal" potential energy due to the payload's angular momentum.

The effective potential provides insight into radial stability. The minimum of $V_{\text{eff}}(r)$ corresponds to a stable circular orbit, while deviations lead to oscillatory changes in $r$, characteristic of elliptical orbits.

### Connecting Initial Conditions to Orbit Types

The initial position and velocity determine both the specific energy $\epsilon$ and the angular momentum $L$:

- **Circular Orbit:** The velocity is exactly $v_{\text{circ}}$, balancing gravitational pull and centripetal force for a constant $r$.
- **Elliptical Orbit:** A velocity lower than $v_{\text{circ}}$ results in an elliptical path with varying $r$.
- **Hyperbolic Trajectory:** A velocity exceeding $v_{\text{esc}}$ results in a positive energy orbit, allowing the payload to escape Earth’s gravitational influence.

### Summary

The trajectory of a payload released near Earth is governed by:
- **Gravitational forces:** As described by Newton’s law.
- **Conservation of angular momentum:** Leading to the derivation of the orbit equation.
- **Energy considerations:** Classifying orbits as elliptical, parabolic, or hyperbolic based on the specific orbital energy.
- **Effective potential:** Providing insight into the stability of orbits.

This theoretical framework forms the basis for designing simulations and computational tools to predict payload trajectories for space missions.

Below is the updated Markdown document that includes the code, the image of the **output**, and a detailed explanation of that output. Note that we’ve replaced references to "simulation" with "output," as requested.

---

## Numerical Simulation

Below are two separate simulation scenarios for comparison. The left side shows **Scenario 1 (200 km Altitude)** and the right side shows **Scenario 2 (400 km Altitude)**. In Scenario 1, the payload is released from 200 km altitude with:

- **Circular Orbit:**  
  \( v = \sqrt{\mu/(R_{\text{earth}}+200\,\text{km})} \)
- **Elliptical / Reentry:**  
  \( v = 0.8 \times v_{\text{circ}} \) (which leads to reentry)
- **Hyperbolic Trajectory:**  
  \( v = 1.1 \times v_{\text{esc}} \)

In Scenario 2, the payload is released from 400 km altitude with:

- **Circular Orbit:**  
  \( v = \sqrt{\mu/(R_{\text{earth}}+400\,\text{km})} \)
- **Elliptical Orbit:**  
  \( v = 0.9 \times v_{\text{circ}} \) (producing a stable, bound elliptical orbit)
- **Hyperbolic Trajectory:**  
  \( v = 1.1 \times v_{\text{esc}} \)

---

### Scenario 1 (200 km Altitude)

#### Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
mu = 3.986004418e14   # Earth's gravitational parameter (m^3/s^2)
R_earth = 6.371e6     # Earth's radius (m)

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

def collision_event(t, state):
    """
    Event function to detect collision with Earth.
    Integration stops when r - R_earth = 0.
    """
    x, y, _, _ = state
    r = np.sqrt(x**2 + y**2)
    return r - R_earth

collision_event.terminal = True
collision_event.direction = -1

def simulate_trajectory(r0, v0, t_span, t_eval):
    """
    Simulates the payload trajectory given initial conditions.
    """
    state0 = [r0[0], r0[1], v0[0], v0[1]]
    return solve_ivp(dynamics, t_span, state0, t_eval=t_eval,
                     events=collision_event, rtol=1e-8)

# Initial Conditions for 200 km altitude
altitude = 200e3                # 200 km altitude
r_mag = R_earth + altitude      # Distance from Earth's center
r0 = [r_mag, 0]                 # Starting along the x-axis

# Trajectory Cases
v_circ = np.sqrt(mu / r_mag)           # Circular Orbit speed
v0_circ = [0, v_circ]
v0_ellipse = [0, 0.8 * v_circ]         # Elliptical / Reentry (0.8×v_circ)
v_esc = np.sqrt(2 * mu / r_mag)          # Escape Velocity
v0_hyper = [0, 1.1 * v_esc]            # Hyperbolic Trajectory (1.1×v_esc)

# Time parameters
t_span = (0, 6000)  # seconds
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Simulate trajectories for 200 km altitude
sol_circ = simulate_trajectory(r0, v0_circ, t_span, t_eval)
sol_ellipse = simulate_trajectory(r0, v0_ellipse, t_span, t_eval)
sol_hyper = simulate_trajectory(r0, v0_hyper, t_span, t_eval)

# Plot the trajectories (normalized by Earth's radius)
plt.figure(figsize=(8, 8))
theta = np.linspace(0, 2*np.pi, 500)
earth_x = np.cos(theta)
earth_y = np.sin(theta)
plt.fill(earth_x, earth_y, 'b', alpha=0.3, label="Earth")
plt.plot(sol_circ.y[0] / R_earth, sol_circ.y[1] / R_earth, 'r', label="Circular Orbit")
plt.plot(sol_ellipse.y[0] / R_earth, sol_ellipse.y[1] / R_earth, 'g', label="Elliptical / Reentry")
plt.plot(sol_hyper.y[0] / R_earth, sol_hyper.y[1] / R_earth, 'm', label="Hyperbolic Trajectory")
plt.xlabel("x (in Earth radii)")
plt.ylabel("y (in Earth radii)")
plt.title("Payload Trajectories Near Earth (200 km Altitude)")
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()
```

#### Explanation of the Code (Scenario 1)

1. **Dynamics Function:**  
   The dynamics function computes the derivatives of the state vector \([x, y, vx, vy]\) using Newton’s law of gravitation, \(\ddot{\vec{r}} = -\mu \vec{r}/r^3\).

2. **Simulation Function:**  
   The simulate_trajectory function sets up the initial state and integrates the equations over the defined time span using the solve_ivp solver. The collision event stops the integration when the payload intersects Earth’s surface.

3. **Initial Conditions:**                                                           
    - The payload is assumed to be released from a position 200 km above Earth's surface.
     - Three initial velocity cases are defined:
        - **Circular orbit:** using the circular velocity \(v_{\text{circ}} = \sqrt{\mu / r}\).
        - **Elliptical trajectory:** using 80% of the circular velocity.
        - **Hyperbolic trajectory:** using 110% of the escape velocity \(v_{\text{esc}} = \sqrt{2\mu / r}\).

4. **Plotting:**  
   The code normalizes positions by Earth's radius to display Earth as a unit circle. Each trajectory is plotted in the \(xy\)-plane.

#### Graphical Representations (Output)

![Payload Trajectories Near Earth](https://raw.githubusercontent.com/akhmeed19/solutions_repo/refs/heads/main/docs/_pics/Gravity/Problem3/Payload200.png)

#### Output Explanation (Scenario 1)

- **Blue Circle (Earth):**  
  Represents Earth as a unit circle. The payload is released at approximately 1.03 Earth radii.
  
- **Red Curve (Circular Orbit):**  
  The payload maintains a tight circular orbit at 200 km altitude.
  
- **Green Curve (Elliptical / Reentry):**  
  With \(0.8 \times v_{\text{circ}}\), the payload’s trajectory intersects Earth (reentry).
  
- **Magenta Curve (Hyperbolic Trajectory):**  
  The payload follows an escape trajectory.

#### Key Takeaways (Scenario 1)

1. Altering the initial velocity significantly changes the orbital path.
2. At 200 km, using \(0.8 \times v_{\text{circ}}\) leads to reentry.
3. The hyperbolic trajectory clearly demonstrates escape conditions.

---

### Scenario 2 (400 km Altitude)

#### Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
mu = 3.986004418e14   # Earth's gravitational parameter (m^3/s^2)
R_earth = 6.371e6     # Earth's radius (m)

def dynamics(t, state):
    """Computes the derivatives for the state vector [x, y, vx, vy] using Newton's law."""
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    ax = -mu * x / r**3
    ay = -mu * y / r**3
    return [vx, vy, ax, ay]

def collision_event(t, state):
    """Stops integration when the payload reaches Earth's surface."""
    x, y, _, _ = state
    return np.sqrt(x**2 + y**2) - R_earth

collision_event.terminal = True
collision_event.direction = -1

def simulate_trajectory(r0, v0, t_span, t_eval):
    """Simulates the payload trajectory given initial conditions."""
    state0 = [r0[0], r0[1], v0[0], v0[1]]
    return solve_ivp(dynamics, t_span, state0, t_eval=t_eval,
                     events=collision_event, rtol=1e-8)

# Initial Conditions for 400 km altitude
altitude = 400e3                # 400 km altitude
r_mag = R_earth + altitude      # Distance from Earth's center
r0 = [r_mag, 0]                 # Starting along the x-axis

# Trajectory Cases
v_circ = np.sqrt(mu / r_mag)         # Circular Orbit speed
v0_circ = [0, v_circ]
v0_ellipse = [0, 0.9 * v_circ]       # Elliptical Orbit (bound) using 0.9×v_circ
v_esc = np.sqrt(2 * mu / r_mag)        # Escape Velocity
v0_hyper = [0, 1.1 * v_esc]          # Hyperbolic Trajectory (1.1×v_esc)

# Time parameters
t_span = (0, 6000)  # seconds
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Simulate trajectories for 400 km altitude
sol_circ_new = simulate_trajectory(r0, v0_circ, t_span, t_eval)
sol_ellipse_new = simulate_trajectory(r0, v0_ellipse, t_span, t_eval)
sol_hyper_new = simulate_trajectory(r0, v0_hyper, t_span, t_eval)

# Plot the trajectories (normalized by Earth's radius)
plt.figure(figsize=(8, 8))
theta = np.linspace(0, 2*np.pi, 500)
earth_x = np.cos(theta)
earth_y = np.sin(theta)
plt.fill(earth_x, earth_y, 'b', alpha=0.3, label="Earth")
plt.plot(sol_circ_new.y[0] / R_earth, sol_circ_new.y[1] / R_earth, 'r', label="Circular Orbit (400 km)")
plt.plot(sol_ellipse_new.y[0] / R_earth, sol_ellipse_new.y[1] / R_earth, 'g', label="Elliptical Orbit (400 km)")
plt.plot(sol_hyper_new.y[0] / R_earth, sol_hyper_new.y[1] / R_earth, 'm', label="Hyperbolic Trajectory (400 km)")
plt.xlabel("x (in Earth radii)")
plt.ylabel("y (in Earth radii)")
plt.title("Payload Trajectories at 400 km Altitude")
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()
```

#### Explanation of the Code (Scenario 2)

1. **Dynamics Function:**  
   Computes the derivatives of the state vector $[x, y, vx, vy]$ using Newton’s law.
2. **Simulation Function:**  
   Sets up the initial state and integrates the equations over time using solve_ivp, with a collision event stopping the integration when the payload hits Earth.
3. **Initial Conditions (400 km):**                                                   
    The payload is released from 400 km altitude (approximately 1.06 Earth radii).  
    - **Circular Orbit:** $v = \sqrt{\mu/(R_{\text{earth}}+400\,\text{km})}$
    - **Elliptical Orbit:** $v = 0.9 \times v_{\text{circ}}$ (producing a bound elliptical orbit)
    - **Hyperbolic Trajectory:** $v = 1.1 \times v_{\text{esc}}$
4. **Plotting:**  
   Normalizes the trajectories by Earth's radius to display Earth as a unit circle.

#### Graphical Representations (Output)

![Payload Trajectories Near Earth](https://raw.githubusercontent.com/akhmeed19/solutions_repo/refs/heads/main/docs/_pics/Gravity/Problem3/Payload400.png)

#### Output Explanation (400 km Altitude)

- **Blue Circle (Earth):**  
  Represents Earth as a unit circle; the payload starts at approximately 1.06 Earth radii.
- **Red Curve (Circular Orbit):**  
  The payload maintains a stable circular orbit at 400 km altitude.
- **Green Curve (Elliptical Orbit):**  
  With $0.9 \times v_{\text{circ}}$, the payload follows a bound elliptical orbit that does not reenter.
- **Magenta Curve (Hyperbolic Trajectory):**  
  The payload follows an escape trajectory, with slight differences due to the higher altitude.

#### Key Takeaways (Scenario 2)

1. Higher altitude produces larger, more stable orbits.
2. Using $0.9 \times v_{\text{circ}}$ yields a bound elliptical orbit without reentry.
3. Both scenarios exhibit hyperbolic escape trajectories when $v$ exceeds $v_{\text{esc}}$.

---

### Side-by-Side Comparison of Outputs

Below is a side-by-side comparison of the outputs from Scenario 1 and Scenario 2.


<table>
  <tr>
    <td align="center">
      <strong>200 km Altitude Output</strong><br>
      <img src="https://raw.githubusercontent.com/akhmeed19/solutions_repo/refs/heads/main/docs/_pics/Gravity/Problem3/Payload200.png" alt="200 km Output" style="width:300px;">
    </td>
    <td align="center">
      <strong>400 km Altitude Output</strong><br>
      <img src="https://raw.githubusercontent.com/akhmeed19/solutions_repo/refs/heads/main/docs/_pics/Gravity/Problem3/Payload400.png" alt="400 km Output" style="width:300px;">
    </td>
  </tr>
</table>

### Comparison Explanation

- **Altitude Difference:**                                                           
    - *Scenario 1 (200 km):* The payload starts at about 1.03 Earth radii, resulting in a tight circular orbit, a reentry elliptical trajectory (with $0.8 \times v_{\text{circ}}$), and an escape hyperbolic path.
    - *Scenario 2 (400 km):* The payload starts at about 1.06 Earth radii, producing a larger, more stable circular orbit, a bound elliptical orbit (with $0.9 \times v_{\text{circ}}$) that does not reenter, and a hyperbolic trajectory with a slightly modified curvature.

- **Orbital Stability and Realism:**  
  A 400 km altitude is more typical for operational satellites due to lower atmospheric drag. The elliptical orbit remains bound at 400 km, whereas at 200 km the elliptical trajectory leads to reentry.

- **Velocity Multipliers:**  
  A multiplier of 0.8 at 200 km results in reentry, while 0.9 at 400 km yields a stable elliptical orbit, demonstrating the sensitivity of orbital dynamics to initial velocity.

- **Hyperbolic Trajectories:**  
  Both scenarios use $1.1 \times v_{\text{esc}}$ to produce escape trajectories; however, the lower escape velocity at 400 km slightly modifies the hyperbolic path's curvature.

This comparison illustrates how realistic changes in initial altitude and velocity multipliers can lead to significantly different orbital behaviors, providing valuable insights for mission planning.

---

## Exploring the Oberth Effect at Periapsis

In this simulation, we compare two cases of a payload initially in an elliptical orbit having a periapsis at 300 km and an apogee at 600 km altitude. We consider:

- **Case 1 (No Burn):** The payload follows the baseline elliptical orbit.
- **Case 2 (With Burn):** At periapsis—where the payload is moving fastest—a small impulsive burn is applied (adding, for example, 100 m/s to the tangential velocity). This extra energy raises the apogee significantly, thereby demonstrating the Oberth effect.

---

## Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
mu = 3.986004418e14   # Earth's gravitational parameter (m^3/s^2)
R_earth = 6.371e6     # Earth's radius (m)

# Define orbital parameters for the elliptical orbit:
# Periapsis at 300 km and Apogee at 600 km altitude
r_p = R_earth + 300e3  # Periapsis radius
r_a = R_earth + 600e3  # Apogee radius

# Compute the velocity at periapsis for the desired elliptical orbit.
# The velocity at periapsis for an ellipse is given by:
#    v_p = sqrt(2*mu*r_a / (r_p * (r_a + r_p)))
v_p = np.sqrt(2 * mu * r_a / (r_p * (r_a + r_p)))

# Define a small delta-V to be applied at periapsis (demonstrating the Oberth effect).
delta_V = 100  # m/s

# Define two cases:
# Case 1: No burn (baseline elliptical orbit)
# Case 2: Burn applied at periapsis (adds delta_V to the tangential speed)
initial_state_no_burn = [r_p, 0, 0, v_p]         # Position at (r_p, 0), velocity (0, v_p)
initial_state_burn    = [r_p, 0, 0, v_p + delta_V]  # Position at (r_p, 0), velocity increased

# Define the dynamics using Newton's law of gravitation.
def dynamics(t, state):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    ax = -mu * x / r**3
    ay = -mu * y / r**3
    return [vx, vy, ax, ay]

# Event function to stop integration if the payload impacts Earth.
def collision_event(t, state):
    x, y, _, _ = state
    return np.sqrt(x**2 + y**2) - R_earth
collision_event.terminal = True
collision_event.direction = -1

# Time parameters
t_span = (0, 6000)    # seconds - roughly a simulation window covering one or more orbits
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Simulate the trajectories for both cases.
sol_no_burn = solve_ivp(dynamics, t_span, initial_state_no_burn, t_eval=t_eval,
                        events=collision_event, rtol=1e-8)
sol_burn = solve_ivp(dynamics, t_span, initial_state_burn, t_eval=t_eval,
                     events=collision_event, rtol=1e-8)

# Plotting the results
plt.figure(figsize=(8, 8))
# Draw Earth as a filled circle (normalized by Earth's radius).
theta = np.linspace(0, 2*np.pi, 500)
earth_x = np.cos(theta)
earth_y = np.sin(theta)
plt.fill(earth_x, earth_y, 'b', alpha=0.3, label="Earth")

# Plot both trajectories (positions normalized by Earth's radius).
plt.plot(sol_no_burn.y[0] / R_earth, sol_no_burn.y[1] / R_earth, 'g', label="No Burn (Baseline)")
plt.plot(sol_burn.y[0] / R_earth, sol_burn.y[1] / R_earth, 'r', label="With Burn (ΔV = 100 m/s)")

plt.xlabel("x (in Earth radii)")
plt.ylabel("y (in Earth radii)")
plt.title("Effect of an Impulsive Burn at Periapsis (Oberth Effect)")
plt.legend(loc="upper right")
plt.axis('equal')
plt.grid()
plt.show()
```

---

## Explanation of the Code

- **Orbital Parameters and Initial Conditions:**                                        
    - **Altitudes:**  
    The periapsis is set at 300 km and the apogee at 600 km above Earth’s surface. These values are converted into radii by adding them to Earth's radius.
    - **Periapsis Velocity Calculation:**  
    The velocity at periapsis is computed with:
    $$
    v_p = \sqrt{\frac{2\mu \, r_a}{r_p (r_a + r_p)}}
    $$
    This calculation ensures that the baseline elliptical orbit will have the specified periapsis and apogee.
    - **Impulsive Burn:**  
    A delta‑$V$ of 100 m/s is applied at periapsis for the second case. Two initial state vectors are defined:
     - **No Burn (Baseline):** $[r_p, 0, 0, v_p]$
     - **With Burn:** $[r_p, 0, 0, v_p + \Delta V]$

- **Dynamics and Numerical Integration:**                                                                      
    - **Dynamics Function:**  
    The `dynamics` function calculates the acceleration using Newton’s law of gravitation:
    $$
    \ddot{x} = -\frac{\mu x}{r^3}, \quad \ddot{y} = -\frac{\mu y}{r^3}
    $$
    - **Event Function:**  
    The `collision_event` function halts the integration if the payload reaches Earth's surface (i.e., when  $r = R_{\text{earth}}$).
    - **Integration:**  
    Both trajectories are simulated over a 6000-second window using `solve_ivp` with a tight relative tolerance of $1 \times 10^{-8}$ to ensure accuracy.

- **Plotting:**                                                                       
    - **Earth Representation:**  
    Earth is plotted as a blue filled circle after normalizing positions by Earth's radius, so it appears as a unit circle.
    - **Trajectory Plotting:**  
    The baseline elliptical orbit (no burn) is plotted in green, and the orbit after the impulsive burn is plotted in red.
    - **Additional Plot Elements:**  
    Labels for the axes, a title, and a legend are included to clarify the two different scenarios.

---

## Output Image

Below is a representative sample output image from the simulation:

![Oberth Effect Simulation Output](https://raw.githubusercontent.com/akhmeed19/solutions_repo/refs/heads/main/docs/_pics/Gravity/Problem3/OberthEffect.png)

---

## Explanation of the Output

1. **Visualization Context:**                   
    - **Normalized Display:**  
     The plot displays Earth as a blue filled unit circle since all distances are normalized by Earth's radius.
    - **Starting Point:**  
     The simulation begins at a periapsis of 300 km (roughly $1.05\,R_{\text{earth}}$), which places the orbits near Earth’s surface on the normalized scale.

2. **Baseline Orbit (No Burn):**                
    - **Green Curve:**  
     The green curve represents the elliptical orbit with a periapsis at 300 km and an apogee at 600 km. This orbit follows the intended path without any modifications.

3. **Orbit with Impulsive Burn (Oberth Effect):**                                   
    - **Red Curve:**  
     The red curve shows the orbit after a 100 m/s impulsive burn is applied at periapsis.
    - **Effect of the Burn:**  
     The burn, performed at the point of highest orbital speed, efficiently increases the orbital energy due to the Oberth effect. This results in a significantly higher apogee compared to the baseline elliptical orbit.
    - **Visual Distinction:**  
     Although the altitude differences are small relative to Earth's overall size, the divergence between the red and green curves clearly illustrates the impact of the impulsive burn.

4. **General Observations:**                    
    - The simulation demonstrates that impulsive burns executed at periapsis are highly effective, as the added delta‑$V$ results in a markedly altered orbit with a raised apogee.
    - Modifying the value of $\Delta V$ or changing the orbital parameters would result in more pronounced visual differences.

---

## Conclusion

In summary, this comprehensive solution demonstrates how a payload's trajectory near Earth is intricately governed by its initial conditions and gravitational forces. The theoretical framework—grounded in Newton's law, energy conservation, and the orbital equations—provides a solid foundation for classifying trajectories into elliptical, parabolic, or hyperbolic types. Our numerical simulations further illustrate how subtle changes in initial velocity, whether by using different multipliers or through impulsive burns, can significantly alter the orbit.

The exploration of the Oberth effect, in particular, underscores the power of executing burns at periapsis, where the payload’s high speed makes additional velocity changes exceptionally efficient. This not only leads to a substantial increase in orbital energy and a higher apogee but also reinforces the importance of burn timing in mission planning.

Together, these findings bridge theoretical insights with practical simulation techniques, offering valuable perspectives for designing efficient orbital maneuvers and space missions.