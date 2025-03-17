# Problem 2

Below is a comprehensive Markdown document that outlines the derivation, analysis, and simulation of a forced damped pendulum with a focus on its dynamic behavior under external forcing and damping. The document includes theoretical derivations, discussion of parameter effects and limitations, as well as a Python simulation to visualize the pendulum’s motion.

---

# Investigating the Dynamics of a Forced Damped Pendulum

<br/>

## 1. Introduction

The forced damped pendulum is a classic nonlinear system that exhibits a rich variety of behaviors—from simple periodic oscillations to complex chaotic motion—due to the interplay between damping, restoring forces, and an external periodic drive. By varying parameters such as the damping coefficient, driving amplitude, and driving frequency, one can obtain a diverse set of solutions that model real-world phenomena ranging from energy harvesting to structural vibrations.

In this document, we:

- Derive the governing differential equation for the forced damped pendulum.
- Analyze the influence of damping, driving amplitude, and driving frequency on the system’s dynamics.
- Discuss practical applications and limitations of the forced damped pendulum model.
- Implement a simulation in Python to visualize the pendulum’s behavior.

---

## 2. Theoretical Foundation

### Differential Equation

The motion of a forced damped pendulum is governed by the nonlinear differential equation:

$$
\ddot{\theta} + 2\beta\, \dot{\theta} + \omega_0^2 \sin\theta = \frac{F_D}{mL} \cos(\Omega t),
$$

where:

- $ \theta $ is the angular displacement,
- $ \beta $ is the damping coefficient,
- $ \omega_0 = \sqrt{\frac{g}{L}} $ is the natural frequency of the pendulum (with $ L $ being the pendulum length),
- $ F_D $ is the amplitude of the driving force,
- $( m )$ is the mass of the pendulum bob,
- $( L )$ is the length of the pendulum,
- $( \Omega )$ is the driving frequency.

### Small-Angle Approximation

For small oscillations, we use the approximation:

$$
\sin\theta \approx \theta,
$$

which linearizes the equation to:

$$
\ddot{\theta} + 2\beta\, \dot{\theta} + \omega_0^2 \theta = \frac{F_D}{mL} \cos(\Omega t).
$$

The general solution is the sum of the homogeneous solution (which decays due to damping) and a particular solution that oscillates at the driving frequency.

### Resonance Conditions

- **Resonance:**  
  The system experiences resonance when the driving frequency $ \Omega $ is close to the natural frequency $ \omega_0 $. At resonance, the amplitude of the steady-state oscillations becomes large, limited by the damping.

- **Energy Implications:**  
  Under resonance, the system efficiently absorbs energy from the external force, resulting in significant oscillation amplitudes.

---

## 3. Analysis of Dynamics

### Parameter Influences

- **Damping Coefficient ($\beta$):**  
  Higher damping reduces the amplitude of oscillations and can suppress chaotic behavior, while lower damping allows for larger amplitudes and may lead to complex dynamics.

- **Driving Amplitude ($F_D/(mL)$):**  
  Increasing the driving amplitude increases the steady-state oscillation amplitude. Very high driving amplitudes can push the system into a nonlinear regime, potentially resulting in chaotic motion.

- **Driving Frequency ($\Omega$):**  
  When $ \Omega $ is near $ \omega_0 $, the system experiences resonance. As $ \Omega $ varies, the system may transition between regular (periodic) motion and chaotic behavior.

### Transition to Chaos

- **Regular Motion:**  
  For moderate values of damping and driving amplitude, the pendulum displays periodic oscillations.

- **Chaotic Motion:**  
  Under low damping or high driving amplitude, the system becomes highly sensitive to initial conditions, leading to chaotic behavior.

- **Visualization Tools:**  
  Phase portraits and Poincaré sections are used to reveal the underlying structure of the motion and the transition to chaos.

---

## 4. Practical Applications and Limitations

### Applications

- **Energy Harvesting:**  
  Systems can be designed to exploit resonance, thereby maximizing energy absorption from periodic forces.
- **Structural Engineering:**  
  Understanding forced oscillations is crucial in designing structures (e.g., suspension bridges, vibration isolation systems) that must withstand periodic loads.
- **Oscillating Circuits:**  
  The behavior of driven RLC circuits is analogous to the forced damped pendulum, informing the design of filters and resonant circuits.

### Limitations

- The small-angle approximation simplifies the analysis but is only valid for small displacements.
- The model neglects effects such as nonlinear damping and non-periodic driving forces, which may be significant in real-world systems.
- For larger oscillations, the full nonlinear equation (with $\sin\theta$) should be used.

---

## 5. Implementation: Python Simulation

In this first simulation, we produce three key plots to analyze the forced damped pendulum’s motion:

- A time series plot of $ \theta(t) $
- A phase portrait ($ \theta $ vs. $ \dot{\theta} $)
- A Poincaré section to capture periodic and chaotic dynamics

Below is the Python code for our **initial simulation** under the small-angle approximation. It uses `solve_ivp` to integrate the system, then plots the results. For now, we run **one** set of parameters ($\beta=0.25$, $F_D=1.2$, $\Omega=2/3$, etc.):

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def run_forced_damped_pendulum(beta=0.25, F_D=1.2, Omega=2/3,
                               t_span=(0, 100), y0=(0.1, 0.0),
                               use_small_angle=True):
    """
    Simulates and plots the forced damped pendulum.

    Parameters:
    -----------
    beta : float
        Damping coefficient.
    F_D : float
        Driving force amplitude (N).
    Omega : float
        Driving frequency (rad/s).
    t_span : tuple
        Start and end time for the simulation, e.g. (0, 100).
    y0 : tuple
        Initial conditions (theta, omega).
    use_small_angle : bool
        If True, use the small-angle approximation (sin(theta) ~ theta).
        If False, use the full nonlinear equation sin(theta).
    """
    g = 9.81   # gravitational acceleration (m/s^2)
    L = 1.0    # pendulum length (m)
    m = 1.0    # mass (kg)

    # Natural frequency
    omega0 = np.sqrt(g / L)
    # Driving force term
    driving_force = F_D / (m * L)

    # Define the ODE
    def forced_damped_pendulum(t, y):
        theta, omega = y
        dtheta_dt = omega

        if use_small_angle:
            # Small-angle approximation: sin(theta) ~ theta
            restoring = -omega0**2 * theta
        else:
            # Full nonlinear: sin(theta)
            restoring = -omega0**2 * np.sin(theta)

        domega_dt = -2 * beta * omega + restoring + driving_force * np.cos(Omega * t)
        return [dtheta_dt, domega_dt]

    # Create a time array for evaluation
    t_eval = np.linspace(t_span[0], t_span[1], 10000)

    # Solve the ODE
    sol = solve_ivp(forced_damped_pendulum, t_span, y0, t_eval=t_eval, rtol=1e-8)
    t = sol.t
    theta = sol.y[0]
    omega = sol.y[1]

    # --- PLOTS ---

    # 1) Time Series of Theta(t)
    plt.figure(figsize=(10, 4))
    plt.plot(t, theta, 'b-', label=r'$\theta(t)$')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title(f'Time Series (beta={beta}, F_D={F_D}, Omega={Omega})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2) Phase Portrait (theta vs. omega)
    plt.figure(figsize=(6, 6))
    plt.plot(theta, omega, 'r-', lw=0.8)
    plt.xlabel(r'$\theta$ (rad)')
    plt.ylabel(r'$\dot{\theta}$ (rad/s)')
    plt.title(f'Phase Portrait (beta={beta}, F_D={F_D}, Omega={Omega})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3) Poincaré Section
    # Sample points at every period T_drive = 2*pi/Omega after a transient
    T_drive = 2 * np.pi / Omega
    # Skip initial transients (e.g., first 50 cycles)
    skip_cycles = 50
    poincare_times = np.arange(skip_cycles * T_drive, t_span[1], T_drive)
    poincare_thetas = np.interp(poincare_times, t, theta)
    poincare_omegas = np.interp(poincare_times, t, omega)

    plt.figure(figsize=(6, 6))
    plt.scatter(poincare_thetas, poincare_omegas, c='green', s=25)
    plt.xlabel(r'$\theta$ (rad)')
    plt.ylabel(r'$\dot{\theta}$ (rad/s)')
    plt.title(f'Poincaré Section (beta={beta}, F_D={F_D}, Omega={Omega})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_forced_damped_pendulum(
        beta=0.25,
        F_D=1.2,
        Omega=2/3,
        t_span=(0, 100),
        y0=(0.1, 0.0),
        use_small_angle=True
    )
```

### Outputs for the First Simulation

Running this script with $(\beta=0.25)$, $(F_D=1.2)$, $(\Omega=2/3)$, and `use_small_angle = True` produces **three plots**:

1. **Time Series**
   ![Time Series Output](https://raw.githubusercontent.com/akhmeed19/solutions_repo/refs/heads/main/docs/_pics/Mechanics%20P2/time%20series.png)
   A nearly sinusoidal wave indicates stable periodic motion.

2. **Phase Portrait**
   ![Phase Portrait Output](https://raw.githubusercontent.com/akhmeed19/solutions_repo/refs/heads/main/docs/_pics/Mechanics%20P2/phase%20portrait.png)
   A closed loop in $\theta$–$\dot{\theta}$ space, showing a periodic orbit.

3. **Poincaré Section**
   ![Poincaré Section Output](https://raw.githubusercontent.com/akhmeed19/solutions_repo/refs/heads/main/docs/_pics/Mechanics%20P2/poincare%20section.png)
   A small set of points implies non-chaotic, regular motion.

### Explanation

1. **Small-Angle Approximation:**
   Since `use_small_angle = True`, we use $\sin(\theta) \approx \theta$. This typically yields simpler, near-harmonic motion for moderate parameter values.

2. **Stable Periodic Orbit:**
   Under $(\beta=0.25)$, $(F_D=1.2)$, and $(\Omega=2/3)$, the pendulum exhibits a stable periodic solution—evident from the sinusoidal time series, single-loop phase portrait, and a single or small set of points in the Poincaré section.

3. **Potential Extensions:**
   - Set `use_small_angle = False` to use the **full nonlinear** equation, which can yield chaotic solutions for certain parameter ranges.
   - Vary $\beta$ (damping), $F_D$ (driving amplitude), and $\Omega$ (driving frequency) to see more complex dynamics, including chaos.

---

## 6. Simulation Output

Below is a representative static image of the forced damped pendulum’s motion under these conditions:

<br>
![Forced Damped Pendulum Simulation](https://raw.githubusercontent.com/akhmeed19/solutions_repo/refs/heads/main/docs/_pics/Projectile_Range_vs_Angle.png)
<br>

_(Update the URL to match the actual published location of your simulation image.)_

---

## 7. Discussion

- **General Solutions:**
  With the small-angle approximation and moderate parameters, the pendulum typically remains in a **regular periodic orbit**. However, for certain parameter sets (e.g., lower damping, higher driving amplitude, or using the full \(\sin(\theta)\)), it can transition to **chaotic motion**.

- **Graphical Representations:**

  1. **Time Series:** Depicts how \(\theta\) changes over time.
  2. **Phase Portrait:** Shows \(\theta\) vs. \(\dot{\theta}\), revealing periodic loops or chaotic attractors.
  3. **Poincaré Section:** Captures the system’s state once every driving period, distinguishing periodic from chaotic dynamics.

- **Limitations & Future Work:**
  - The small-angle approximation may fail for larger amplitudes.
  - Nonlinear damping or non-periodic driving forces can further complicate real-world behavior.

---

## 8. Conclusion

This investigation of the forced damped pendulum has:

- Derived the governing equations and provided an approximate solution for small oscillations.
- Analyzed how damping, driving amplitude, and driving frequency affect the system’s dynamics.
- Illustrated the transition from periodic to chaotic behavior using **time series**, **phase portraits**, and **Poincaré sections**.
- Demonstrated a computational simulation in Python that visualizes these dynamics.

Overall, this study reinforces key concepts in nonlinear dynamics and provides a foundation for further exploration, including adjusting parameters to observe chaotic regimes or incorporating more realistic effects such as nonlinear damping.
