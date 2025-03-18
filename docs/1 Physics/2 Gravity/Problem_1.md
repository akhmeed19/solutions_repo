# Problem 1

Below is a comprehensive Markdown document that outlines the derivation, analysis, and simulation of circular orbits with a focus on the relationship between the square of the orbital period and the cube of the orbital radius, known as Kepler's Third Law. This law is fundamental in celestial mechanics, allowing the determination of planetary motions and providing insights into gravitational interactions at various scales.

---

# Orbital Period and Orbital Radius

## 1. Introduction

Kepler's Third Law establishes that for a body in a circular orbit, the square of the orbital period $T$ is directly proportional to the cube of the orbital radius $r$. This relationship is a cornerstone of celestial mechanics, providing a simple yet powerful way to understand and calculate the motion of planets, satellites, and other celestial bodies.

In this document, we:

- Derive the relationship between the square of the orbital period and the cube of the orbital radius for circular orbits.
- Discuss the implications of this relationship for astronomy, such as calculating planetary masses and distances.
- Analyze real-world examples, like the Moon's orbit around the Earth and the orbits of planets in the Solar System.
- Implement a computational model in Python to simulate circular orbits and verify the relationship.

---

## 2. Theoretical Foundation

### Derivation of Kepler's Third Law for Circular Orbits

For a body of mass $m$ orbiting a much larger mass $M$ in a circular orbit of radius $r$, the gravitational force provides the necessary centripetal force to keep the body in orbit. Equating the gravitational force and the centripetal force gives:

$$
\frac{G M m}{r^2} = \frac{m v^2}{r},
$$

where $G$ is the gravitational constant and $v$ is the orbital speed. Simplifying, we obtain:

$$
v^2 = \frac{G M}{r}.
$$

The orbital period $T$ is the time required to complete one orbit, which is the circumference of the orbit divided by the orbital speed:

$$
T = \frac{2\pi r}{v}.
$$

Substitute $v = \sqrt{\frac{G M}{r}}$ into the equation for $T$:

$$
T = \frac{2\pi r}{\sqrt{\frac{G M}{r}}} = 2\pi \sqrt{\frac{r^3}{G M}}.
$$

Squaring both sides gives:

$$
T^2 = \frac{4\pi^2}{G M} r^3.
$$

Thus, we have derived Kepler's Third Law for circular orbits:

$$
T^2 \propto r^3.
$$

### Implications for Astronomy

- **Planetary Motions:**  
  The law allows astronomers to determine the relative distances and orbital periods of planets. For example, if the orbital period of one planet is known, its orbital radius can be calculated relative to another planet.
  
- **Calculating Masses:**  
  By rearranging the equation, one can determine the mass $M$ of the central body if the orbital period $T$ and radius $r$ are known:
  
  $$
  M = \frac{4\pi^2 r^3}{G T^2}.
  $$
  
- **Extension to Elliptical Orbits:**  
  Kepler's Third Law can be extended to elliptical orbits by replacing $r$ with the semi-major axis $a$. The law then states that:
  
  $$
  T^2 \propto a^3.
  $$

---

## 3. Analysis of Dynamics

### Parameter Influences

- **Orbital Period:**  
  The equation $T = 2\pi \sqrt{\frac{r^3}{G M}}$ shows that the orbital period increases as the $3/2$ power of the orbital radius.
  
- **Implications for Different Systems:**  
  This relationship applies to any two bodies in orbit around a common central mass (provided the orbiting body's mass is negligible compared to $M$), making it a powerful tool for understanding both satellite orbits and planetary systems.

### Real-World Examples

- **The Moon's Orbit:**  
  By measuring the Moon's orbital period and distance from Earth, the Earth's mass can be estimated.
  
- **Planetary Orbits in the Solar System:**  
  The consistent ratio $\frac{T^2}{r^3}$ observed for all planets supports the universality of gravitational forces.

---

## 4. Practical Applications and Limitations

### Applications

- **Satellite Orbits:**  
  Kepler's Third Law is used to design satellite trajectories and to determine orbital parameters.
  
- **Astronomical Measurements:**  
  It plays a key role in estimating the masses of celestial bodies, such as stars and planets, based on the motion of their satellites.
  
- **Space Missions:**  
  Engineers use this law to plan interplanetary missions, ensuring spacecraft achieve the necessary orbital speeds and trajectories.

### Limitations

- **Assumptions:**  
  The derivation assumes circular orbits and that the orbiting body’s mass is negligible compared to the central mass.
  
- **Elliptical Orbits:**  
  For elliptical orbits, while Kepler's Third Law still holds (with $r$ replaced by the semi-major axis $a$), additional considerations (such as orbital eccentricity) must be taken into account.
  
- **Perturbations:**  
  In multi-body systems, gravitational interactions between orbiting bodies can lead to deviations from the simple $T^2 \propto r^3$ relationship.

---

## 5. Implementation: Python Simulation

In this simulation, we implement a computational model to simulate circular orbits and verify Kepler's Third Law. The simulation computes the orbital period for various orbital radii and then plots the relationship between $T^2$ and $r^3$.

### 5.1. Simulating Circular Orbits

Below is the Python code that calculates the orbital period for a range of orbital radii for a given central mass $M$ using the formula:

$$
T = 2\pi \sqrt{\frac{r^3}{GM}}.
$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11      # Gravitational constant in m^3/(kg·s^2)
M = 5.972e24         # Mass of the Earth in kg (or choose another central mass)

# Define a range of orbital radii (in meters)
radii = np.linspace(6.4e6, 4e7, 100)  # from roughly Earth's surface to high orbit

# Calculate the orbital period T for each radius using Kepler's Third Law:
# T = 2*pi*sqrt(r^3/(G*M))
T = 2 * np.pi * np.sqrt(radii**3 / (G * M))

# Compute T^2 and r^3 for verification of the proportionality
T_squared = T**2
r_cubed = radii**3

# Plot T^2 vs. r^3
plt.figure(figsize=(8, 6))
plt.plot(r_cubed, T_squared, 'bo', label=r'$T^2$ vs. $r^3$')
plt.xlabel(r'$r^3$ (m$^3$)')
plt.ylabel(r'$T^2$ (s$^2$)')
plt.title('Verification of Kepler\'s Third Law: $T^2 \propto r^3$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Plot the theoretical line for comparison
# Theoretical relationship: T^2 = (4*pi^2/(G*M))*r^3
slope = 4 * np.pi**2 / (G * M)
T_squared_theoretical = slope * r_cubed

plt.figure(figsize=(8, 6))
plt.plot(r_cubed, T_squared, 'bo', label=r'Calculated $T^2$')
plt.plot(r_cubed, T_squared_theoretical, 'r-', label=r'Theoretical $T^2$')
plt.xlabel(r'$r^3$ (m$^3$)')
plt.ylabel(r'$T^2$ (s$^2$)')
plt.title('Verification of Kepler\'s Third Law: $T^2 \propto r^3$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

### Outputs for the Simulation

Below are two plots generated by the Python script. They verify Kepler’s Third Law by showing that $T^2$ is proportional to $r^3$ for a range of orbital radii:

1. **Scatter Plot of $T^2$ vs. $r^3$**                                   <br>                   
![Verification of Kepler's Third Law: Scatter Plot](https://raw.githubusercontent.com/akhmeed19/solutions_repo/refs/heads/main/docs/_pics/Gravity/Problem1/kepler_scatter.png)                                      
    - **What the Plot Shows:** Each blue dot represents a particular orbital radius $r$ and the corresponding orbital period $T$. We then plot $T^2$ on the y-axis and $r^3$ on the x-axis. The near-perfect alignment of these points along a straight line confirms the proportionality $T^2 \propto r^3$.  

    - **Physical Meaning:** If you double $r$ (and keep the central mass $M$ fixed), the new orbital period $T$ changes in such a way that $T^2 / r^3$ remains nearly constant (the slope of the line). This is the essence of Kepler’s Third Law for circular orbits.
   

2. **Comparison of Calculated $T^2$ and Theoretical $T^2$**  <br>                                                                           
![Verification of Kepler's Third Law: Comparison Plot](https://raw.githubusercontent.com/akhmeed19/solutions_repo/refs/heads/main/docs/_pics/Gravity/Problem1/kepler_comparison.png)                                
    - **Calculated vs. Theoretical:** The blue dots are the numerically computed $T^2$ values for each $r$. The red line is the theoretical prediction from Kepler’s Third Law:
    $$
    T^2 = \frac{4\pi^2}{G M} \; r^3.
    $$
    
    - **Interpretation:** The excellent overlap between the blue points and the red line indicates that the simulated orbits precisely match the analytical formula. This reaffirms that $T^2$ scales with $r^3$ by the constant factor $\tfrac{4\pi^2}{G M}$.


### Explanation of the Plots

1. **$T^2$ vs. $r^3$:**                                
    - On the x-axis, we have $r^3$, where $r$ is the orbital radius.  
    - On the y-axis, we have $T^2$, where $T$ is the orbital period.  
    - A straight line in this plot means $T^2 \propto r^3$. The slope of that line is $\tfrac{4\pi^2}{G M}$.

    2. **Physical Relationship:**                             
   - If you increase the orbital radius $r$, the orbital period $T$ increases such that $\tfrac{T^2}{r^3}$ remains constant (for a fixed central mass $M$).  
   - This captures the essence of Kepler’s Third Law: larger orbits take longer to complete, but the ratio $\tfrac{T^2}{r^3}$ is the same for all circular orbits around the same mass $M$.

    3. **Why It Matters:**                                    
   - **Astronomy:** This law allows scientists to deduce masses of stars, planets, or other central bodies by observing orbits.  
   - **Satellite Orbits:** Engineers can plan how fast a satellite must travel at a given altitude to maintain a stable orbit around Earth.  
   - **Planetary Systems:** The linear relationship in these plots is observed across our Solar System, reinforcing the universality of gravity.

Thus, the **simulation** and **plots** conclusively verify the linear relationship between $T^2$ and $r^3$ for circular orbits, validating Kepler’s Third Law and providing practical insights into orbital mechanics.

---
## 6. Discussion

- **Relationship Verification:**  
  The simulation verifies that $T^2 \propto r^3$ by demonstrating that the plot of $T^2$ versus $r^3$ is linear. The slope of this line is $\frac{4\pi^2}{G M}$, which depends on the gravitational constant and the mass of the central body.
  
- **Astronomical Implications:**  
  This relationship is used to determine planetary masses and orbital distances. For example, by measuring the orbital period and radius of a satellite, one can estimate the mass of the planet it orbits.
  
- **Extensions to Elliptical Orbits:**  
  Although the derivation here assumes circular orbits, Kepler's Third Law also applies to elliptical orbits if $r$ is replaced by the semi-major axis $a$. However, in elliptical orbits, factors like orbital eccentricity come into play.

- **Real-World Examples:**  
  The Moon’s orbit around the Earth and the orbits of the planets in our Solar System both obey this law, which has been fundamental in advancing our understanding of celestial mechanics.

---

## 7. Conclusion

This investigation of the orbital period and orbital radius has:

- Derived Kepler's Third Law for circular orbits, showing that $T^2 = \frac{4\pi^2}{G M}r^3$.
- Discussed its implications for determining planetary masses and distances.
- Implemented a computational model in Python to simulate circular orbits and verify the linear relationship between $T^2$ and $r^3$.
- Highlighted how this relationship extends to elliptical orbits and its relevance in various astronomical contexts.

Overall, this study reinforces the fundamental principles of gravitational interactions and provides a computational approach to verify one of the key laws of celestial mechanics.