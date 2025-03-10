# Problem 1

import numpy as np
import matplotlib.pyplot as plt

# Constants

G = 9.81 # Acceleration due to gravity (m/s^2)
V0 = 20 # Initial velocity (m/s)
angles = np.linspace(0, 90, 100) # Angle range from 0 to 90 degrees

# Function to calculate range of projectile

def calculate_range(v0, theta, g=G):
theta_rad = np.radians(theta) # Convert degrees to radians
return (v0\*_2 _ np.sin(2 \* theta_rad)) / g

# Compute ranges for different angles

ranges = [calculate_range(V0, angle) for angle in angles]

# Plot Range vs. Angle

plt.figure(figsize=(10, 5))
plt.plot(angles, ranges, label=f'Initial Velocity = {V0} m/s')
plt.xlabel("Angle of Projection (degrees)")
plt.ylabel("Range (m)")
plt.title("Projectile Range vs. Angle of Projection")
plt.legend()
plt.grid()
plt.show()

# Discussion

print("\nObservations:")
print("1. The range is maximum at 45 degrees.")
print("2. The range decreases symmetrically for angles less and greater than 45 degrees.")
print("3. Higher initial velocity results in a longer range.")
print("4. A lower gravitational acceleration (e.g., on the Moon) would result in a greater range.")
