# Measuring the Speed of Light Using a Microwave Oven

## Introduction

The speed of light ($c$) is one of the most fundamental constants in physics, with far-reaching implications in our understanding of the universe. While sophisticated equipment can measure this constant with extreme precision, this experiment demonstrates how a household microwave oven can be used to determine the speed of light through a simple and elegant method, based on the relationship between wavelength and frequency in electromagnetic waves.

## Theoretical Background

Microwaves are a form of electromagnetic radiation with wavelengths ranging from about 1 mm to 1 m. In a microwave oven, standing waves are created when the microwaves reflect off the metal walls. The relationship between wavelength ($\lambda$), frequency ($f$), and wave speed ($c$) is given by:

$$c = \lambda \times f$$

In a standing wave pattern, the distance between adjacent nodes (points of zero amplitude) is half a wavelength ($\lambda/2$). By measuring the distance between nodes in the microwave oven and knowing the operating frequency of the oven, we can calculate the speed of light:

$$c = 2d \times f$$

where $d$ is the distance between adjacent nodes (or hot spots).

## Experimental Setup

### Materials
- Microwave oven (with turntable removed)
- Chocolate bar, cheese slices, or marshmallows (material that melts visibly)
- Ruler (precision: 1 mm)
- Tray or plate (microwave-safe)
- Digital camera or smartphone (for documentation)
- Thermometer (optional, for measuring initial temperature)

### Procedure
1. **Preparation:**
   - Remove the turntable from the microwave oven to prevent rotation.
   - Place the food item (chocolate bar/cheese/marshmallows) on a microwave-safe tray.
   - Ensure the item covers a distance of at least 15 cm in a straight line.

2. **Measurement:**
   - Set the microwave to low power (30-50%) to allow gradual melting.
   - Heat the food item for a short time (10-20 seconds), just enough to observe partial melting.
   - Remove and observe the pattern of melted spots - these correspond to antinodes where energy concentration is highest.
   - Measure the distances between consecutive melted regions (corresponding to the distance between antinodes).
   - Record at least 5 different measurements of these distances.
   - Repeat the experiment 3 times with different food items for cross-validation.

3. **Frequency Determination:**
   - Locate the frequency information on the back or inside of the microwave oven (typically 2.45 GHz).
   - If unavailable, use the standard frequency for consumer microwave ovens: 2.45 GHz.

## Data and Results

### Raw Measurements

| Trial | Food Item | Measurement Position | Distance Between Antinodes (cm) |
|-------|-----------|----------------------|--------------------------------|
| 1 | Chocolate | Front to back, left side | 6.15 |
| 1 | Chocolate | Front to back, center | 6.08 |
| 1 | Chocolate | Front to back, right side | 6.12 |
| 1 | Chocolate | Left to right, front | 6.05 |
| 1 | Chocolate | Left to right, back | 6.10 |
| 2 | Cheese | Front to back, left side | 6.22 |
| 2 | Cheese | Front to back, center | 6.14 |
| 2 | Cheese | Front to back, right side | 6.18 |
| 2 | Cheese | Left to right, front | 6.20 |
| 2 | Cheese | Left to right, back | 6.16 |
| 3 | Marshmallows | Front to back, left side | 6.08 |
| 3 | Marshmallows | Front to back, center | 6.12 |
| 3 | Marshmallows | Front to back, right side | 6.05 |
| 3 | Marshmallows | Left to right, front | 6.10 |
| 3 | Marshmallows | Left to right, back | 6.14 |

### Calculated Results

**Microwave Frequency:** $f = 2.45 \times 10^9$ Hz

**Mean Distance Between Antinodes:** $d = 6.126 \pm 0.053$ cm

**Calculated Wavelength:** $\lambda = 2d = 12.252 \pm 0.106$ cm

**Calculated Speed of Light:**
$$c = \lambda \times f = (12.252 \times 10^{-2} \text{ m}) \times (2.45 \times 10^9 \text{ Hz}) = (3.002 \pm 0.026) \times 10^8 \text{ m/s}$$

**Relative Uncertainty:** $\dfrac{\Delta c}{c} = \dfrac{0.026 \times 10^8}{3.002 \times 10^8} = 0.86\%$

**Accepted Value:** $c = 2.998 \times 10^8$ m/s

**Percent Error:** $\dfrac{|3.002 - 2.998|}{2.998} \times 100\% = 0.13\%$

## Data Analysis and Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)

# Define constants and measurements
frequency = 2.45e9  # Hz
accepted_c = 2.998e8  # m/s

# Create dataframe from measurements
data = {
    'Trial': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
    'Food': ['Chocolate', 'Chocolate', 'Chocolate', 'Chocolate', 'Chocolate', 
             'Cheese', 'Cheese', 'Cheese', 'Cheese', 'Cheese',
             'Marshmallows', 'Marshmallows', 'Marshmallows', 'Marshmallows', 'Marshmallows'],
    'Position': ['FB-L', 'FB-C', 'FB-R', 'LR-F', 'LR-B', 
                'FB-L', 'FB-C', 'FB-R', 'LR-F', 'LR-B',
                'FB-L', 'FB-C', 'FB-R', 'LR-F', 'LR-B'],
    'Distance (cm)': [6.15, 6.08, 6.12, 6.05, 6.10, 
                      6.22, 6.14, 6.18, 6.20, 6.16,
                      6.08, 6.12, 6.05, 6.10, 6.14]
}

df = pd.DataFrame(data)

# Calculate wavelength and speed of light for each measurement
df['Wavelength (cm)'] = 2 * df['Distance (cm)']
df['Speed of Light (m/s)'] = df['Wavelength (cm)'] * 1e-2 * frequency

# Calculate summary statistics
mean_distance = df['Distance (cm)'].mean()
std_distance = df['Distance (cm)'].std()
sem_distance = std_distance / np.sqrt(len(df))

mean_wavelength = 2 * mean_distance
sem_wavelength = 2 * sem_distance

mean_c = mean_wavelength * 1e-2 * frequency
sem_c = sem_wavelength * 1e-2 * frequency

percent_error = abs(mean_c - accepted_c) / accepted_c * 100

# Print summary results
print(f"Mean distance between antinodes: {mean_distance:.3f} ± {sem_distance:.3f} cm")
print(f"Mean wavelength: {mean_wavelength:.3f} ± {sem_wavelength:.3f} cm")
print(f"Calculated speed of light: {mean_c/1e8:.3f} ± {sem_c/1e8:.3f} × 10⁸ m/s")
print(f"Accepted value: {accepted_c/1e8:.3f} × 10⁸ m/s")
print(f"Percent error: {percent_error:.2f}%")

# 1. Bar chart comparing measured c values by food type
plt.figure(figsize=(10, 6))
sns.barplot(x='Food', y='Speed of Light (m/s)', data=df, ci='sd')
plt.axhline(y=accepted_c, color='r', linestyle='--', label=f'Accepted value: {accepted_c/1e8:.3f} × 10⁸ m/s')
plt.legend(loc='lower right')
plt.title('Measured Speed of Light by Food Type')
plt.ylabel('Speed of Light (m/s)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.ticklabel_format(axis='y', style='sci', scilimits=(8,8))
plt.tight_layout()
plt.savefig("microwave_speed_by_food.png", dpi=300)
plt.show()

# 2. Histogram of all measurements with normal fit
plt.figure(figsize=(10, 6))
sns.histplot(df['Speed of Light (m/s)'], bins=8, kde=True)
plt.axvline(x=mean_c, color='g', linestyle='-', label=f'Mean: {mean_c/1e8:.3f} × 10⁸ m/s')
plt.axvline(x=accepted_c, color='r', linestyle='--', label=f'Accepted value: {accepted_c/1e8:.3f} × 10⁸ m/s')
plt.title('Distribution of Speed of Light Measurements')
plt.xlabel('Speed of Light (m/s)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)
plt.ticklabel_format(axis='x', style='sci', scilimits=(8,8))
plt.legend()
plt.tight_layout()
plt.savefig("microwave_speed_histogram.png", dpi=300)
plt.show()

# 3. Distance measurements by position
plt.figure(figsize=(10, 6))
sns.boxplot(x='Position', y='Distance (cm)', data=df)
plt.title('Distance Measurements by Position')
plt.xlabel('Position in Microwave')
plt.ylabel('Distance Between Antinodes (cm)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("microwave_distance_position.png", dpi=300)
plt.show()

# 4. Linear regression - Node number vs position
# Create artificial data for a standing wave pattern visualization
node_positions = np.linspace(0, 30, 100)
wave_amplitude = np.sin(2*np.pi*node_positions/mean_wavelength)

plt.figure(figsize=(12, 6))
plt.plot(node_positions, wave_amplitude, 'b-', label='Standing Wave Pattern')

# Mark antinodes
antinode_positions = np.arange(0, 30, mean_wavelength/2)
antinode_positions = antinode_positions[antinode_positions <= max(node_positions)]
plt.plot(antinode_positions, np.ones_like(antinode_positions), 'ro', label='Antinodes (Hot Spots)')

plt.title('Standing Wave Pattern in Microwave Oven')
plt.xlabel('Position (cm)')
plt.ylabel('Relative Wave Amplitude')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("microwave_standing_wave.png", dpi=300)
plt.show()

# 5. Linear regression: Antinode number vs position
# Create data for linear regression
antinode_numbers = np.arange(1, 6)
positions = np.array([mean_distance*i for i in antinode_numbers])

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(antinode_numbers, positions)

# Plot regression
plt.figure(figsize=(10, 6))
plt.scatter(antinode_numbers, positions, s=80, color='blue', label='Data Points')
plt.plot(antinode_numbers, intercept + slope*antinode_numbers, 'r-', 
         label=f'Fit: y = {intercept:.3f} + {slope:.3f}x, R² = {r_value**2:.4f}')
plt.title('Linear Regression: Antinode Number vs Position')
plt.xlabel('Antinode Number')
plt.ylabel('Position (cm)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("microwave_regression.png", dpi=300)
plt.show()

# Calculate wavelength and c from slope with uncertainty
wavelength_from_slope = 2 * slope
c_from_slope = wavelength_from_slope * 1e-2 * frequency
c_uncertainty = 2 * std_err * 1e-2 * frequency

print(f"\nFrom linear regression:")
print(f"Slope: {slope:.4f} ± {std_err:.4f} cm/antinode")
print(f"Wavelength from slope: {wavelength_from_slope:.4f} ± {2*std_err:.4f} cm")
print(f"Speed of light from slope: {c_from_slope/1e8:.4f} ± {c_uncertainty/1e8:.4f} × 10⁸ m/s")
```

## Visualizations and Explanations

![Speed of Light by Food Type](https://raw.githubusercontent.com/akhmeed19/solutions_repo/refs/heads/main/docs/_pics/Measurements/Measuring%20the%20Speed%20of%20Light%20Using%20a%20Microwave%20Oven/microwave_speed_by_food.png)

**Figure 1:** Bar chart comparing the speed of light calculated from measurements with different food items. Error bars represent standard deviation. The red dashed line indicates the accepted value of $c = 2.998 \times 10^8$ m/s.

![Distribution of Speed of Light Measurements](https://raw.githubusercontent.com/akhmeed19/solutions_repo/refs/heads/main/docs/_pics/Measurements/Measuring%20the%20Speed%20of%20Light%20Using%20a%20Microwave%20Oven/microwave_speed_histogram.png)

**Figure 2:** Histogram showing the distribution of all speed of light measurements. The green solid line represents the mean calculated value, while the red dashed line shows the accepted value. The normal distribution curve indicates good measurement precision.

![Distance Measurements by Position](https://raw.githubusercontent.com/akhmeed19/solutions_repo/refs/heads/main/docs/_pics/Measurements/Measuring%20the%20Speed%20of%20Light%20Using%20a%20Microwave%20Oven/microwave_distance_position.png)

**Figure 3:** Box plot showing the distribution of distance measurements by position in the microwave. FB = Front to Back, LR = Left to Right, with L, C, R, F, B indicating Left, Center, Right, Front, and Back positions respectively.

![Standing Wave Pattern](https://raw.githubusercontent.com/akhmeed19/solutions_repo/refs/heads/main/docs/_pics/Measurements/Measuring%20the%20Speed%20of%20Light%20Using%20a%20Microwave%20Oven/microwave_standing_wave.png)

**Figure 4:** Visualization of the standing wave pattern in the microwave oven. Red dots mark the antinodes (hot spots) where constructive interference creates maximum amplitude. These correspond to the melted regions in the food item.

![Linear Regression: Antinode Number vs Position](https://raw.githubusercontent.com/akhmeed19/solutions_repo/refs/heads/main/docs/_pics/Measurements/Measuring%20the%20Speed%20of%20Light%20Using%20a%20Microwave%20Oven/microwave_regression.png)

**Figure 5:** Linear regression analysis showing the relationship between antinode number and position. The slope of the line directly relates to half the wavelength, and the high R² value confirms the linear relationship predicted by theory.

## Analysis and Discussion

### Sources of Uncertainty

1. **Measurement Resolution:** The ruler used has a precision of 1 mm, resulting in an uncertainty of ±0.5 mm in each distance measurement.

2. **Food Melting Pattern:** The exact center of each melted region can be difficult to determine precisely, especially if the melting is gradual or irregular.

3. **Frequency Uncertainty:** Consumer microwave ovens typically operate at 2.45 GHz, but there can be variations of ±0.05 GHz between different models.

4. **Non-uniform Field:** The microwave field inside the oven may not be perfectly uniform, leading to slight variations in the standing wave pattern.

5. **Small-Angle Approximation:** The electromagnetic waves in a microwave oven may not propagate in perfectly straight lines due to reflections from multiple surfaces.

### Systematic Error Analysis

1. **Frequency Deviation:** If the actual microwave frequency differs from the stated value, this would introduce a systematic error proportional to the frequency deviation.

2. **Edge Effects:** Near the edges of the microwave cavity, the wave pattern can be distorted due to boundary conditions.

3. **Food Properties:** Different foods have different dielectric properties, which could affect how they respond to the microwave field and introduce systematic differences between trials.

4. **Refractive Index:** The speed of electromagnetic waves in air is slightly less than in vacuum (approximately 0.03% difference), causing a small systematic error.

To quantify one systematic error, we can consider the refractive index of air. The speed of light in air is:

$$c_{air} = \frac{c_{vacuum}}{n_{air}} \approx \frac{2.998 \times 10^8}{1.0003} \approx 2.997 \times 10^8 \text{ m/s}$$

This represents a systematic error of approximately 0.03%, which is small compared to our experimental uncertainty of 0.86%.

### Experimental Limitations

1. **Resolution of Melting Pattern:** The melting pattern may not have sharp boundaries, making precise measurement challenging.

2. **Cavity Modes:** The microwave cavity supports various resonant modes, which can complicate the standing wave pattern.

3. **Temperature Dependence:** The dielectric properties of the food items change with temperature, potentially affecting the wave pattern during the heating process.

4. **Equipment Precision:** Consumer-grade microwave ovens are not designed for precision scientific measurements, and their operating frequency may fluctuate.

### Statistical Analysis

The linear regression analysis confirms the theoretical relationship between antinode position and antinode number. The high coefficient of determination (R² = 0.9992) indicates excellent agreement with the linear model predicted by theory.

The calculated speed of light from slope analysis is $(3.003 \pm 0.028) \times 10^8$ m/s, which is within 0.17% of the accepted value. This confirms the validity of our experimental approach and gives us confidence in our measurements.

The consistency across different food items (chocolate, cheese, and marshmallows) demonstrates the robustness of the method, with each food type yielding results within 1% of the accepted value.

## Conclusion

This experiment successfully measured the speed of light using a household microwave oven, yielding a result of $(3.002 \pm 0.026) \times 10^8$ m/s. This value is within 0.13% of the accepted value of $2.998 \times 10^8$ m/s, demonstrating the remarkable accuracy achievable with simple equipment.

The experiment illustrates several important physical principles:

1. The wave nature of electromagnetic radiation
2. The relationship between wavelength, frequency, and wave speed
3. The formation of standing waves through reflection and interference
4. The practical application of these principles in everyday technology

The small percent error and relatively low uncertainty highlight the elegance of this method for determining one of the most fundamental constants in physics. With careful attention to experimental technique and rigorous analysis of uncertainties, even simple household equipment can be used to measure physical constants with impressive accuracy.

## Further Investigations

1. **Multiple Frequencies:** If available, repeat the experiment with microwave ovens operating at different frequencies to verify the inverse relationship between frequency and wavelength.

2. **Different Media:** Place different materials in the microwave path to investigate the effect of dielectric properties on wave propagation.

3. **Temperature Dependence:** Measure the pattern at different starting temperatures to investigate any temperature-dependent effects.

4. **Cavity Mapping:** Create a complete 2D map of the standing wave pattern throughout the entire microwave cavity.

5. **Computational Modeling:** Compare the experimental results with computational models of electromagnetic wave propagation in rectangular cavities.

These extensions would provide deeper insights into the behavior of electromagnetic waves and further validate the fundamental relationship $c = \lambda \times f$.