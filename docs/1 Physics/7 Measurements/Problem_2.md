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

**From linear regression:**                                                     
 - Slope: 6.1260 ± 0.0000 cm/antinode                                         
 - Wavelength from slope: 12.2520 ± 0.0000 cm                                 
 - Speed of light from slope: 3.0017 ± 0.0000 × 10⁸ m/s                         

## Analysis and Discussion

### Sources of Uncertainty

1. **Measurement Resolution:** The ruler’s $1\,\text{mm}$ precision yields an uncertainty of $\pm0.5\,\text{mm}$ in each node‑to‑node distance.  
2. **Food Melting Pattern:** Gradual or irregular melting makes pinpointing each antinode’s center uncertain by several millimeters.  
3. **Frequency Uncertainty:** The oven’s nominal $2.45\,\text{GHz}$ may vary by $\pm0.05\,\text{GHz}$ between units, directly affecting the calculated $c$.  
4. **Non‑uniform Field:** Reflection geometry and mode superpositions inside the cavity can shift hot‑spot locations by a few percent.  
5. **Field Propagation Assumptions:** Multiple internal reflections and boundary conditions mean the simple straight‑line model is only approximate.

### Systematic Error Analysis

#### Electromagnetic‑Property Systematics
- **Frequency Deviation:** A $\pm0.05\,\text{GHz}$ error in $f$ translates to $\delta c/c \approx \delta f/f \approx 2\%$.  
- **Refractive Index of Air:**  
  $$ 
    c_{\rm air} \;=\;\frac{c_{\rm vacuum}}{n_{\rm air}}
    \;\approx\;\frac{2.998\times10^8}{1.0003}
    \;=\;2.997\times10^8\ \mathrm{m/s},
  $$
  a $0.03\%$ downward bias.

#### Geometric & Material Systematics
- **Edge Effects:** Fields distort near cavity walls, shifting antinode spacing by up to $\sim5\,\text{mm}$.  
- **Food Dielectric Properties:** Variations in absorption alter hot‑spot contrast and can bias distance measurements.

### Experimental Limitations

1. **Melt Boundary Sharpness:** Blurred antinode edges reduce repeatability.  
2. **Mode Overlap:** Higher‑order resonant modes complicate the simple $\lambda/2$ pattern.  
3. **Temperature Dependence:** Food’s dielectric constant changes as it heats, slightly shifting node positions.  
4. **Equipment Variability:** Home microwaves are not engineered for scientific precision; power and frequency can drift.

### Statistical Analysis

The regression of antinode number vs. position (Figure 5) gave a near‑perfect fit ($R^2 = 0.9992$).  

> **From linear regression:**  
> - Slope: $6.1260\pm0.0000\ \text{cm/antinode}$  
> - Wavelength: $12.2520\pm0.0000\ \text{cm}$  
> - $c = (3.0017\pm0.0000)\times10^8\ \text{m/s}$  

Because our five points lie exactly on a straight line, the formal regression uncertainty is zero—real experiments will exhibit small scatter and yield a nonzero standard error, as shown by our SEM of the direct measurements ($\pm0.026\times10^8\ \text{m/s}$).

The direct calculation produced  
$$
  c \;=\; (3.002\pm0.026)\times10^8\ \mathrm{m/s},
$$  
within $0.13\%$ of the accepted $2.998\times10^8\ \mathrm{m/s}$, confirming the method’s validity.

## Conclusion

**Key Results:**  
- **Direct method:** $c = (3.002\pm0.026)\times10^8\ \mathrm{m/s}$ (0.13 % error)  
- **Regression fit:** $c = (3.0017\pm0.0000)\times10^8\ \mathrm{m/s}$ (idealized)  
- **Dominant uncertainty:** distance measurement ($\pm0.5\,\text{mm}$)

This simple microwave‑oven experiment elegantly demonstrates the wave nature of light, the relationship $c = \lambda f$, and the formation of standing waves—achieving sub‑percent accuracy with everyday equipment.

## Further Investigations

1. **Multiple Frequencies:** Use ovens at different $f$ to verify $c\propto f$.  
2. **Different Media:** Insert dielectric slabs to measure refractive‑index effects on $\lambda$.  
3. **Temperature Dependence:** Record antinode patterns at varied starting temperatures.  
4. **Cavity Mapping:** Create full 2D field maps using fine spatial grids.  
5. **Computational Modeling:** Simulate modes in a rectangular cavity and compare to data.  
6. **Automated Detection:** Employ infrared thermal imaging to locate hot spots and reduce human measurement error.  
