import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Parameters
L = 1.0  # Length of bar
alpha = 0.01  # Thermal diffusivity
nx = 100  # Number of spatial points
dx = L / (nx - 1)  # Spatial step
dt = 0.0001  # Time step (chosen for stability)
t_final = 2.0  # Final time

# Stability check (CFL condition: alpha*dt/dx^2 <= 0.5)
r = alpha * dt / dx**2
print(f"Stability parameter r = {r:.4f} (should be <= 0.5)")

# Initialize spatial grid
x = np.linspace(0, L, nx)

# Initial condition: Hat function (triangle) centered at L/2
T = np.zeros(nx)
for i in range(nx):
    if x[i] <= L/2:
        T[i] = 100 * (2 * x[i] / L)  # Rising slope
    else:
        T[i] = 100 * (2 - 2 * x[i] / L)  # Falling slope

# Store initial condition
T_initial = T.copy()

# Time evolution using explicit finite difference method
# We'll save snapshots at different times
times_to_plot = [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 10.0]
T_snapshots = []
time_labels = []

t = 0
snapshot_idx = 0

while t <= max(times_to_plot):
    # Check if we should save this snapshot
    if snapshot_idx < len(times_to_plot) and t >= times_to_plot[snapshot_idx]:
        T_snapshots.append(T.copy())
        time_labels.append(f't = {times_to_plot[snapshot_idx]:.2f}')
        snapshot_idx += 1
    
    # Update temperature using explicit finite difference
    T_new = T.copy()
    for i in range(1, nx-1):
        T_new[i] = T[i] + r * (T[i+1] - 2*T[i] + T[i-1])
    
    # Boundary conditions: Fixed at T=0 at both ends
    T_new[0] = 0
    T_new[-1] = 0
    
    T = T_new
    t += dt

# Create visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Top plot: Physical bar representation at different times
ax1.set_xlim(-0.05, L+0.05)
ax1.set_ylim(-0.5, len(T_snapshots))
ax1.set_xlabel('Position along bar (m)', fontsize=12)
ax1.set_title('Temperature Distribution in Bar Over Time', fontsize=14, fontweight='bold')
ax1.set_yticks(range(len(T_snapshots)))
ax1.set_yticklabels(time_labels)
ax1.invert_yaxis()

# Draw bars with color mapping
for idx, (T_snap, label) in enumerate(zip(T_snapshots, time_labels)):
    # Normalize temperature for color mapping
    T_norm = T_snap / 100
    
    # Draw bar segments with color
    for i in range(nx-1):
        color = plt.cm.hot(1 - T_norm[i])  # Hot colormap
        rect = Rectangle((x[i], idx - 0.3), dx, 0.6, 
                         facecolor=color, edgecolor='none')
        ax1.add_patch(rect)
    
    # Add border
    ax1.plot([0, L], [idx - 0.3, idx - 0.3], 'k-', linewidth=1)
    ax1.plot([0, L], [idx + 0.3, idx + 0.3], 'k-', linewidth=1)
    ax1.plot([0, 0], [idx - 0.3, idx + 0.3], 'k-', linewidth=1)
    ax1.plot([L, L], [idx - 0.3, idx + 0.3], 'k-', linewidth=1)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=0, vmax=100))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax1, orientation='vertical', pad=0.02)
cbar.set_label('Temperature (°C)', fontsize=11)

# Bottom plot: Temperature profiles
ax2.set_xlabel('Position along bar (m)', fontsize=12)
ax2.set_ylabel('Temperature (°C)', fontsize=12)
ax2.set_title('Temperature Profile Evolution', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot all snapshots
colors = plt.cm.viridis(np.linspace(0, 1, len(T_snapshots)))
for idx, (T_snap, label) in enumerate(zip(T_snapshots, time_labels)):
    if idx == 0:
        ax2.plot(x, T_snap, color=colors[idx], linewidth=2.5, 
                label=label, linestyle='--')
    elif idx == len(T_snapshots) - 1:
        ax2.plot(x, T_snap, color=colors[idx], linewidth=2.5, 
                label=label + ' (t → ∞)')
    else:
        ax2.plot(x, T_snap, color=colors[idx], linewidth=1.5, 
                label=label, alpha=0.8)

ax2.legend(loc='upper right', fontsize=10)
ax2.set_xlim(0, L)
ax2.set_ylim(-5, 105)

plt.tight_layout()
plt.savefig('heat_equation_bar.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'heat_equation_bar.png'")

# Print steady state information
print(f"\nSteady state (t → ∞): T = 0 everywhere (both ends fixed at T=0)")
print(f"Maximum temperature at final time: {T_snapshots[-1].max():.6f}°C")
plt.show()
