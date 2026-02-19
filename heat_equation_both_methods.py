import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solve_heat_equation_explicit(L, f, n, alpha, dt, t_final, BC_type='dirichlet', BC_values=(0, 0)):
    """
    Solve 1D heat equation using explicit (forward Euler) finite difference method.
    
    Parameters:
    -----------
    L : float
        Length of the bar
    f : callable
        Initial condition u(x,0) = f(x)
    n : int
        Number of spatial intervals (n+1 total points)
    alpha : float
        Thermal diffusivity
    dt : float
        Time step size
    t_final : float
        Final simulation time
    BC_type : str
        'dirichlet' for fixed temperature at boundaries
    BC_values : tuple
        (T_left, T_right) boundary temperatures
    
    Returns:
    --------
    x : array
        Spatial grid points
    T : array
        Temperature distribution at t_final
    times : list
        Time points where snapshots were saved
    T_snapshots : list
        Temperature distributions at saved times
    """
    n = int(n)
    dx = L / n
    x = np.linspace(0, L, n + 1)
    T_initial = f(x)

    # Stability parameter
    r = alpha * dt / dx**2
    print(f"Explicit method - Stability parameter r = {r:.4f}")
    if r > 0.5:
        print("WARNING: Explicit method is unstable! Need r <= 0.5")

    T = T_initial.copy()
    t = 0

    # Store snapshots
    times = [0]
    T_snapshots = [T.copy()]
    snapshot_interval = t_final / 10
    next_snapshot = snapshot_interval

    # Time stepping
    n_steps = int(t_final / dt)
    for step in range(n_steps):
        T_new = T.copy()

        # Update interior points
        for i in range(1, len(x) - 1):
            T_new[i] = T[i] + r * (T[i+1] - 2*T[i] + T[i-1])

        # Apply boundary conditions
        if BC_type == 'dirichlet':
            T_new[0] = BC_values[0]
            T_new[-1] = BC_values[1]

        T = T_new
        t += dt

        # Save snapshots
        if t >= next_snapshot:
            times.append(t)
            T_snapshots.append(T.copy())
            next_snapshot += snapshot_interval

    # Make sure we have the final state
    if times[-1] < t_final:
        times.append(t)
        T_snapshots.append(T.copy())

    return x, T, times, T_snapshots


def solve_heat_equation_crank_nicolson(L, f, n, alpha, dt, t_final, BC_type='dirichlet', BC_values=(0, 0)):
    """
    Solve 1D heat equation using Crank-Nicolson (implicit) finite difference method.
    
    This method is unconditionally stable and second-order accurate in both space and time.
    
    Parameters: Same as explicit method
    
    Returns: Same as explicit method
    """
    n = int(n)
    dx = L / n
    x = np.linspace(0, L, n + 1)
    T_initial = f(x)

    r = alpha * dt / dx**2
    print(f"Crank-Nicolson method - Stability parameter r = {r:.4f}")
    print("(Crank-Nicolson is unconditionally stable)")

    T = T_initial.copy()
    t = 0

    # Store snapshots
    times = [0]
    T_snapshots = [T.copy()]
    snapshot_interval = t_final / 10
    next_snapshot = snapshot_interval

    # Build the tridiagonal matrices for Crank-Nicolson
    # The scheme is: (I - r/2 * A) * T^(n+1) = (I + r/2 * A) * T^n
    n_interior = n - 1

    # Left side matrix: (I - r/2 * A)
    main_diag_L = np.ones(n_interior) * (1 + r)
    off_diag_L = np.ones(n_interior - 1) * (-r/2)
    A_left = diags([off_diag_L, main_diag_L, off_diag_L], [-1, 0, 1], format='csr')

    # Right side matrix: (I + r/2 * A)
    main_diag_R = np.ones(n_interior) * (1 - r)
    off_diag_R = np.ones(n_interior - 1) * (r/2)
    A_right = diags([off_diag_R, main_diag_R, off_diag_R], [-1, 0, 1], format='csr')

    # Time stepping
    n_steps = int(t_final / dt)
    for step in range(n_steps):
        rhs = A_right @ T[1:-1]

        if BC_type == 'dirichlet':
            rhs[0] += (r/2) * BC_values[0]
            rhs[-1] += (r/2) * BC_values[1]

        T_interior = spsolve(A_left, rhs)

        T[1:-1] = T_interior
        T[0] = BC_values[0]
        T[-1] = BC_values[1]

        t += dt

        if t >= next_snapshot:
            times.append(t)
            T_snapshots.append(T.copy())
            next_snapshot += snapshot_interval

    if times[-1] < t_final:
        times.append(t)
        T_snapshots.append(T.copy())

    return x, T, times, T_snapshots


# =============================================================================
# User inputs
# =============================================================================
L = 1.0          # Length of bar (m)
alpha = 0.01     # Thermal diffusivity (m^2/s)
n = 100          # Number of spatial intervals (n+1 points)
f = lambda x: np.sin(np.pi * x)   # Initial condition f(x)
t_final = 0.5    # Final time (s)

# =============================================================================
# Solve using both methods
# =============================================================================
print("="*60)
print("EXPLICIT METHOD")
print("="*60)
dt_explicit = 0.0001
x_exp, T_exp, times_exp, snapshots_exp = solve_heat_equation_explicit(
    L, f, n, alpha, dt_explicit, t_final)

print("\n" + "="*60)
print("CRANK-NICOLSON METHOD")
print("="*60)
dt_cn = 0.001
x_cn, T_cn, times_cn, snapshots_cn = solve_heat_equation_crank_nicolson(
    L, f, n, alpha, dt_cn, t_final)

# =============================================================================
# Comparison
# =============================================================================
x = x_exp  # Both grids are identical
T_initial = f(x)

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"Explicit method used dt = {dt_explicit} ({int(t_final/dt_explicit)} steps)")
print(f"Crank-Nicolson used dt = {dt_cn} ({int(t_final/dt_cn)} steps)")
print(f"Maximum difference at t={t_final}: {np.max(np.abs(T_exp - T_cn)):.6e} °C")

# =============================================================================
# Plots
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(x, T_initial, 'k--', linewidth=2, label='Initial condition', alpha=0.5)
ax.plot(x_exp, T_exp, 'b-', linewidth=2, label=f'Explicit (dt={dt_explicit})')
ax.plot(x_cn, T_cn, 'r--', linewidth=2, label=f'Crank-Nicolson (dt={dt_cn})')
ax.set_xlabel('Position (m)', fontsize=11)
ax.set_ylabel('Temperature (°C)', fontsize=11)
ax.set_title(f'Final Temperature Distribution (t = {t_final} s)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
colors_exp = plt.cm.viridis(np.linspace(0, 1, len(snapshots_exp)))
for i, (T_snap, t) in enumerate(zip(snapshots_exp, times_exp)):
    alpha_val = 0.3 + 0.7 * (i / len(snapshots_exp))
    ax.plot(x, T_snap, color=colors_exp[i], linewidth=1.5,
            alpha=alpha_val, label=f't={t:.3f}s' if i % 2 == 0 else '')
ax.set_xlabel('Position (m)', fontsize=11)
ax.set_ylabel('Temperature (°C)', fontsize=11)
ax.set_title(f'Explicit Method Evolution (dt={dt_explicit})', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
colors_cn = plt.cm.plasma(np.linspace(0, 1, len(snapshots_cn)))
for i, (T_snap, t) in enumerate(zip(snapshots_cn, times_cn)):
    alpha_val = 0.3 + 0.7 * (i / len(snapshots_cn))
    ax.plot(x, T_snap, color=colors_cn[i], linewidth=1.5,
            alpha=alpha_val, label=f't={t:.3f}s' if i % 2 == 0 else '')
ax.set_xlabel('Position (m)', fontsize=11)
ax.set_ylabel('Temperature (°C)', fontsize=11)
ax.set_title(f'Crank-Nicolson Evolution (dt={dt_cn})', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
difference = np.abs(T_exp - T_cn)
ax.plot(x, difference, 'g-', linewidth=2)
ax.set_xlabel('Position (m)', fontsize=11)
ax.set_ylabel('|T_explicit - T_CN| (°C)', fontsize=11)
ax.set_title('Absolute Difference Between Methods', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('heat_equation_comparison.png', dpi=300, bbox_inches='tight')
print("\nComparison plot saved as 'heat_equation_comparison.png'")

# Scheme diagrams
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

ax = axes2[0]
ax.text(0.5, 0.95, 'Explicit (Forward Euler) Scheme',
        ha='center', va='top', fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.85, r'$T_i^{n+1} = T_i^n + r(T_{i+1}^n - 2T_i^n + T_{i-1}^n)$',
        ha='center', va='top', fontsize=12, transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.text(0.5, 0.70, r'where $r = \alpha \frac{\Delta t}{\Delta x^2}$',
        ha='center', va='top', fontsize=11, transform=ax.transAxes)
ax.text(0.1, 0.55, 'Advantages:', ha='left', va='top', fontsize=11, fontweight='bold', transform=ax.transAxes)
ax.text(0.15, 0.48, '• Simple to implement', ha='left', va='top', fontsize=10, transform=ax.transAxes)
ax.text(0.15, 0.42, '• Computationally cheap per step', ha='left', va='top', fontsize=10, transform=ax.transAxes)
ax.text(0.15, 0.36, '• Explicit update (no linear solve)', ha='left', va='top', fontsize=10, transform=ax.transAxes)
ax.text(0.1, 0.28, 'Disadvantages:', ha='left', va='top', fontsize=11, fontweight='bold', transform=ax.transAxes)
ax.text(0.15, 0.21, '• Conditionally stable (r ≤ 0.5)', ha='left', va='top', fontsize=10, transform=ax.transAxes)
ax.text(0.15, 0.15, '• Requires small time steps', ha='left', va='top', fontsize=10, transform=ax.transAxes)
ax.text(0.15, 0.09, '• First-order accurate in time', ha='left', va='top', fontsize=10, transform=ax.transAxes)
ax.axis('off')

ax = axes2[1]
ax.text(0.5, 0.95, 'Crank-Nicolson (Implicit) Scheme',
        ha='center', va='top', fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.85, r'$T_i^{n+1} - \frac{r}{2}(T_{i+1}^{n+1} - 2T_i^{n+1} + T_{i-1}^{n+1})$',
        ha='center', va='top', fontsize=11, transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.text(0.5, 0.75, r'$= T_i^n + \frac{r}{2}(T_{i+1}^n - 2T_i^n + T_{i-1}^n)$',
        ha='center', va='top', fontsize=11, transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.text(0.5, 0.62, '(Requires solving a tridiagonal linear system)',
        ha='center', va='top', fontsize=10, transform=ax.transAxes, style='italic')
ax.text(0.1, 0.52, 'Advantages:', ha='left', va='top', fontsize=11, fontweight='bold', transform=ax.transAxes)
ax.text(0.15, 0.45, '• Unconditionally stable (any dt)', ha='left', va='top', fontsize=10, transform=ax.transAxes)
ax.text(0.15, 0.39, '• Second-order accurate in time', ha='left', va='top', fontsize=10, transform=ax.transAxes)
ax.text(0.15, 0.33, '• Can use larger time steps', ha='left', va='top', fontsize=10, transform=ax.transAxes)
ax.text(0.1, 0.25, 'Disadvantages:', ha='left', va='top', fontsize=11, fontweight='bold', transform=ax.transAxes)
ax.text(0.15, 0.18, '• More complex to implement', ha='left', va='top', fontsize=10, transform=ax.transAxes)
ax.text(0.15, 0.12, '• Requires linear system solve', ha='left', va='top', fontsize=10, transform=ax.transAxes)
ax.text(0.15, 0.06, '• More expensive per time step', ha='left', va='top', fontsize=10, transform=ax.transAxes)
ax.axis('off')

plt.tight_layout()
plt.savefig('numerical_schemes_comparison.png', dpi=300, bbox_inches='tight')
print("Schemes comparison diagram saved as 'numerical_schemes_comparison.png'")

plt.show()