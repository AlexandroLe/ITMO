import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G     = 6.67430e-11
M_sun = 1.98847e30
mu    = G * M_sun
au    = 1.495978707e11

a = 39.48 * au
e = 0.2488

r = np.array([ a*(1-e), 0.0 ])
v = np.array([ 0.0, np.sqrt(mu*(1+e)/(a*(1-e))) ])

dt       = 24*3600.0
max_days = int(248 * 365.25)

positions = []
y_prev    = r[1]

for i in range(max_days):
    a_vec  = -mu * r / np.linalg.norm(r)**3
    v_half = v + 0.5 * a_vec * dt
    r      = r + v_half * dt
    a_vec  = -mu * r / np.linalg.norm(r)**3
    v      = v_half + 0.5 * a_vec * dt

    positions.append(r.copy())
    if y_prev < 0 and r[1] >= 0 and v[1] > 0 and r[0] > 0:
        positions = positions[:i+1]
        break
    y_prev = r[1]

positions = np.array(positions)

period_days = len(positions) * dt / (24*3600)
print(f"Pluto's orbital period: {period_days:.2f} days")
period_years = period_days / 365.25
print(f"Pluto's orbital period: {period_years:.2f} Earth years")

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(positions[:,0]/au, positions[:,1]/au, lw=0.5)
ax.scatter(0, 0, color='orange', s=100, label='Солнце')
ax.set_title("Анимация орбиты Плутона")
ax.set_xlabel("x [AU]")
ax.set_ylabel("y [AU]")
ax.axis('equal')

pluto_dot, = ax.plot([], [], 'o',
                     color='white', markeredgecolor='black',
                     markersize=8, label='Плутон')
ax.legend(loc='upper right')

def init():
    x0, y0 = positions[0] / au
    pluto_dot.set_data([x0], [y0])
    return pluto_dot,

def update(frame):
    x, y = positions[frame] / au
    pluto_dot.set_data([x], [y])
    return pluto_dot,

ani = FuncAnimation(fig, update,
                    frames=len(positions),
                    init_func=init,
                    interval=0.1, blit=True)

plt.show()
