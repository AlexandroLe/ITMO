import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Константы ---
G     = 6.67430e-11            # гравитационная постоянная, м^3⋅кг^–1⋅с^–2
M_sun = 1.98847e30             # масса Солнца, кг
mu    = G * M_sun
au    = 1.495978707e11         # 1 AU в метрах

dt       = 24 * 3600.0         # шаг интегрирования — 1 день в секундах
max_days = int(248 * 365.25)   # количество дней для ~1 оборота Плутона

# --- Орбитальные параметры (полуось a, эксцентриситет e) ---
# Pluto
a_p, e_p = 39.48 * au, 0.2488
r_p      = np.array([ a_p*(1-e_p), 0.0 ])
v_p      = np.array([ 0.0, np.sqrt(mu*(1+e_p)/(a_p*(1-e_p))) ])

# Earth
a_e, e_e = 1.0   * au, 0.0167
r_e      = np.array([ a_e*(1-e_e), 0.0 ])
v_e      = np.array([ 0.0, np.sqrt(mu*(1+e_e)/(a_e*(1-e_e))) ])

# Venus
a_v, e_v = 0.723 * au, 0.0068
r_v      = np.array([ a_v*(1-e_v), 0.0 ])
v_v      = np.array([ 0.0, np.sqrt(mu*(1+e_v)/(a_v*(1-e_v))) ])

# Mercury
a_m, e_m = 0.387 * au, 0.2056
r_m      = np.array([ a_m*(1-e_m), 0.0 ])
v_m      = np.array([ 0.0, np.sqrt(mu*(1+e_m)/(a_m*(1-e_m))) ])

# Mars
a_ma, e_ma = 1.524 * au, 0.0934
r_ma       = np.array([ a_ma*(1-e_ma), 0.0 ])
v_ma       = np.array([ 0.0, np.sqrt(mu*(1+e_ma)/(a_ma*(1-e_ma))) ])

# Jupiter
a_j, e_j = 5.203 * au, 0.0489
r_j      = np.array([ a_j*(1-e_j), 0.0 ])
v_j      = np.array([ 0.0, np.sqrt(mu*(1+e_j)/(a_j*(1-e_j))) ])

# Saturn
a_s, e_s = 9.537 * au, 0.0541
r_s      = np.array([ a_s*(1-e_s), 0.0 ])
v_s      = np.array([ 0.0, np.sqrt(mu*(1+e_s)/(a_s*(1-e_s))) ])

# Uranus
a_u, e_u = 19.191 * au, 0.0472
r_u      = np.array([ a_u*(1-e_u), 0.0 ])
v_u      = np.array([ 0.0, np.sqrt(mu*(1+e_u)/(a_u*(1-e_u))) ])

# Neptune
a_n, e_n = 30.069 * au, 0.0086
r_n      = np.array([ a_n*(1-e_n), 0.0 ])
v_n      = np.array([ 0.0, np.sqrt(mu*(1+e_n)/(a_n*(1-e_n))) ])

# --- Подготовка списков траекторий и переменных для детекции периодов ---
traj_p = []
traj_e = []
traj_v = []
traj_m = []
traj_ma = []
traj_j = []
traj_s = []
traj_u = []
traj_n = []

y_prev_p  = r_p[1]
y_prev_e  = r_e[1]
y_prev_v  = r_v[1]
y_prev_m  = r_m[1]
y_prev_ma = r_ma[1]
y_prev_j  = r_j[1]
y_prev_s  = r_s[1]
y_prev_u  = r_u[1]
y_prev_n  = r_n[1]

period_e  = None
period_v  = None
period_m  = None
period_ma = None
period_j  = None
period_s  = None
period_u  = None
period_n  = None

# --- Интегрирование методом Verlet ---
for i in range(max_days):
    # Pluto
    a_vec = -mu * r_p / np.linalg.norm(r_p)**3
    v_half = v_p + 0.5 * a_vec * dt
    r_p     = r_p    + v_half * dt
    a_vec   = -mu * r_p / np.linalg.norm(r_p)**3
    v_p     = v_half + 0.5 * a_vec * dt
    traj_p.append(r_p.copy())

    # Earth
    a_vec = -mu * r_e / np.linalg.norm(r_e)**3
    v_half = v_e + 0.5 * a_vec * dt
    r_e     = r_e    + v_half * dt
    a_vec   = -mu * r_e / np.linalg.norm(r_e)**3
    v_e     = v_half + 0.5 * a_vec * dt
    traj_e.append(r_e.copy())
    if period_e is None and y_prev_e < 0 and r_e[1]>=0 and v_e[1]>0 and r_e[0]>0:
        period_e = i+1
    y_prev_e = r_e[1]

    # Venus
    a_vec = -mu * r_v / np.linalg.norm(r_v)**3
    v_half = v_v + 0.5 * a_vec * dt
    r_v     = r_v    + v_half * dt
    a_vec   = -mu * r_v / np.linalg.norm(r_v)**3
    v_v     = v_half + 0.5 * a_vec * dt
    traj_v.append(r_v.copy())
    if period_v is None and y_prev_v < 0 and r_v[1]>=0 and v_v[1]>0 and r_v[0]>0:
        period_v = i+1
    y_prev_v = r_v[1]

    # Mercury
    a_vec = -mu * r_m / np.linalg.norm(r_m)**3
    v_half = v_m + 0.5 * a_vec * dt
    r_m     = r_m    + v_half * dt
    a_vec   = -mu * r_m / np.linalg.norm(r_m)**3
    v_m     = v_half + 0.5 * a_vec * dt
    traj_m.append(r_m.copy())
    if period_m is None and y_prev_m < 0 and r_m[1]>=0 and v_m[1]>0 and r_m[0]>0:
        period_m = i+1
    y_prev_m = r_m[1]

    # Mars
    a_vec = -mu * r_ma / np.linalg.norm(r_ma)**3
    v_half = v_ma + 0.5 * a_vec * dt
    r_ma    = r_ma   + v_half * dt
    a_vec   = -mu * r_ma / np.linalg.norm(r_ma)**3
    v_ma    = v_half + 0.5 * a_vec * dt
    traj_ma.append(r_ma.copy())
    if period_ma is None and y_prev_ma < 0 and r_ma[1]>=0 and v_ma[1]>0 and r_ma[0]>0:
        period_ma = i+1
    y_prev_ma = r_ma[1]

    # Jupiter
    a_vec = -mu * r_j / np.linalg.norm(r_j)**3
    v_half = v_j + 0.5 * a_vec * dt
    r_j     = r_j    + v_half * dt
    a_vec   = -mu * r_j / np.linalg.norm(r_j)**3
    v_j     = v_half + 0.5 * a_vec * dt
    traj_j.append(r_j.copy())
    if period_j is None and y_prev_j < 0 and r_j[1]>=0 and v_j[1]>0 and r_j[0]>0:
        period_j = i+1
    y_prev_j = r_j[1]

    # Saturn
    a_vec = -mu * r_s / np.linalg.norm(r_s)**3
    v_half = v_s + 0.5 * a_vec * dt
    r_s     = r_s    + v_half * dt
    a_vec   = -mu * r_s / np.linalg.norm(r_s)**3
    v_s     = v_half + 0.5 * a_vec * dt
    traj_s.append(r_s.copy())
    if period_s is None and y_prev_s < 0 and r_s[1]>=0 and v_s[1]>0 and r_s[0]>0:
        period_s = i+1
    y_prev_s = r_s[1]

    # Uranus
    a_vec = -mu * r_u / np.linalg.norm(r_u)**3
    v_half = v_u + 0.5 * a_vec * dt
    r_u     = r_u    + v_half * dt
    a_vec   = -mu * r_u / np.linalg.norm(r_u)**3
    v_u     = v_half + 0.5 * a_vec * dt
    traj_u.append(r_u.copy())
    if period_u is None and y_prev_u < 0 and r_u[1]>=0 and v_u[1]>0 and r_u[0]>0:
        period_u = i+1
    y_prev_u = r_u[1]

    # Neptune
    a_vec = -mu * r_n / np.linalg.norm(r_n)**3
    v_half = v_n + 0.5 * a_vec * dt
    r_n     = r_n    + v_half * dt
    a_vec   = -mu * r_n / np.linalg.norm(r_n)**3
    v_n     = v_half + 0.5 * a_vec * dt
    traj_n.append(r_n.copy())
    if period_n is None and y_prev_n < 0 and r_n[1]>=0 and v_n[1]>0 and r_n[0]>0:
        period_n = i+1
    y_prev_n = r_n[1]

    if y_prev_p < 0 and r_p[1]>=0 and v_p[1]>0 and r_p[0]>0:
        traj_p  = traj_p[:i+1]
        traj_e  = traj_e[:i+1]
        traj_v  = traj_v[:i+1]
        traj_m  = traj_m[:i+1]
        traj_ma = traj_ma[:i+1]
        traj_j  = traj_j[:i+1]
        traj_s  = traj_s[:i+1]
        traj_u  = traj_u[:i+1]
        traj_n  = traj_n[:i+1]
        break
    y_prev_p = r_p[1]

# --- Конвертация в numpy ---
traj_p  = np.array(traj_p)
traj_e  = np.array(traj_e)
traj_v  = np.array(traj_v)
traj_m  = np.array(traj_m)
traj_ma = np.array(traj_ma)
traj_j  = np.array(traj_j)
traj_s  = np.array(traj_s)
traj_u  = np.array(traj_u)
traj_n  = np.array(traj_n)

# --- Статичные орбиты (один полный оборот каждой) ---
static_p  = traj_p
static_e  = traj_e[:period_e]
static_v  = traj_v[:period_v]
static_m  = traj_m[:period_m]
static_ma = traj_ma[:period_ma]
static_j  = traj_j[:period_j]
static_s  = traj_s[:period_s]
static_u  = traj_u[:period_u]
static_n  = traj_n[:period_n]

# --- Вывод периодов ---
def print_period(name, steps):
    days = steps * dt / (24*3600)
    years = days / 365.25
    print(f"{name} orbital period: {days:.2f} days ---- {years:.2f} Earth years")
    print("_________________________________________________________")

# --------------- ПОСЛЕДНЕЕ ЗАДАНИЕ ---------------
distances_p = np.linalg.norm(traj_p, axis=1)
r_max_p = np.max(distances_p)
r_min_p = np.min(distances_p)
epsilon_computed = (r_max_p - r_min_p) / (r_max_p + r_min_p)
print(f"Tuta ia: {epsilon_computed:.4f}")
print("nauka: 0.246")
# --------------- ПОСЛЕДНЕЕ ЗАДАНИЕ ---------------

print_period("Pluto", len(traj_p))
print_period("Earth", period_e)
print_period("Venus", period_v)
print_period("Mercury", period_m)
print_period("Mars", period_ma)
print_period("Jupiter", period_j)
print_period("Saturn", period_s)
print_period("Uranus", period_u)
print_period("Neptune", period_n)

# --- Рисуем орбиты ---
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(static_p[:,0]/au,  static_p[:,1]/au,  lw=0.5, label='Pluto')
ax.plot(static_e[:,0]/au,  static_e[:,1]/au,  lw=0.5, label='Earth')
ax.plot(static_v[:,0]/au,  static_v[:,1]/au,  lw=0.5, label='Venus')
ax.plot(static_m[:,0]/au,  static_m[:,1]/au,  lw=0.5, label='Mercury')
ax.plot(static_ma[:,0]/au, static_ma[:,1]/au, lw=0.5, label='Mars')
ax.plot(static_j[:,0]/au,  static_j[:,1]/au,  lw=0.5, label='Jupiter')
ax.plot(static_s[:,0]/au,  static_s[:,1]/au,  lw=0.5, label='Saturn')
ax.plot(static_u[:,0]/au,  static_u[:,1]/au,  lw=0.5, label='Uranus')
ax.plot(static_n[:,0]/au,  static_n[:,1]/au,  lw=0.5, label='Neptune')
ax.scatter(0, 0, color='orange', s=100, label='Sun')
ax.set_title("Orbits of Pluto, Earth, Venus, Mercury, Mars, Jupiter, Saturn, Uranus and Neptune")
ax.set_xlabel("x [AU]")
ax.set_ylabel("y [AU]")
ax.axis('equal')

# --- Точки-планеты ---
dots = {
    "Pluto":   dict(marker='o', color='grey',       ms=8),
    "Earth":   dict(marker='o', color='blue',        ms=6),
    "Venus":   dict(marker='o', color='yellow',      ms=6),
    "Mercury": dict(marker='o', color='grey',        ms=5),
    "Mars":    dict(marker='o', color='red',         ms=6),
    "Jupiter": dict(marker='o', color='saddlebrown', ms=7),
    "Saturn":  dict(marker='o', color='gold',        ms=7),
    "Uranus":  dict(marker='o', color='cyan',        ms=7),
    "Neptune": dict(marker='o', color='darkblue',    ms=7),
}
pluto_dot   = ax.plot([], [], **dots["Pluto"],   label="Pluto")[0]
earth_dot   = ax.plot([], [], **dots["Earth"],   label="Earth")[0]
venus_dot   = ax.plot([], [], **dots["Venus"],   label="Venus")[0]
mercury_dot = ax.plot([], [], **dots["Mercury"], label="Mercury")[0]
mars_dot    = ax.plot([], [], **dots["Mars"],    label="Mars")[0]
jupiter_dot = ax.plot([], [], **dots["Jupiter"], label="Jupiter")[0]
saturn_dot  = ax.plot([], [], **dots["Saturn"],  label="Saturn")[0]
uranus_dot  = ax.plot([], [], **dots["Uranus"],  label="Uranus")[0]
neptune_dot = ax.plot([], [], **dots["Neptune"], label="Neptune")[0]
ax.legend(loc='upper right')

def init():
    for dot, traj in zip(
        (pluto_dot, earth_dot, venus_dot, mercury_dot, mars_dot, jupiter_dot, saturn_dot, uranus_dot, neptune_dot),
        (traj_p,    traj_e,     traj_v,     traj_m,      traj_ma,      traj_j,       traj_s,       traj_u,       traj_n)
    ):
        x0, y0 = traj[0] / au
        dot.set_data([x0], [y0])
    return pluto_dot, earth_dot, venus_dot, mercury_dot, mars_dot, jupiter_dot, saturn_dot, uranus_dot, neptune_dot

def update(frame):
    pts = []
    for dot, traj in zip(
        (pluto_dot, earth_dot, venus_dot, mercury_dot, mars_dot, jupiter_dot, saturn_dot, uranus_dot, neptune_dot),
        (traj_p,    traj_e,     traj_v,     traj_m,      traj_ma,      traj_j,       traj_s,       traj_u,       traj_n)
    ):
        x, y = traj[frame] / au
        dot.set_data([x], [y])
        pts.append(dot)
    return pts

ani = FuncAnimation(fig, update,
                    frames=len(traj_p),
                    init_func=init,
                    interval=30,   
                    blit=True)

plt.show()