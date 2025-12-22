import numpy as np
import matplotlib.pyplot as plt

# Гравитационная постоянная и масса Солнца
G      = 6.67430e-11       # м^3⋅кг^–1⋅c^–2
M_sun  = 1.98847e30        # кг
mu     = G * M_sun

# Перевод AU в метры
au = 1.495978707e11         # м

# Параметры орбиты Плутона
a = 39.48 * au              # полуось, м
e = 0.2488                  # эксцентриситет

# Начальные условия в перигелии
r0 = a * (1 - e)
v0 = np.sqrt(mu * (1 + e) / (a * (1 - e)))

# Векторы состояния
r = np.array([r0, 0.0])     # позиция, м
v = np.array([0.0, v0])     # скорость, м/с

# Параметры интегрирования
dt = 24 * 3600.0            # шаг 1 день в секундах
simulation_years = 260      # моделируем 260 лет
max_steps = int(simulation_years * 365.25 * 24 * 3600 / dt)

positions = []
t = 0.0
y_prev = r[1]
period = None

# Эйлеров интегратор
for i in range(max_steps):
    r_norm = np.linalg.norm(r)
    a_vec  = -mu * r / r_norm**3

    v += a_vec * dt
    r += v * dt

    t += dt
    positions.append(r.copy())

    # Детекция прохождения через x>0 с y переходящим снизу вверх
    y  = r[1]
    vy = v[1]
    if y_prev < 0 and y >= 0 and vy > 0 and r[0] > 0:
        period = t
        break
    y_prev = y

if period is not None:
    period_years = period / (365.25 * 24 * 3600)
    print(f"Найден период: {period_years:.2f} лет")
else:
    print("Не удалось отследить полный оборот за отведённое время.")

# Переводим результат в массив для графика
positions = np.array(positions)

# Рисуем орбиту
plt.figure(figsize=(6,6))
plt.plot(positions[:,0]/au, positions[:,1]/au, label='Путь Плутона')
plt.scatter([0], [0], color='orange', s=100, label='Солнце')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.axis('equal')
plt.title('Орбита Плутона вокруг Солнца')
plt.legend()
plt.show()
