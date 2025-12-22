import matplotlib.pyplot as plt

# Данные из таблицы
I = [0, 4, 8, 11.997, 16, 20, 24.009, 28.012, 32, 35.971, 40]   # ток Ik, мА
U = [20, 18, 16, 14.001, 12, 10, 7.995, 5.9945, 4, 2.0144, 0]   # напряжение Uk, В
P = [0, 0.072, 0.128, 0.16797, 0.192, 0.2, 0.191952, 0.167916, 0.128, 0.072461, 0]  # мощность, Вт
eta = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.101, 0]       # КПД (η)

# --- 1. Внешняя характеристика источника ---
plt.figure(figsize=(6,4))
plt.plot(I, U, 'o-', label="U(I)", color="blue")
for x, y in zip(I, U):
    plt.annotate(f"{x:.5g}; {y:.5g}", (x, y), textcoords="offset points", xytext=(5,5), fontsize=8)
plt.xlabel("I, мА")
plt.ylabel("U, В")
plt.title("Внешняя характеристика источника")
plt.grid(True)
plt.legend()
plt.show()

# --- 2. Рабочие характеристики ---
plt.figure(figsize=(6,4))
plt.plot(I, P, 'o-', label="P(I)", color="blue")
plt.plot(I, eta, 's-', label="η", color="orange")

# подписи для P(I)
for x, y in zip(I, P):
    plt.annotate(f"{y:.3g}", (x, y), textcoords="offset points", xytext=(5,5), fontsize=8, color="blue")

# подписи для η(I)
for x, y in zip(I, eta):
    plt.annotate(f"{y:.2g}", (x, y), textcoords="offset points", xytext=(5,-10), fontsize=8, color="orange")

plt.xlabel("I, мА")
plt.ylabel("P, Вт / η")
plt.title("Рабочие характеристики источника")
plt.grid(True)
plt.legend()
plt.show()
