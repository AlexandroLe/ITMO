import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define the grid for plotting
x1 = np.linspace(0, 20, 500)
x2 = np.linspace(0, 15, 500)
X1, X2 = np.meshgrid(x1, x2)

# Define the constraints
c1 = X2 >= 3
c2 = 3 * X1 + 2.3 * X2 >= 27
c3 = X1 + 2.5 * X2 >= 20
feasible = c1 & c2 & c3

# Compute intersection of the two lines for base level
A = np.array([[3, 2.3], [1, 2.5]])
b = np.array([27, 20])
x_int = np.linalg.solve(A, b)
base_level = 0.1 * x_int[0] + 0.195 * x_int[1]

# Define a slightly higher level
higher_level = base_level + 0.2

# Plotting
plt.figure(figsize=(8, 6))
plt.contourf(X1, X2, feasible, levels=[-0.5, 0.5, 1.5], colors=['#e0f7fa', '#80deea'], alpha=0.7)

# Plot constraint lines
plt.plot(x1, np.full_like(x1, 3), label=r'$x_2 = 3$')
plt.plot(x1, (27 - 3 * x1) / 2.3, label=r'$3x_1 + 2.3x_2 = 27$')
plt.plot(x1, (20 - x1) / 2.5, label=r'$x_1 + 2.5x_2 = 20$')

# Plot level lines
plt.contour(X1, X2, 0.1 * X1 + 0.195 * X2, levels=[base_level], linestyles='--', colors='black')
plt.contour(X1, X2, 0.1 * X1 + 0.195 * X2, levels=[higher_level], linestyles='-.', colors='black')

# Dummy legend entries for level lines
line1 = Line2D([0], [0], color='black', linestyle='--',
               label=f'Level: {base_level:.2f}')
line2 = Line2D([0], [0], color='black', linestyle='-.',
               label=f'Higher level: {higher_level:.2f}')

# Plot magnified gradient
grad = np.array([0.1, 0.195])
plt.quiver(0, 0, *(grad * 60 ), scale=1, scale_units='xy', angles='xy',
           color='red')

# Final settings
plt.xlim(0, 20)
plt.ylim(0, 15)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Feasible Region with Two Level Lines')
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=[line1, line2] + handles, loc='upper right')
plt.grid(True)
plt.show()
