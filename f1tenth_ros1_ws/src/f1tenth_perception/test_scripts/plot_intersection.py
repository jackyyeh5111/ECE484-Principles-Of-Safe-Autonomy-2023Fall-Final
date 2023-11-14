import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

# Define the symbols
x, y = sp.symbols('x y')

# Equation of the circle
circle_eq = sp.Eq((x - 2)**2 + (y - 2)**2, 4)

# Equation of the curve (replace this with your specific curve equation)
curve_eq = sp.Eq(y, x**2)
curve_eq = sp.Eq(y**(1/2), x)

# Find the intersection points
intersection_points = sp.solve((circle_eq, curve_eq), (x, y))

# Plot the circle and the curve
theta = np.linspace(0, 2*np.pi, 100)
circle_x = 2 + 2 * np.cos(theta)
circle_y = 2 + 2 * np.sin(theta)

curve_x = np.linspace(-3, 3, 100)
curve_y = curve_x**2

plt.plot(circle_x, circle_y, label='Circle: $(x - 2)^2 + (y - 2)^2 = 4$')
plt.plot(curve_x, curve_y, label='Curve: $y = x^2$')
# plt.scatter(*zip(*intersection_points), color='red', label='Intersection Points')

# Plot only the real part of the intersection points
real_intersection_points = [(sp.re(point[0]), sp.re(
    point[1])) for point in intersection_points if sp.im(point[0]) == 0 and sp.im(point[1]) == 0]
plt.scatter(*zip(*real_intersection_points),
            color='red', label='Intersection Points')
# Display the intersection points
print("num of Intersection Points:", len(real_intersection_points))
print("Intersection Points:", real_intersection_points)

# plt.plot(curve_x, curve_y, label='Curve: $y = x^2$')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Intersection of Circle and Curve')
plt.grid(True)
plt.show()
