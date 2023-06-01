import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import linprog

f = lambda x: (x-1)**4 + x**2
f_prime = lambda x: 4*(x-1)**3 + 2*x

A = np.array([[2,1],[-4,5],[1,-2]])
b = np.array([10,8,3])
c = np.array([-1,-2])

def print_a_function(f, values):
    y = f(values)
    plt.plot(x, y)
    plt.show()

def find_root_bisection(f, a, b):
    if f(a)*f(b)>0:
        return abs(a - b) - 0.1
    root = (a + b)/2
    while abs(f(root)) > 0.001:
        if f(a) * f(root) > 0:
            a = root
        else:
            b = root
        root = (a + b)/2
    return root

def find_root_newton_raphson(f, f_prime, a):
    while abs(f(a)) > 0.001:
        a = a - f(a)/f_prime(a)
        print(a)
    return a

res = minimize_scalar(f, method='brent')
print('%s: %.02f, %s: %.02f' % ('x_min', res.x, 'f(x_min)', res.fun))

# plot curve
x = np.linspace(res.x - 1, res.x + 1, 100)
y = [f(val) for val in x]
plt.plot(x, y, color='blue', label='f')

# plot optima
plt.scatter(res.x, res.fun, color='red', marker='x', label='Minimum')

plt.grid()
plt.legend(loc = 1)
plt.show()

def gradient_descent(f, f_prime, start, learning_rate = 0.1):
    old = start
    new = old - learning_rate * f_prime(old)
    while abs(old-new) > 0.0001:
        old = new
        new = new - learning_rate * f_prime(new)
    return new 

f = lambda x : (x - 1) ** 4 + x ** 2
f_prime = lambda x : 4*((x-1)**3) + 2*x
start = -1
x_min = gradient_descent(f, f_prime, start, 0.1)
f_min = f(x_min)
# print(f"xmin: {round(x_min, 2)}, f(x_min): {round(f_min, 2)}")

def solve_linear_problem(A, b, c):
    result = linprog(c, A, b)
    return (round(result.fun), result.x)
    
optimal_value, optimal_arg = solve_linear_problem(A, b, c)

print("The optimal value is: ", optimal_value, " and is reached for x = ", optimal_arg)
