from sympy import *

# Define the symbols
fx, fy, cx, cy = symbols('fx fy cx cy')
X, Y, Z = symbols('X Y Z')
ui, vi = symbols('u_i v_i')

Xi = Matrix([X, Y, Z])
K = Matrix([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
R = Matrix([[symbols(f'r{i}{j}') for j in range(1, 3+1)] for i in range(1, 3+1)])
t = Matrix([symbols(f't{i}') for i in range(1, 3+1)])

A = Matrix([[symbols(f'a{i}{j}') for j in range(1, 9+1)] for i in range(1, 2+1)])
m = Matrix([R.row(i) for i in range(R.rows)]).reshape(9, 1)
b = Matrix([symbols(f'b{i}') for i in range(1, 2+1)])

ui_tilde = K * R * Xi + K * t

eq1 = Eq(ui, ui_tilde[0] / ui_tilde[2])
eq2 = Eq(vi, ui_tilde[1] / ui_tilde[2])
eq3 = Eq(A * m, b)
solution = solve((eq1, eq2, eq3), (A, b))

print(solution)
