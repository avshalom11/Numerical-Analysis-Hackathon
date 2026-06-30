# =====================================================================
# Stage A - Neville vs Natural Cubic Spline
# =====================================================================

import matplotlib.pyplot as plt

def solve_linear_system(matrix, vector):
    n = len(matrix)

    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(matrix[k][i]) > abs(matrix[max_row][i]):
                max_row = k

        matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
        vector[i], vector[max_row] = vector[max_row], vector[i]

        if matrix[i][i] == 0:
            print("Cannot solve system: zero pivot.")
            return None

        for k in range(i + 1, n):
            factor = matrix[k][i] / matrix[i][i]
            for j in range(i, n):
                matrix[k][j] -= factor * matrix[i][j]
            vector[k] -= factor * vector[i]

    solution = [0.0] * n

    for i in range(n - 1, -1, -1):
        total = 0.0
        for j in range(i + 1, n):
            total += matrix[i][j] * solution[j]
        solution[i] = (vector[i] - total) / matrix[i][i]

    return solution


def neville_interpolation(x_points, y_points, x_target):
    n = len(x_points)
    Q = [[0.0] * n for _ in range(n)]

    for i in range(n):
        Q[i][0] = float(y_points[i])

    for j in range(1, n):
        for i in range(n - j):
            numerator1 = (x_target - x_points[i + j]) * Q[i][j - 1]
            numerator2 = (x_target - x_points[i]) * Q[i + 1][j - 1]
            denominator = x_points[i] - x_points[i + j]
            Q[i][j] = (numerator1 - numerator2) / denominator

    return Q[0][n - 1]


def cubic_spline_interpolation(x_points, y_points, x_target):
    if x_target in x_points:
        return y_points[x_points.index(x_target)]

    n = len(x_points)

    h = []
    for i in range(n - 1):
        h.append(x_points[i + 1] - x_points[i])

    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    vector = [0.0 for _ in range(n)]

    # Natural Cubic Spline conditions
    matrix[0][0] = 1.0
    matrix[n - 1][n - 1] = 1.0

    for i in range(1, n - 1):
        matrix[i][i - 1] = h[i - 1]
        matrix[i][i] = 2 * (h[i - 1] + h[i])
        matrix[i][i + 1] = h[i]

        vector[i] = 6 * (
            ((y_points[i + 1] - y_points[i]) / h[i])
            - ((y_points[i] - y_points[i - 1]) / h[i - 1])
        )

    second_derivatives = solve_linear_system(matrix, vector)

    interval_index = None
    for i in range(n - 1):
        if x_points[i] <= x_target <= x_points[i + 1]:
            interval_index = i
            break

    i = interval_index

    x_i = x_points[i]
    x_next = x_points[i + 1]
    y_i = y_points[i]
    y_next = y_points[i + 1]
    h_i = x_next - x_i

    m_i = second_derivatives[i]
    m_next = second_derivatives[i + 1]

    term1 = m_i * ((x_next - x_target) ** 3) / (6 * h_i)
    term2 = m_next * ((x_target - x_i) ** 3) / (6 * h_i)
    term3 = (y_i - (m_i * h_i ** 2) / 6) * ((x_next - x_target) / h_i)
    term4 = (y_next - (m_next * h_i ** 2) / 6) * ((x_target - x_i) / h_i)

    return term1 + term2 + term3 + term4


def print_spline_coefficients(x_points, y_points):
    n = len(x_points)

    h = []
    for i in range(n - 1):
        h.append(x_points[i + 1] - x_points[i])

    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    vector = [0.0 for _ in range(n)]

    matrix[0][0] = 1.0
    matrix[n - 1][n - 1] = 1.0

    for i in range(1, n - 1):
        matrix[i][i - 1] = h[i - 1]
        matrix[i][i] = 2 * (h[i - 1] + h[i])
        matrix[i][i + 1] = h[i]

        vector[i] = 6 * (
            ((y_points[i + 1] - y_points[i]) / h[i])
            - ((y_points[i] - y_points[i - 1]) / h[i - 1])
        )

    second_derivatives = solve_linear_system(matrix, vector)

    print("\nSpline coefficients for each interval:")
    print("Formula: S_i(x) = a + b(x-x_i) + c(x-x_i)^2 + d(x-x_i)^3")
    print("Interval\t a\t\t b\t\t c\t\t d")

    for i in range(n - 1):
        a = y_points[i]
        b = ((y_points[i + 1] - y_points[i]) / h[i]) - (
            h[i] * (2 * second_derivatives[i] + second_derivatives[i + 1]) / 6
        )
        c = second_derivatives[i] / 2
        d = (second_derivatives[i + 1] - second_derivatives[i]) / (6 * h[i])

        print(f"[{x_points[i]}, {x_points[i+1]}]\t {a:.6f}\t {b:.6f}\t {c:.6f}\t {d:.6f}")


def main():
    x_points = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
    y_points = [0.950, 1.166, 0.264, -0.317, -0.156, -0.549, 0.074, 1.076]

    # 200 equally spaced points in [0, 3]
    x_test = []
    step = 3.0 / 199.0

    for i in range(200):
        x_test.append(i * step)

    neville_values = []
    spline_values = []
    delta_values = []

    for x in x_test:
        n_val = neville_interpolation(x_points, y_points, x)
        s_val = cubic_spline_interpolation(x_points, y_points, x)

        neville_values.append(n_val)
        spline_values.append(s_val)
        delta_values.append(abs(n_val - s_val))

    R = max(y_points) - min(y_points)
    max_delta = max(delta_values)
    max_index = delta_values.index(max_delta)
    x_max_delta = x_test[max_index]
    ratio = max_delta / R

    print("===== Stage A - Neville vs Natural Cubic Spline =====")
    print("R =", R)
    print("max_delta =", max_delta)
    print("x at max_delta =", x_max_delta)
    print("max_delta / R =", ratio)

    if ratio < 0.01:
        print("Conclusion: Full agreement. Continue to Stage B.")
    elif ratio < 0.05:
        print("Conclusion: Partial agreement. Explain and continue.")
    else:
        print("Conclusion: Difference above 5%. Stop, analyze and fix/explain.")

    print("\nNeville example at x = 1.5:")
    print("Neville(1.5) =", neville_interpolation(x_points, y_points, 1.5))

    print("\nTable A")
    print("x\tNeville\t\tSpline\t\tDelta")

    table_points = [0.5, 1.0, 1.5, 2.0, 2.5]

    for x in table_points:
        n_val = neville_interpolation(x_points, y_points, x)
        s_val = cubic_spline_interpolation(x_points, y_points, x)
        delta = abs(n_val - s_val)

        print(f"{x}\t{n_val}\t{s_val}\t{delta}")

    print_spline_coefficients(x_points, y_points)

    # Graph 1: Neville vs Natural Cubic Spline
    plt.figure()
    plt.plot(x_test, neville_values, label="Neville")
    plt.plot(x_test, spline_values, label="Natural Cubic Spline")
    plt.scatter(x_points, y_points, label="Data Points")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Neville vs Natural Cubic Spline")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Graph 2: Delta
    plt.figure()
    plt.plot(x_test, delta_values, label="Delta")
    plt.scatter([x_max_delta], [max_delta], label="max_delta")
    plt.xlabel("x")
    plt.ylabel("|Neville - Spline|")
    plt.title("Delta between Neville and Spline")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()