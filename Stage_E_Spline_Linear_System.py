# =====================================================================
# Stage E - Spline Linear System: LU vs Jacobi
# =====================================================================

def copy_matrix(matrix):
    return [row[:] for row in matrix]


def build_spline_system(x_points, y_points):
    n = len(x_points)
    h = []

    for i in range(n - 1):
        h.append(x_points[i + 1] - x_points[i])

    A = [[0.0 for _ in range(n)] for _ in range(n)]
    b = [0.0 for _ in range(n)]

    # Natural Spline boundary conditions
    A[0][0] = 1.0
    A[n - 1][n - 1] = 1.0

    for i in range(1, n - 1):
        A[i][i - 1] = h[i - 1]
        A[i][i] = 2 * (h[i - 1] + h[i])
        A[i][i + 1] = h[i]

        b[i] = 6 * (
            ((y_points[i + 1] - y_points[i]) / h[i])
            - ((y_points[i] - y_points[i - 1]) / h[i - 1])
        )

    return A, b


def gaussian_elimination(A, b):
    A = copy_matrix(A)
    b = b[:]
    n = len(A)

    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > abs(A[max_row][i]):
                max_row = k

        A[i], A[max_row] = A[max_row], A[i]
        b[i], b[max_row] = b[max_row], b[i]

        for k in range(i + 1, n):
            factor = A[k][i] / A[i][i]
            for j in range(i, n):
                A[k][j] -= factor * A[i][j]
            b[k] -= factor * b[i]

    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        total = 0.0
        for j in range(i + 1, n):
            total += A[i][j] * x[j]

        x[i] = (b[i] - total) / A[i][i]

    return x


def lu_decomposition(A, b):
    n = len(A)

    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0

    # Doolittle LU: A = L * U
    for i in range(n):
        for j in range(i, n):
            total = 0.0
            for k in range(i):
                total += L[i][k] * U[k][j]
            U[i][j] = A[i][j] - total

        for j in range(i + 1, n):
            total = 0.0
            for k in range(i):
                total += L[j][k] * U[k][i]
            L[j][i] = (A[j][i] - total) / U[i][i]

    # Forward substitution: L*y = b
    y = [0.0] * n

    for i in range(n):
        total = 0.0
        for j in range(i):
            total += L[i][j] * y[j]
        y[i] = b[i] - total

    # Back substitution: U*x = y
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        total = 0.0
        for j in range(i + 1, n):
            total += U[i][j] * x[j]
        x[i] = (y[i] - total) / U[i][i]

    return x, L, U


def max_difference(v1, v2):
    max_diff = 0.0

    for i in range(len(v1)):
        diff = abs(v1[i] - v2[i])
        if diff > max_diff:
            max_diff = diff

    return max_diff


def jacobi(A, b, epsilon=0.00001, max_iterations=1000):
    n = len(A)

    previous = [0.0] * n
    current = [0.0] * n

    for iteration in range(1, max_iterations + 1):
        current = [0.0] * n

        for i in range(n):
            row_sum = 0.0

            for j in range(n):
                if i != j:
                    row_sum += A[i][j] * previous[j]

            current[i] = (b[i] - row_sum) / A[i][i]

        if max_difference(current, previous) <= epsilon:
            return current, iteration, True

        previous = current[:]

    return current, max_iterations, False


def print_matrix(title, matrix):
    print("\n" + title)
    for row in matrix:
        print(["{:.6f}".format(value) for value in row])


def print_vector(title, vector):
    print("\n" + title)
    print(["{:.10f}".format(value) for value in vector])


def main():
    x_points = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
    y_points = [0.950, 1.166, 0.264, -0.317, -0.156, -0.549, 0.074, 1.076]

    A, b = build_spline_system(x_points, y_points)

    print("===== Stage E - Internal Linear System of Natural Spline =====")

    print_matrix("Matrix A:", A)
    print_vector("Vector b:", b)

    # Original solution from Stage A
    original_solution = gaussian_elimination(A, b)
    print_vector("Original solution from Stage A - Gaussian Elimination:", original_solution)

    # Method 1: LU
    lu_solution, L, U = lu_decomposition(A, b)
    print_matrix("L matrix:", L)
    print_matrix("U matrix:", U)
    print_vector("Solution by LU Decomposition:", lu_solution)

    # Method 2: Jacobi
    jacobi_solution, iterations, converged = jacobi(A, b)
    print_vector("Solution by Jacobi:", jacobi_solution)
    print("\nJacobi iterations:", iterations)
    print("Jacobi converged:", converged)

    print("\nComparison:")
    print("Index\tOriginal(Gauss)\t\tLU\t\t\tJacobi")

    for i in range(len(original_solution)):
        print(
            f"M[{i}]\t"
            f"{original_solution[i]:.10f}\t"
            f"{lu_solution[i]:.10f}\t"
            f"{jacobi_solution[i]:.10f}"
        )

    diff_lu = max_difference(original_solution, lu_solution)
    diff_jacobi = max_difference(original_solution, jacobi_solution)

    print("\nMax difference Original vs LU:", diff_lu)
    print("Max difference Original vs Jacobi:", diff_jacobi)

    if diff_lu < 0.01 and diff_jacobi < 0.01:
        print("Conclusion: LU and Jacobi agree with the original Spline solution.")
    else:
        print("Conclusion: There is a significant difference and further analysis is required.")


if __name__ == "__main__":
    main()