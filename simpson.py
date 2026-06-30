import numpy as np


# 1. פונקציית אינטרפולציית נוויל המקורית שלך
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


# 2. שיטת סמפסון המורכבת המקורית שלך
def simpsons_13_composite(f=None, a=None, b=None, n=None, x=None, y=None):
    if f is not None and a is not None and b is not None and n is not None:
        if n % 2 != 0:
            raise ValueError(f"שיטת סמפסון 1/3 מחייבת מספר מקטעים זוגי!")
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = np.array([f(val) for val in x])
    else:
        raise ValueError("ממשק שגוי")
    sum_odd = np.sum(y[1:-1:2])
    sum_even = np.sum(y[2:-1:2])
    integral = (h / 3) * (y[0] + y[-1] + 4 * sum_odd + 2 * sum_even)
    return integral


# 3. שיטת רומברג המקורית שלך
def romberg_integration(f, a, b, K=10, tol=1e-12):
    if a == b:
        return 0.0, np.array([[0.0]])
    R = np.zeros((K, K))
    h = b - a
    R[0, 0] = 0.5 * h * (f(a) + f(b))
    for k in range(1, K):
        h /= 2.0
        num_new_points = 2 ** (k - 1)
        sum_new_fx = sum(f(a + (2 * i + 1) * h) for i in range(num_new_points))
        R[k, 0] = 0.5 * R[k - 1, 0] + h * sum_new_fx
        for j in range(1, k + 1):
            R[k, j] = R[k, j - 1] + (R[k, j - 1] - R[k - 1, j - 1]) / (4 ** j - 1)
        if abs(R[k, k] - R[k - 1, k - 1]) < tol:
            return R[k, k], R[:k + 1, :k + 1]
    return R[K - 1, K - 1], R


def main():
    # נתוני הטבלה המעודכנים שלך
    x_points = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
    y_points = [0.950, 1.166, 0.264, -0.317, -0.156, -0.549, 0.074, 1.076]

    # הגדרת פונקציית המטרה f(x) המבוססת על נוויל
    f_target = lambda x: neville_interpolation(x_points, y_points, x)

    a, b = 0.0, 3.0

    print("==================================================")
    print("Executing Step C: Integration for Neville Function")
    print("==================================================")

    # חישוב באמצעות סמפסון n=100
    I_simpson = simpsons_13_composite(f=f_target, a=a, b=b, n=100)
    print(f"Simpson's 1/3 Result (n=100): {I_simpson:.8f}")
    print("--------------------------------------------------")

    # חישוב באמצעות רומברג
    I_romberg, table = romberg_integration(f_target, a, b, K=6, tol=1e-8)

    print("Romberg Table (Table C):")
    for idx, row in enumerate(table):
        valid_elements = row[:idx + 1]
        formatted_row = "  |  ".join([f"{x:.8f}" for x in valid_elements])
        print(f"R[{idx}]: {formatted_row}")
    print("--------------------------------------------------")

    # חישוב הפרש יחסי והסכמה
    rel_diff = abs(I_simpson - I_romberg) / abs(I_romberg)
    print(f"Relative Difference: {rel_diff:.6f} ({rel_diff * 100:.4f}%)")
    if rel_diff < 0.01:
        print("Result: Agreement Achieved (< 1%)")
    else:
        print("Result: No Agreement (>= 1%)")
    print("==================================================")


if __name__ == "__main__":
    main()