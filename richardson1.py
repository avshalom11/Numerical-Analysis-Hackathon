import math
import matplotlib.pyplot as plt


def neville_interpolation(x_points, y_points, x_target):
    """
    Evaluates the interpolating polynomial at x_target using Neville's Algorithm.
    """
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


def richardson_derivative(f, x: float, h: float = 0.1, steps: int = 3) -> float:
    """
    Calculates the derivative of f at x using Richardson Extrapolation.
    """
    table = [[0.0] * (i + 1) for i in range(steps)]

    for i in range(steps):
        current_h = h / (2 ** i)
        table[i][0] = (f(x + current_h) - f(x - current_h)) / (2 * current_h)

    for k in range(1, steps):
        for i in range(k, steps):
            denominator = (4 ** k) - 1
            table[i][k] = table[i][k - 1] + (table[i][k - 1] - table[i - 1][k - 1]) / denominator

    return table[steps - 1][steps - 1]


def main():
    # 8 נקודות הדגימה המקוריות שלך
    x_points = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
    y_points = [0.950, 1.166, 0.264, -0.317, -0.156, -0.549, 0.074, 1.076]

    # הגדרת פונקציית המטרה f(x) המבוססת על נוויל
    f_target = lambda x: neville_interpolation(x_points, y_points, x)

    print("==================================================")
    print("       STEP D: NUMERICAL DIFFERENTIATION          ")
    print("==================================================")

    # --------------------------------------------------
    # חלק 1: הדפסת טבלה ד' - נגזרות ב-8 נקודות הדגימה
    # --------------------------------------------------
    print("\n--- Table D: Richardson Derivative at Sample Points ---")
    print("x Point    | f(x) Value   | Richardson f'(x)")
    print("-" * 55)

    sample_derivatives = []
    for i in range(len(x_points)):
        x_val = x_points[i]
        y_val = y_points[i]

        current_h = 0.05
        if x_val - current_h < 0.0:
            deriv = (f_target(x_val + current_h) - f_target(x_val)) / current_h
        elif x_val + current_h > 3.0:
            deriv = (f_target(x_val) - f_target(x_val - current_h)) / current_h
        else:
            deriv = richardson_derivative(f_target, x=x_val, h=current_h, steps=3)

        print(f"{x_val:<10.2f} | {y_val:<12.3f} | {deriv:<20.8f}")
    print("-" * 55)

    # --------------------------------------------------
    # חלק 2: סריקת 200 נקודות למציאת נקודות הקיצון
    # --------------------------------------------------
    num_points = 200
    start_x = 0.0
    end_x = 3.0
    step = (end_x - start_x) / (num_points - 1)
    x_200 = [start_x + i * step for i in range(num_points)]

    max_y = -float('inf')
    min_y = float('inf')
    max_x_loc = 0.0
    min_x_loc = 0.0

    # סריקה לאיתור קיצון בתחום המוגן [0.1, 2.9]
    for x_val in x_200:
        if 0.1 <= x_val <= 2.9:
            y_val = f_target(x_val)
            if y_val > max_y:
                max_y = y_val
                max_x_loc = x_val
            if y_val < min_y:
                min_y = y_val
                min_x_loc = x_val

    # חישוב הנגזרות בנקודות הקיצון שנמצאו
    deriv_at_max = richardson_derivative(f_target, x=max_x_loc, h=0.05, steps=3)
    deriv_at_min = richardson_derivative(f_target, x=min_x_loc, h=0.05, steps=3)

    print("\n--- Extrema Points Found (Within [0.1, 2.9]) ---")
    print(f"🥇 GLOBAL MAXIMUM: at x = {max_x_loc:.4f}, f(x) = {max_y:.4f}, f'(x) ~ {deriv_at_max:.6f}")
    print(f"🛑 GLOBAL MINIMUM: at x = {min_x_loc:.4f}, f(x) = {min_y:.4f}, f'(x) ~ {deriv_at_min:.6f}")
    print("==================================================")

    # --------------------------------------------------
    # חלק 3: יצירת תרשים 3 - גרף הנגזרת f'(x)
    # --------------------------------------------------
    print("\nGenerating Graph for Diagram 3...")

    # חישוב רשימת נגזרות נקייה עבור הטווח המוגן כדי למנוע חריגות קצה בגרף
    graph_x = [x for x in x_200 if 0.05 <= x <= 2.95]
    graph_y = [richardson_derivative(f_target, x, h=0.05, steps=3) for x in graph_x]

    plt.figure(figsize=(10, 6))
    plt.plot(graph_x, graph_y, label="f'(x) - Richardson Extrapolation", color="red", linewidth=2)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)

    # סימון מיקומי נקודות הקיצון שמצאנו בגרף (היכן שהנגזרת חוצה או שואפת ל-0)
    plt.scatter([max_x_loc, min_x_loc], [deriv_at_max, deriv_at_min], color='blue', s=80, zorder=5,
                label="Extrema Locations")

    # הוספת תגיות טקסט קטנות מעל הנקודות בגרף
    plt.text(max_x_loc, deriv_at_max + 0.5, f"Max (x={max_x_loc:.2f})", ha='center', fontsize=9, fontweight='bold',
             color='blue')
    plt.text(min_x_loc, deriv_at_min - 0.7, f"Min (x={min_x_loc:.2f})", ha='center', fontsize=9, fontweight='bold',
             color='blue')

    plt.title("Diagram 3: Derivative f'(x) and Extrema Points via Richardson", fontsize=12, fontweight='bold')
    plt.xlabel("x Axis", fontsize=10)
    plt.ylabel("f'(x) Value", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="upper left")

    # שמירת הגרף כקובץ תמונה והצגתו על המסך
    plt.savefig("diagram3.png", dpi=300)
    print("Success: Graph saved as 'diagram3.png'!")
    plt.show()


if __name__ == "__main__":
    main()