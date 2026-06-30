import sympy


def initializeSympyPolynomialData(poly_expr, symbol):
    # בדיקה: האם יש נעלמים שהמשתמש הכניס בטעות ולא הגדרנו?
    extra_symbols = poly_expr.atoms(sympy.Symbol) - {symbol}
    if extra_symbols:
        print(f"⚠️ Warning: Found extra symbols in your function: {extra_symbols}")
        print(f"Make sure you only use '{symbol}' and constants like 'E' or 'pi'.")

    derivative_expr = sympy.diff(poly_expr, symbol)
    second_derivative_expr = sympy.diff(derivative_expr, symbol)

    f = sympy.utilities.lambdify(symbol, poly_expr, 'math')
    fTag = sympy.utilities.lambdify(symbol, derivative_expr, 'math')
    fTagTag = sympy.utilities.lambdify(symbol, second_derivative_expr, 'math')

    return f, fTag, fTagTag


def Bisection_Method(poly_func, start_point, end_point, epsilon=0.0001):
    if poly_func(start_point) * poly_func(end_point) >= 0:
        print(f"\n--- Searching in range [{start_point:.2f}, {end_point:.2f}] ---")
        print("-> Message: The method does not converge (No sign change detected).")
        return None, 0

    print(f"\n--- Searching in range [{start_point:.2f}, {end_point:.2f}] ---")
    print("| Iteration | Current x (Root) |")
    print("|-----------|------------------|")

    iterations = 0
    a, b = start_point, end_point
    c = a

    while (b - a) >= epsilon:
        iterations += 1
        c = (a + b) / 2.0
        print(f"| {iterations:<9} | {c:<16.5f} |")

        if poly_func(c) == 0.0:
            break
        elif poly_func(c) * poly_func(a) < 0:
            b = c
        else:
            a = c

    print("-" * 32)
    print(f"-> Final Root found: {c:.5f} | Total Iterations: {iterations}")
    return c, iterations


def Newton_Raphson(poly_func, derivative_func, start_point, end_point, epsilon=0.0001):
    current_x = (start_point + end_point) / 2.0
    iterations = 0
    max_iterations = 100

    print(f"\n--- Searching in range [{start_point:.2f}, {end_point:.2f}] ---")
    print("| Iteration | Current x (Root) |")
    print("|-----------|------------------|")

    while iterations < max_iterations:
        f_x = poly_func(current_x)
        f_prime_x = derivative_func(current_x)

        if f_prime_x == 0:
            print("-" * 32)
            print("-> Message: The method does not converge (derivative is zero).")
            return None, 0

        next_x = current_x - (f_x / f_prime_x)
        iterations += 1
        print(f"| {iterations:<9} | {next_x:<16.5f} |")

        if abs(next_x - current_x) < epsilon:
            if min(start_point, end_point) <= next_x <= max(start_point, end_point):
                print("-" * 32)
                print(f"-> Final Root found: {next_x:.5f} | Total Iterations: {iterations}")
                return next_x, iterations
            else:
                print("-" * 32)
                print("-> Message: The method converged, but the root is outside range.")
                return None, 0

        current_x = next_x

    print("-" * 32)
    print("-> Message: The method does not converge (max iterations).")
    return None, 0


def secant_method(poly_func, start_point, end_point, epsilon=0.0001):
    x0, x1 = start_point, end_point
    iterations = 0
    max_iterations = 100

    print(f"\n--- Searching in range [{start_point:.2f}, {end_point:.2f}] ---")
    print("| Iteration | Current x (Root) |")
    print("|-----------|------------------|")

    while iterations < max_iterations:
        f_x0 = poly_func(x0)
        f_x1 = poly_func(x1)

        if abs(f_x1 - f_x0) < 1e-12:
            print("-" * 32)
            print("-> Message: The method does not converge (division by zero).")
            return None, 0

        x2 = x1 - f_x1 * ((x1 - x0) / (f_x1 - f_x0))
        iterations += 1
        print(f"| {iterations:<9} | {x2:<16.5f} |")

        if abs(x2 - x1) < epsilon or abs(poly_func(x2)) < epsilon:
            if min(start_point, end_point) - epsilon <= x2 <= max(start_point, end_point) + epsilon:
                print("-" * 32)
                print(f"-> Final Root found: {x2:.5f} | Total Iterations: {iterations}")
                return x2, iterations
            else:
                print("-" * 32)
                print("-> Message: The method converged, but the root is outside range.")
                return None, 0

        x0, x1 = x1, x2

    print("-" * 32)
    print("-> Message: The method does not converge (max iterations).")
    return None, 0


# =========================================================
# התוכנית הראשית
# =========================================================
if __name__ == "__main__":
    x = sympy.symbols('x')

    print("--- Equation Root Finder ---")

    # 1. פולינום קבוע (ללא צורך בקלט מהמשתמש)
    poly_str = "(x - 1)**2 * (x + 2)"
    print(f"Analyzing polynomial: f(x) = {poly_str}")
    my_polynomial = sympy.sympify(poly_str)

    f_func, fTag_func, fTagTag_func = initializeSympyPolynomialData(my_polynomial, x)

    # 3. תחום וגודל מקטע קבועים שמותאמים לפולינום
    large_start = -3.0
    large_end = 3.0
    step = 0.6  # Segment size
    epsilon = 0.0001

    print(f"Search range: [{large_start}, {large_end}]")
    print(f"Scanning segment size: {step}\n")

    # השארתי את הבחירה בשיטה כדי שהמרצה יוכל לבדוק את שלושתן על אותו פולינום
    print("Select method: 1. Bisection | 2. Newton-Raphson | 3. Secant")
    choice = input("Choice: ")

    found_roots = []
    current = large_start

    print("\nScanning for roots...")
    while current < large_end:
        a = current
        b = min(current + step, large_end)

        # Check for regular root at endpoints
        if abs(f_func(a)) < epsilon:
            if not any(abs(a - r[0]) < 0.01 for r in found_roots):
                found_roots.append((a, 0, "Odd (Regular cross)"))
        if abs(f_func(b)) < epsilon:
            if not any(abs(b - r[0]) < 0.01 for r in found_roots):
                found_roots.append((b, 0, "Odd (Regular cross)"))

        # חיפוש שורש רגיל
        if f_func(a) * f_func(b) < 0:
            root, iters = None, 0
            if choice == '1':
                root, iters = Bisection_Method(f_func, a, b)
            elif choice == '2':
                root, iters = Newton_Raphson(f_func, fTag_func, a, b)
            elif choice == '3':
                root, iters = secant_method(f_func, a, b)

            if root is not None and not any(abs(root - r[0]) < 0.01 for r in found_roots):
                found_roots.append((root, iters, "Odd (Regular cross)"))

        # Check for even root at endpoints
        if abs(fTag_func(a)) < epsilon and abs(f_func(a)) < epsilon:
            if not any(abs(a - r[0]) < 0.01 for r in found_roots):
                found_roots.append((a, 0, "Even (Tangent touch)"))
        if abs(fTag_func(b)) < epsilon and abs(f_func(b)) < epsilon:
            if not any(abs(b - r[0]) < 0.01 for r in found_roots):
                found_roots.append((b, 0, "Even (Tangent touch)"))

        # חיפוש שורש מריבוב זוגי
        if fTag_func(a) * fTag_func(b) < 0:
            ext_root, iters = None, 0
            if choice == '1':
                ext_root, iters = Bisection_Method(fTag_func, a, b)
            elif choice == '2':
                ext_root, iters = Newton_Raphson(fTag_func, fTagTag_func, a, b)
            elif choice == '3':
                ext_root, iters = secant_method(fTag_func, a, b)

            if ext_root is not None and abs(f_func(ext_root)) < epsilon:
                if not any(abs(ext_root - r[0]) < 0.01 for r in found_roots):
                    found_roots.append((ext_root, iters, "Even (Tangent touch)"))

        current += step

    # הדפסת טבלת התוצאות הסופית כולל איטרציות (לפי סעיף 3.4 במטלה)
    print("\n--- Final Summary Table ---")
    if not found_roots:
        print("No roots found.")
    else:
        print("| Root # | x          | Iterations | Type                  |")
        print("|--------|------------|------------|-----------------------|")
        for i, (r, iters, r_type) in enumerate(found_roots):
            print(f"| {i + 1:<6} | {r:<10.5f} | {iters:<10} | {r_type:<21} |")