import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

Y_FROM, Y_TO, Y_POINTS = 0, 0.5, 100
X_FROM, X_TO, X_POINTS = 0, 2 * np.pi, 100

BEZIER_WEIGHTS = [1, 1, 1, 1]
BEZIER_STEP = 3  # overlap for smoothness

# Базисные полиномы Бернштейна
def bernstein_poly(i, t):
    if i == 0:
        return (1 - t)**3
    elif i == 1:
        return 3 * t * (1 - t)**2
    elif i == 2:
        return 3 * t**2 * (1 - t)
    elif i == 3:
        return t**3
    return None

# Функция для построения рациональной кривой Безье
def rational_bezier_3d(P, w, num_points=100):
    t_vals = np.linspace(0, 1, num_points)
    curve = []
    for t in t_vals:
        numerator = np.zeros(3)
        denominator = 0
        for i in range(4):
            b = bernstein_poly(i, t)
            numerator += w[i] * P[i] * b
            denominator += w[i] * b
        curve.append(numerator / denominator)
    return curve

def build_bezier_curve(contour):
    contour_closed = np.vstack((contour, contour[:3]))

    full_curve = []

    # Проходимо по точках контуру та будуємо прямі
    for i in range(0, len(contour_closed) - 3, BEZIER_STEP):
        P = contour_closed[i:i+4]
        if len(P) < 4:
            continue
        bezier_curve = rational_bezier_3d(P, BEZIER_WEIGHTS, 100)
        full_curve.extend(bezier_curve)

    return np.array(full_curve)


def map_to_surface(point_2d):
    u = point_2d[0]
    v = point_2d[1]
    R = 1 + abs(np.sin(2 * np.pi * v))
    x = R * np.cos(u)
    y = R * np.sin(u)
    z = v
    return np.array([x, y, z])

def normalize_2d_to_range(points, x_range, y_range):
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # Normalize to [0, 1]
    normalized = (points - min_vals) / (max_vals - min_vals)

    # Scale to target ranges
    x_scaled = normalized[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
    y_scaled = normalized[:, 1] * (y_range[1] - y_range[0]) + y_range[0]

    return np.stack((x_scaled, y_scaled), axis=1)

# Функція для побудови поверхні горщика
def create_pot_surface(u_vals, v_vals):
    X = np.zeros((len(u_vals), len(v_vals)))
    Y = np.zeros((len(u_vals), len(v_vals)))
    Z = np.zeros((len(u_vals), len(v_vals)))

    for i, u in enumerate(u_vals):
        for j, v in enumerate(v_vals):
            R = 1 + abs(np.sin(2 * np.pi * v))  # Заданий радіус
            X[i, j] = R * np.cos(u)
            Y[i, j] = R * np.sin(u)
            Z[i, j] = v

    return X, Y, Z

# Завантажуємо зображення
image = cv.imread('plane.png', cv.IMREAD_GRAYSCALE)

# Знаходимо контури
_, thresh = cv.threshold(image, 127, 255, cv.THRESH_BINARY_INV)
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

print(f"Знайдено {len(contours)} контурів")

contour_image = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
for contour in contours:
    cv.drawContours(contour_image, [contour], -1, (0, 0, 0), 2)

plt.figure("Знайдені контури")
plt.imshow(contour_image)
plt.axis('off')
plt.title("Знайдені контури")

contour = max(contours, key=cv.contourArea)

epsilon = 0.0005 * cv.arcLength(contour, True)
contour = cv.approxPolyDP(contour, epsilon, True)
contour = contour.squeeze()
print(f"Контур має {len(contour)} точок")
contour_normalized = normalize_2d_to_range(contour, (X_FROM, X_TO), (Y_FROM, Y_TO))

plt.plot(contour[:, 0], contour[:, 1], 'r.', label='Точки')
plt.legend()

# Преобразование 2D-контуров в 3D
contour_3d = [map_to_surface(point) for point in contour_normalized]
contour_3d = np.array(contour_3d)

# Построение кривой Безье на 3D-поверхности
# bezier_curve_3d = rational_bezier_3d(contour_3d, BEZIER_WEIGHTS)
bezier_curve_3d = build_bezier_curve(contour_3d)

# Створюємо поверхню горщика
u_vals = np.linspace(X_FROM, X_TO, X_POINTS)
v_vals = np.linspace(Y_FROM, Y_TO, Y_POINTS)

X, Y, Z = create_pot_surface(u_vals, v_vals)

# Створюємо 3D графік
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Побудова поверхні
ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.6, edgecolor=None, shade=False)

# Кривая Безье
ax.plot(bezier_curve_3d[:, 0], bezier_curve_3d[:, 1], bezier_curve_3d[:, 2], 'r-', label='Кривая Безье')

# Налаштування графіка
ax.set_title("Горщик з нанесеним малюнком літака")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=30, azim=60)  # Встановлюємо бажаний кут огляду

# Показуємо графік
plt.show()
