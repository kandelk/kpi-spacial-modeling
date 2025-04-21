import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

Y_FROM, Y_TO, Y_POINTS = 0, 0.5, 50
X_FROM, X_TO, X_POINTS = 0, 2 * np.pi, 50

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

plt.plot(contour[:, 0], contour[:, 1], 'r.', label='Точки')
plt.legend()

# Створюємо сітку параметрів u та v
u_vals = np.linspace(X_FROM, X_TO, X_POINTS)
v_vals = np.linspace(Y_FROM, Y_TO, Y_POINTS)

# Створюємо поверхню горщика
X, Y, Z = create_pot_surface(u_vals, v_vals)

# Створюємо 3D графік
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Побудова поверхні
ax.plot_surface(X, Y, Z, color='yellow', edgecolor='k', alpha=1, shade=False)

# # Наносимо літак на поверхню горщика
# # Розміщуємо літак на верхній частині поверхні
for point in contour:
    # Перетворюємо кожну точку літака на поверхню
    # Вибираємо v = 2 для верхнього краю горщика
    R = 1 + abs(np.sin(2 * np.pi * Y_TO))
    u = point[0] / image.shape[1] * 2 * np.pi  # Перетворюємо x-координату зображення в кут u
    X_l, Y_l, Z_l = R * np.cos(u), R * np.sin(u), Y_TO
    ax.scatter(X_l, Y_l, Z_l, color='r', s=10)  # Наносимо точки літака на поверхню

# Налаштування графіка
ax.set_title("Горщик з нанесеним малюнком літака")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=30, azim=60)  # Встановлюємо бажаний кут огляду

# Показуємо графік
plt.show()
