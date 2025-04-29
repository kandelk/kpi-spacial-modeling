import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

BEZIER_WEIGHTS = [1, 1, 1, 1]
BEZIER_STEP = 3  # overlap for smoothness

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


def rational_bezier(P, w, num_points=100):
    t_vals = np.linspace(0, 1, num_points)
    curve = []
    for t in t_vals:
        numerator = np.zeros(2)
        denominator = 0
        for i in range(4):
            b = bernstein_poly(i, t)
            numerator += w[i] * P[i] * b
            denominator += w[i] * b
        curve.append(numerator / denominator)
    return np.array(curve)

def print_contour(contour, title):
    plt.figure(title)
    plt.axis('equal')
    plt.title(title + " Кубічна крива Безьє")
    plt.axis('off')
    plt.plot(contour[:, 0], contour[:, 1], 'r.', label='Точки')

    contour_closed = np.vstack((contour, contour[:3]))

    # Проходимо по точках контуру та будуємо прямі
    for i in range(0, len(contour_closed) - 3, BEZIER_STEP):
        P = contour_closed[i:i+4]
        if len(P) < 4:
            continue
        bezier_curve = rational_bezier(P, BEZIER_WEIGHTS, 100)
        if i == 0:
            plt.plot(bezier_curve[:, 0], bezier_curve[:, 1], 'b', label='Безьє 3-го ступеня')
        else:
            plt.plot(bezier_curve[:, 0], bezier_curve[:, 1], 'b')

    plt.gca().invert_yaxis()
    plt.legend()

# Завантажуємо зображення
image = cv.imread('../plane.png', cv.IMREAD_GRAYSCALE)
plt.figure("Вхідний малюнок")
plt.axis('off')
plt.title("Вхідний малюнок")
plt.imshow(image, cmap='gray')

# Застосовуємо порогову обробку для бінаризації зображення
_, thresh = cv.threshold(image, 127, 255, cv.THRESH_BINARY_INV)

# Знаходимо контури на зображенні
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# Перевіряємо, скільки контурів знайдено
print(f"Знайдено {len(contours)} контурів")

# Створюємо зображення для відображення контурів
contour_image = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255

# Для кожного контурного елемента малюємо контур на зображенні
for contour in contours:
    cv.drawContours(contour_image, [contour], -1, (0, 0, 0), 2)

# Відображаємо зображення з контурами
plt.figure("Знайдені контури")
plt.imshow(contour_image)
plt.axis('off')
plt.title("Знайдені контури")

# Обираємо контур для роботи
contour = max(contours, key=cv.contourArea)
contour = contour.squeeze()
print(f"Контур має {len(contour)} точок")

# Будуємо криву по точках контуру
print_contour(contour, "Повний контур")

# Зменшимо кількість точок використоуючи алгоритм Рамера-Дугласа-Пекера
epsilon = 0.0005 * cv.arcLength(contour, True)
simplified_contour = cv.approxPolyDP(contour, epsilon, True)
simplified_contour = simplified_contour.squeeze()
print(f"Спрощений контур має {len(simplified_contour)} точок")

# Будуємо криву по точках спрощеного контуру
print_contour(simplified_contour, "Спрощений контур")

# Друкуємо результат
plt.show()
