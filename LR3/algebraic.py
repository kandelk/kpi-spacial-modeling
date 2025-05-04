import numpy as np
import matplotlib.pyplot as plt

def newton_fractal(width=800, height=800, max_iter=50, epsilon=1e-6):
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    result = np.zeros((height, width), dtype=float)

    for i, yi in enumerate(y):
        for j, xj in enumerate(x):
            z = complex(xj, yi)
            for k in range(max_iter):
                try:
                    z_next = (3 * z**4 + 1) / (4 * z**3)
                except ZeroDivisionError:
                    break
                if abs(z_next**4 - 1) <= epsilon:
                    break
                z = z_next
            # Normalize iterations for grayscale
            result[i, j] = k / max_iter

    return result

# Generate and plot
fractal = newton_fractal(width=1000, height=1000, max_iter=50, epsilon=1e-6)

plt.imshow(fractal, cmap='gray', extent=(-1, 1, -1, 1))
plt.title("Grayscale Newton Fractal")
plt.axis('off')
plt.show()
