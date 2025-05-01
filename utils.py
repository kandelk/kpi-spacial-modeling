import numpy as np


def scale_points(points: np.ndarray, factor: float) -> np.ndarray:
    return points * factor

def translate_points(points: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return points + np.array([dx, dy])

def rotate_points(points: np.ndarray, angle_degrees: float, center: tuple = (0.0, 0.0)) -> np.ndarray:
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians),  np.cos(angle_radians)]
    ])
    centered = points - center  # broadcast subtraction
    rotated = centered @ rotation_matrix.T  # matrix multiplication
    return rotated + center