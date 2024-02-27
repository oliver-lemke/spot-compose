import numpy as np

from sklearn.linear_model import LinearRegression, RANSACRegressor


def plane_fitting(pcd: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    assert pcd.shape[-1] == 3

    X = pcd[:, :2]  # x and y coordinates
    y = pcd[:, 2]  # z coordinates
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=3,
        residual_threshold=threshold,
        max_trials=100,
    )
    ransac.fit(X, y)

    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_
    normal = np.array([a, b, -1])
    normal_normalized = normal / np.linalg.norm(normal)
    return normal_normalized
