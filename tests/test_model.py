import numpy as np
from sklearn.ensemble import RandomForestRegressor


def test_model_training():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([10, 20, 30, 40])

    model = RandomForestRegressor(
        n_estimators=10,
        random_state=42
    )

    model.fit(X, y)

    prediction = model.predict([[2, 3]])

    assert len(prediction) == 1
    assert prediction[0] > 0