import pickle
import pytest
import numpy as np


# Load the trained model from the pickle file
@pytest.fixture
def trained_model():
    with open('california_housing_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


# Sample test data for input
@pytest.fixture
def test_data_point():
    return np.array([0.5, 2.0, 1.0, 1.5, 3.0, 5.0, 0.7, 2.5])


def test_model_prediction(trained_model, test_data_point):
    # Use the model to make predictions on the test data
    predicted_value = trained_model.predict(test_data_point.reshape(1, -1))
    print(predicted_value[0])
    # Check if the prediction is within an acceptable range
    assert -1 <= predicted_value[0] <= 1, "Predicted value is out of range"
