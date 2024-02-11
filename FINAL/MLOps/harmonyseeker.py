
from joblib import load

# Assume necessary imports (pickled model, etc.)

model = load('model.joblib')


def predict_pop(processed_data):
    # Load the trained model
    prediction = model.predict(processed_data)
    return prediction
