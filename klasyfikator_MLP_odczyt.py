import os
import numpy as np
from skimage import io
from keras.src.utils import load_img, img_to_array
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_model_from_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            model = joblib.load(file)
        return model
    except Exception as e:
        print(f"Błąd ładowania modelu: {e}")
        return None

def classify_image(model, img_path):
    try:
        img = io.imread(img_path)
        img_array = img.flatten().reshape(1, -1)
        prediction = model.predict(img_array)
        return prediction[0]
    except Exception as e:
        print(f"Błąd: {e}")
        return None

# Wczytaj nauczony model
model_file_path = 'trained_model_mlp.pkl'
loaded_model = load_model_from_file(model_file_path)

if loaded_model:
    # Przykład użycia do klasyfikacji nowego obrazu
    new_image_path = 'zdjecia_uczenie_mini/obraz5 (869).png'
    prediction = classify_image(loaded_model, new_image_path)

    if prediction is not None:
        print(f'Przewidziana klasa: {prediction}')
    else:
        print('Błąd klasyfikacji')
