import os
import numpy as np
import pickle
from keras.preprocessing.image import load_img, img_to_array
from sklearn.linear_model import LogisticRegression

def load_trained_model(model_filename):
    try:
        with open(model_filename, 'rb') as model_file:
            trained_model = pickle.load(model_file)
        return trained_model
    except Exception as e:
        print(f"Błąd podczas wczytywania modelu z pliku: {e}")
        return None

def preprocess_image(img_path, target_size=(512, 512)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array_flat = img_array.reshape((1, -1))
    return img_array_flat

def classify_image(model, img_path):
    try:
        img_array_flat = preprocess_image(img_path)
        prediction = model.predict(img_array_flat)
        return prediction[0]
    except Exception as e:
        print(f"Błąd podczas klasyfikacji obrazu: {e}")
        return None

# Przykład użycia:
model_filename = 'logistic_model2.pkl'
new_image_path = 'zdjecia_uczenie_mini/obraz5 (860).png'

# Wczytanie nauczonego modelu
loaded_model = load_trained_model(model_filename)

if loaded_model:
    # Klasyfikacja nowego obrazu
    prediction = classify_image(loaded_model, new_image_path)

    if prediction is not None:
        print(f"Przewidziana klasa: {prediction}")
    else:
        print("Nie można dokonać klasyfikacji.")
else:
    print("Wczytanie modelu nie powiodło się.")
