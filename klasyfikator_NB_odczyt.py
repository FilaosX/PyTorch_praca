import os
from skimage import io
from keras.preprocessing.image import img_to_array, load_img
import joblib

def load_model_from_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            model = joblib.load(file)
        return model
    except Exception as e:
        print(f"Błąd wczytania modelu: {e}")
        return None

def classify_image(model, img_path):
    try:
        img = io.imread(img_path)
        img_array = img.flatten().reshape(1, -1)
        prediction = model.predict(img_array)
        return prediction[0]
    except Exception as e:
        print(f"Błą klasyfikacji: {e}")
        return None

# Wczytaj nauczony model
model_file_path = 'plik_naive_bayes.pkl'
loaded_model = load_model_from_file(model_file_path)

if loaded_model:
    # Przykład użycia do klasyfikacji nowego obrazu
    new_image_path = 'zdjecia_uczenie_mini/obraz5 (869).png'
    prediction = classify_image(loaded_model, new_image_path)

    if prediction is not None:
        print(f'Przewidziana klasa: {prediction}')
    else:
        print('Klasyfikacja nieudana')
