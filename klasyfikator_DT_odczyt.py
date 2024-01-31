import os
from skimage import io
from keras.preprocessing.image import img_to_array, load_img
import joblib
import pickle
import pandas as pd

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
        #print(f"Błąd klasyfikacji: {e}")
        return None

def load_images_from_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            train_images = data.get(1, {}).get('train', [])
            val_images = data.get(1, {}).get('val', [])
            test_images = data.get(1, {}).get('test', [])
            return train_images, val_images, test_images
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku pickle: {e}")
        return [], [], []


def get_image_names(csv_file_path, selected_rows):
    try:
        # Wczytanie danych z pliku CSV
        data = pd.read_csv(csv_file_path)

        # Pobranie nazw obrazów dla wybranych wierszy
        image_names = data.iloc[selected_rows, 0].tolist()

        return image_names
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku CSV: {e}")
        return []

def get_image_values(csv_file_path, selected_rows):
    try:
        # Wczytanie danych z pliku CSV
        data = pd.read_csv(csv_file_path)

        # Pobranie nazw obrazów dla wybranych wierszy
        image_names = data.iloc[selected_rows, 1].tolist()

        return image_names
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku CSV: {e}")
        return []

# Wczytaj nauczony model
model_file_path = 'plik_decision_tree.pkl'
loaded_model = load_model_from_file(model_file_path)


file_path = 'train_split_info/los1mini.pkl'
train_images, val_images, test_images = load_images_from_pickle(file_path)

print(f"Liczba zdjęć trenujących: {len(train_images)}")
print(f"Liczba zdjęć walidujących: {len(val_images)}")
print(f"Liczba zdjęć testujących: {len(test_images)}")
csv_file_path = 'meta_info/oceny_zdjec_mini.csv'

image_test_names = get_image_names(csv_file_path, test_images)
image_train_names = get_image_names(csv_file_path, train_images)
image_test_values = get_image_values(csv_file_path, test_images)
image_train_values = get_image_values(csv_file_path, train_images)

i=0
a=0
if loaded_model:
    for obraz in image_test_names:
        # Przykład użycia do klasyfikacji nowego obrazu
        new_image_path = f"zdjecia_uczenie_mini/{obraz}"
        prediction = classify_image(loaded_model, new_image_path)
        klasa=image_test_values[i]
        i=i+1
        if prediction is not None:
           print(f'Przewidziana klasa: {prediction}')
           print(f'Prawdziwa klasa: {klasa}')
           if int(prediction) == int(klasa):
               a=a+1
        #else:
         #   print('Błąd klasyfikacji.')
    print(i)
    print(a)
    dok_ac = (a / i) * 100
    print(f'Dokładność modelu accuracy: {dok_ac:.2f}%')