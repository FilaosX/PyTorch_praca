import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

import pandas as pd
import pickle

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

# Przykład użycia:
# Przykład użycia:
file_path = 'train_split_info/los1mini.pkl'
train_images, val_images, test_images = load_images_from_pickle(file_path)

print(f"Liczba zdjęć trenujących: {len(train_images)}")
print(f"Liczba zdjęć walidujących: {len(val_images)}")
print(f"Liczba zdjęć testujących: {len(test_images)}")
csv_file_path = 'meta_info/oceny_zdjec_mini.csv'
#selected_rows = [2, 5, 8, 10, 15]  # Numery wierszy do pobrania

image_test_names = get_image_names(csv_file_path, test_images)
image_train_names = get_image_names(csv_file_path, train_images)
image_test_values = get_image_values(csv_file_path, test_images)
image_train_values = get_image_values(csv_file_path, train_images)
# Wyświetlenie pobranych nazw obrazów
print("Pobrane nazwy obrazów:")
#for name in image_train_names:
    #print(name)
#for value in image_train_values:
    #print(value)
train_data_folder = 'zdjecia_uczenie_mini'

# Tworzenie generatora danych treningowych
datagen = ImageDataGenerator(rescale=1./255)
print("jestem0")
# Dane treningowe do numpy array
X_train = []
for img_name in image_train_names:
    img_path = os.path.join(train_data_folder, img_name)
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    X_train.append(img_array)
print("jestem1")
X_train = np.array(X_train)
y_train = image_train_values
print("jestem2")
# Dane testowe do numpy array
X_test = []
for img_name in image_test_names:
    img_path = os.path.join(train_data_folder, img_name)
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    X_test.append(img_array)
print("jestem3")
X_test = np.array(X_test)
y_test = image_test_values
print("jestem4")
# Trenowanie modelu drzewa decyzyjnego
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train.reshape(-1, 224*224*3), y_train)
print("jestem5")
# Klasyfikacja na danych testowych
y_pred = model.predict(X_test.reshape(-1, 224*224*3))
print("jestem6")
# Obliczenie dokładności
x=0
tp=0
for i in y_pred:
    a=int(i)
    b=int(y_test[x])
    if b==a:
        tp=tp+1
    x=x+1
dok_ac=(tp/x)*100
print(tp)
print(x)
print(f'Dokładność modelu accuracy: {dok_ac:.2f}%')
