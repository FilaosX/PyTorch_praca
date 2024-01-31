import os
import numpy as np
import pickle
from keras.preprocessing.image import load_img, img_to_array
from sklearn.linear_model import LogisticRegression
from keras.preprocessing.image import ImageDataGenerator
import joblib
import pandas as pd
import numpy as np

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
model_filename = 'logistic_model2.pkl'

# Wczytanie nauczonego modelu
loaded_model = load_trained_model(model_filename)


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

train_data_folder = 'zdjecia_uczenie_mini'
# Tworzenie generatora danych treningowych
datagen = ImageDataGenerator(rescale=1./255)
print('jestem0')
# Dane testowe do numpy array
X_test = []
for img_name in image_test_names:
    img_path = os.path.join(train_data_folder, img_name)
    img = load_img(img_path, target_size=(512, 512))
    img_array = img_to_array(img)
    X_test.append(img_array)
print('jestem1')
X_test = np.array(X_test)
y_test = image_test_values

print('jestem2')
# Klasyfikacja na danych testowych
y_pred = loaded_model.predict(X_test.reshape(-1, 512*512*3))
print('jestem3')
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
