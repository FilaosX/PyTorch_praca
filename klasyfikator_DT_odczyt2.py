import os
from skimage import io
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
import joblib
import pickle
import pandas as pd
import numpy as np

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