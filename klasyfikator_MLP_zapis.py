import os
import pandas as pd
import numpy as np
from skimage import io
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.src.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import pickle
import joblib

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = io.imread(img_path)
        images.append((filename, img))
    return images

def load_data_from_csv(csv_file_path):
    data = pd.read_csv(csv_file_path)
    return data

def preprocess_data(images, data):
    X = [img[1] for img in images if img[0] in data['filename'].values]
    y = data['label'].values
    X = np.array([img.flatten() for img in X])
    return X, y

def train_mlp_classifier(X_train, y_train, hidden_layer_sizes=(100,), max_iter=1000):
    classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42)
    classifier.fit(X_train, y_train)
    return classifier

def evaluate_classifier(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Dokładność klasyfikatora: {accuracy * 100:.2f}%')

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

def save_model_to_file(model, file_path):
    try:
        with open(file_path, 'wb') as file:
            joblib.dump(model, file)
        print(f'Model saved to {file_path}')
    except Exception as e:
        print(f"Error saving the model: {e}")

# Przykład użycia:
# Przykład użycia:
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
# Wyświetlenie pobranych nazw obrazów
print("Pobrane nazwy obrazów:")
#for name in image_train_names:
    #print(name)
#for value in image_train_values:
    #print(value)
train_data_folder = 'zdjecia_uczenie_mini'

# Tworzenie generatora danych treningowych
datagen = ImageDataGenerator(rescale=1./255)
print('jestem0')
# Dane treningowe do numpy array
X_train = []
for img_name in image_train_names:
    img_path = os.path.join(train_data_folder, img_name)
    img = load_img(img_path, target_size=(512, 512))
    img_array = img_to_array(img)
    X_train.append(img_array)
print('jestem1')
X_train = np.array(X_train)
y_train = image_train_values
print('jestem2')
# Dane testowe do numpy array
X_test = []
for img_name in image_test_names:
    img_path = os.path.join(train_data_folder, img_name)
    img = load_img(img_path, target_size=(512, 512))
    img_array = img_to_array(img)
    X_test.append(img_array)
print('jestem3')
X_test = np.array(X_test)
y_test = image_test_values
print('jestem4')
# Obrazy do wektora
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))
print('jestem5')
# Utworzenie i trenowanie klasyfikatora MLP
classifier = train_mlp_classifier(X_train_flat, y_train)
print('jestem6')
# Zapiszmy model do pliku
model_file_path = 'trained_model_mlp.pkl'
save_model_to_file(classifier, model_file_path)
print('jestem6')
# Ocena klasyfikatora na danych testujących
evaluate_classifier(classifier, X_test_flat, y_test)


