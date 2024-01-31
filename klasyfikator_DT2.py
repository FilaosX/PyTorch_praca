import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from skimage import io

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

def main():
    # Ścieżka do folderu z obrazami
    image_folder_path = 'zdjecia_uczenie_mini'

    # Ścieżka do pliku CSV
    csv_file_path = 'meta_info/oceny_zdjec_mini.csv'

    # Wczytanie obrazów z folderu
    images = load_images_from_folder(image_folder_path)

    # Wczytanie danych z pliku CSV
    data = load_data_from_csv(csv_file_path)

    # Kodowanie etykiet numerycznych
    label_encoder = LabelEncoder()
    data['mos'] = label_encoder.fit_transform(data['mos'])

    # Podział danych na zbiór uczący i testujący
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Przygotowanie danych uczących
    X_train = [img[1] for img in images if img[0] in train_data['img_name'].values]
    y_train = train_data['mos'].values

    # Przygotowanie danych testujących
    X_test = [img[1] for img in images if img[0] in test_data['img_name'].values]
    y_test = test_data['mos'].values

    # Utworzenie i trenowanie klasyfikatora Decision Tree
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Predykcja na danych testujących
    y_pred = classifier.predict(X_test)

    # Ocenia dokładność klasyfikatora
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Dokładność klasyfikatora: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()
