import os
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Ścieżka do folderu z obrazami
images_folder = 'H:\Projekt badawczy\zdjecia_uczenie'
ocena_folder = 'H:\Projekt badawczy\zdjecia_do_oceny'
# Ścieżka do pliku CSV z ocenami w skali Likerta
ratings_file = 'H:\Projekt badawczy\zdjecia_uczenie\oceny_zdjec.csv'
do_oceny_file = 'H:\Projekt badawczy\zdjecia_do_oceny\zdjecia_do_oceny.csv'
# Wczytaj oceny z pliku CSV
ratings_df = pd.read_csv(ratings_file)
do_oceny_df = pd.read_csv(do_oceny_file)
# Tworzenie list na obrazy i oceny
images = []
ratings = []
oceny = []
nazwy_zdjec = []
oceny_generowane = []
przeskalowane_oceny = []
# Przetwarzanie obrazów i ocen
for index, row in ratings_df.iterrows():
    image_name = row['nazwa']
    image_path = os.path.join(images_folder, image_name)

    image = load_img(image_path, target_size=(500, 500))
    image = img_to_array(image)
    image = preprocess_input(image)

    rating = row['ocena']

    images.append(image)
    ratings.append(rating)

for index, row in do_oceny_df.iterrows():
    image_name_o = row['nazwa']
    image_path_o = os.path.join(ocena_folder, image_name_o)
    nazwy_zdjec.extend([image_name_o])
    image_o = load_img(image_path_o, target_size=(500, 500))
    image_o = img_to_array(image_o)
    image_o = preprocess_input(image_o)
    oceny.append(image_o)

# Konwertuj listy na tablice numpy
images = np.array(images)
ratings = np.array(ratings)

# Podziel dane na zbiór treningowy i walidacyjny
split = int(0.8 * len(images))
train_images, val_images = images[:split], images[split:]
train_ratings, val_ratings = ratings[:split], ratings[split:]

# Wczytaj pre-trenowany model VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(500, 500, 3))

# Zamrożenie wag dla warstw bazowego modelu
for layer in base_model.layers:
    layer.trainable = False

# Dodaj warstwy klasyfikatora do modelu
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))  # Warstwa wyjściowa z jednym neuronem dla oceny

# Kompilacja modelu
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Trenowanie modelu
model.fit(train_images, train_ratings, epochs=10, batch_size=32, validation_data=(val_images, val_ratings))
zdjecia_ocenione_siec = []
i=0;
for image_o in oceny:#new_images:
    #image_name_o = row['nazwa']
    #image_path_o = os.path.join(ocena_folder, image_name_o)
    #image_o = load_img(image_path_o, target_size=(500, 500))
    #image_o = img_to_array(image_o)
    #image_o = preprocess_input(image_o)
    image_o = np.expand_dims(image_o, axis=0)
    rating = model.predict(image_o)
    #print("Ocena:", rating)
    oceny_generowane.extend([rating])

najmniejsza = min(oceny_generowane)
najwieksza = max(oceny_generowane)

# Przeskalowanie wartości na zakres od 1 do 5
skala_min = 1
skala_max = 5
przeskalowana_tablica = []

for liczba in oceny_generowane:
    przeskalowana = skala_min + ((liczba - najmniejsza) * (skala_max - skala_min)) / (najwieksza - najmniejsza)
    przeskalowane_oceny.append(przeskalowana)
i=0
for a in nazwy_zdjec:
    print("Ocena:", int(przeskalowane_oceny[i]))
    zdjecia_ocenione_siec.append({"nazwa": nazwy_zdjec[i], "ocena": int(przeskalowane_oceny[i])})
    i=i+1


# Ścieżka do pliku CSV
plik_csv = "H:\Projekt badawczy\zdjecia_do_oceny\zdjecia_ocenione_siec.csv"

# Zapis ocen do pliku CSV
with open(plik_csv, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["nazwa", "ocena"])
    writer.writeheader()
    writer.writerows(zdjecia_ocenione_siec)

print("Plik CSV został wygenerowany.")
