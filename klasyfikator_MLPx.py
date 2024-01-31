import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

# Tworzenie generatora danych testowych
test_generator = datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=train_data_folder,
    x_col="filename",
    y_col="label",
    batch_size=32,
    seed=42,
    shuffle=False,  # Ustawienie na False, aby zachować kolejność
    class_mode="sparse",  # Zmienione na "sparse"
    target_size=(224, 224),
    validate_filenames=False
)

# Model MLP
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_labels_df['tvmonitor'].unique()), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Trenowanie modelu na danych treningowych
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_data_folder,
    x_col="filename",
    y_col="label",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="sparse",
    target_size=(224, 224),
    validate_filenames=False
)

model.fit(train_generator, epochs=10)

# Klasyfikacja danych testowych
predictions = model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = test_generator.labels

# Etykiety na odpowiadające im nazwy
class_names = train_labels_df['window'].unique()
class_names_dict = dict(zip(range(len(class_names)), class_names))

predicted_labels_names = [class_names_dict[label] for label in predicted_labels]
true_labels_names = [class_names_dict[label] for label in true_labels]


results_df = pd.DataFrame({'Filename': test_df['filename'], 'True Label': true_labels_names, 'Predicted Label': predicted_labels_names})
#x=0
#w=0
#for result in predicted_labels_names:
  #  if true_labels_names[x]==result:
  #      w=w+1
 #   x=x+1
#print(x)
#print(w)
#poprawnosc=(w*100)/x
#print("Poprawność: "+str(poprawnosc) +"%")
x=0
tp=0
tn=0
fp=0
fn=0
for i in predicted_labels_names:
    if true_labels_names[x]==1 and i==1:
        tp=tp+1
    if true_labels_names[x]==0 and i==0:
        tn=tn+1
    if true_labels_names[x]==0 and i==1:
        fp=fp+1
    if true_labels_names[x]==1 and i==0:
        fn=fn+1
    x=x+1
dok_ac=((tp+tn)*100)/(tp+tn+fp+fn)
tp=tp/6
tn=tn/6
fp=fp/6
fn=fn/6
print(f'Dokładność true positive: {tp:.2f}%')
print(f'Dokładność false positive: {fp:.2f}%')
print(f'Dokładność true negative: {tn:.2f}%')
print(f'Dokładność false negative: {fn:.2f}%')
print(f'Dokładność modelu accuracy: {dok_ac:.2f}%')