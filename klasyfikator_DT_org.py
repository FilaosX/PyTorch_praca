import os
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Ścieżka do foldera z danymi treningowymi
train_data_folder = "Sekwencje/Seq8/F107_RGB_train"
num_train_images = 800

# Ścieżka do foldera z danymi testowymi
test_data_folder = "Sekwencje/Seq8/F107_RGB_test"
num_test_images = 600

# Etykiety z pliku CSV dla danych treningowych
train_labels_path = "output_train_lab3.csv"
train_labels_df = pd.read_csv(train_labels_path)

# Etykiety z pliku CSV dla danych testowych
test_labels_path = "output_test_lab3.csv"
test_labels_df = pd.read_csv(test_labels_path)

# Tworzenie zbioru z nazwami plików i etykietami dla danych treningowych
train_image_list = [f"img_{i}.png" for i in range(0, num_train_images)]
train_df = pd.DataFrame({'filename': train_image_list, 'label': train_labels_df.iloc[:, 2].astype(str)})
print(len(train_image_list))
# Tworzenie zbioru z nazwami plików i etykietami dla danych testowych
test_image_list = [f"img_{i}.png" for i in range(0, num_test_images)]
test_df = pd.DataFrame({'filename': test_image_list, 'label': test_labels_df.iloc[:, 2].astype(str)})

# Etykiety
le = LabelEncoder()
train_df['label'] = le.fit_transform(train_df['label'])
test_df['label'] = le.transform(test_df['label'])

# Tworzenie generatora danych treningowych
datagen = ImageDataGenerator(rescale=1./255)

# Dane treningowe do numpy array
X_train = []
for img_name in train_df['filename']:
    img_path = os.path.join(train_data_folder, img_name)
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    X_train.append(img_array)

X_train = np.array(X_train)
y_train = np.array(train_df['label'])


# Dane testowe do numpy array
X_test = []
for img_name in test_df['filename']:
    img_path = os.path.join(test_data_folder, img_name)
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    X_test.append(img_array)

X_test = np.array(X_test)
y_test = np.array(test_df['label'])

# Trenowanie modelu drzewa decyzyjnego
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train.reshape(-1, 224*224*3), y_train)

# Klasyfikacja na danych testowych
y_pred = model.predict(X_test.reshape(-1, 224*224*3))

# Obliczenie dokładności
x=0
tp=0
tn=0
fp=0
fn=0
for i in y_pred:
    if y_test[x]==1 and i==1:
        tp=tp+1
    if y_test[x]==0 and i==0:
        tn=tn+1
    if y_test[x]==0 and i==1:
        fp=fp+1
    if y_test[x]==1 and i==0:
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
