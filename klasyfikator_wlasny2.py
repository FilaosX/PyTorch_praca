import numpy as np
import pandas as pd
import pickle
from PIL import Image

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

def utworz_tablice(x):
    wymiary = (x, 512, 512, 3)
    print(x)
    tablica = np.zeros(wymiary, dtype=np.uint8)
    return tablica


# Przykład użycia:
# Przykład użycia:
file_path = 'train_split_info/los1mini.pkl'
train_images, val_images, test_images = load_images_from_pickle(file_path)

print(f"Liczba zdjęć trenujących: {len(train_images)}")
print(f"Liczba zdjęć walidujących: {len(val_images)}")
print(f"Liczba zdjęć testujących: {len(test_images)}")
csv_file_path = 'meta_info/oceny_zdjec_mini.csv'
tablica_obrazow = utworz_tablice(len(train_images))
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

ilosc_pikseli=512*512
ilosc_obrazow=len(image_train_values)
print(ilosc_obrazow)
for o in range(0, ilosc_obrazow):
    print('obraz wczytany '+str(o))
    sciezka_do_zdjecia = f"zdjecia_uczenie_mini/{image_train_names[o]}"
    for i in range(0, 5): #512
        for a in range(0, 12): #512
            obraz = Image.open(sciezka_do_zdjecia)
            kolor_piksela = obraz.getpixel((i, a))
            #kolor_c=kolor_piksela[0]+kolor_piksela[1]+kolor_piksela[2]
            tablica_obrazow[o][i][a][0]=kolor_piksela[0]
            tablica_obrazow[o][i][a][1] = kolor_piksela[1]
            tablica_obrazow[o][i][a][2] = kolor_piksela[2]

print('utworzono')
rozmiar = 512
macierz1 = [[0] * rozmiar for _ in range(rozmiar)]
macierz2 = [[0] * rozmiar for _ in range(rozmiar)]
macierz3 = [[0] * rozmiar for _ in range(rozmiar)]
macierz4 = [[0] * rozmiar for _ in range(rozmiar)]
macierz5 = [[0] * rozmiar for _ in range(rozmiar)]
macierzt1 = [[0] * rozmiar for _ in range(rozmiar)]
macierzt2 = [[0] * rozmiar for _ in range(rozmiar)]
macierzt3 = [[0] * rozmiar for _ in range(rozmiar)]
macierzt4 = [[0] * rozmiar for _ in range(rozmiar)]
macierzt5 = [[0] * rozmiar for _ in range(rozmiar)]



for o in range(0, ilosc_obrazow):
    print('obraz do analizy '+str(o))
    sciezka_do_zdjecia = f"zdjecia_uczenie_mini/{image_train_names[o]}"
    etykieta = sciezka_do_zdjecia[26]
    #print(f"Kolor: {kolor_piksela[0]}")
    for i in range(0, 512): #512
        for a in range(0, 512): #512
            kolor_c=(tablica_obrazow[o][i][a][0]+tablica_obrazow[o][i][a][1]+tablica_obrazow[o][i][a][2])/3
            #print('k1 '+str(tablica_obrazow[o][i][a][0]))
            #print('k2 ' + str(tablica_obrazow[o][i][a][1]))
            #print('kk ' + str(tablica_obrazow[o][i][a][2]))
            #print(kolor_c)
            #if kolor_c > 384:
          #      kolor_b = 1
                #print(kolor_b)
          #  else:
          #      kolor_b = 0
            if int(etykieta)==1:
                macierz1[i][a]=macierz1[i][a]+kolor_c
            if int(etykieta)==2:
                macierz2[i][a]=macierz2[i][a]+kolor_c
            if int(etykieta)==3:
                macierz3[i][a]=macierz3[i][a]+kolor_c
            if int(etykieta)==4:
                macierz4[i][a]=macierz4[i][a]+kolor_c
            if int(etykieta)==5:
                macierz5[i][a]=macierz5[i][a]+kolor_c
wiersz1 = macierz1[0]
print(wiersz1)
print('nastepny')
wiersz2 = macierz2[0]
print(wiersz2)
wiersz3 = macierz3[0]
print(wiersz2)
wiersz4 = macierz4[0]
print(wiersz2)
wiersz5 = macierz5[0]
print(wiersz2)
suma1=0
suma2=0
suma3=0
suma4=0
suma5=0
sumat1=0
sumat2=0
sumat3=0
sumat4=0
sumat5=0

for i in range(0, 512): #512
    #print('wiersz_standard'+str(i))
    for a in range(0, 512): #512
        macierzt1[i][a] = macierz1[i][a] / ilosc_obrazow
        macierzt2[i][a] = macierz2[i][a] / ilosc_obrazow
        macierzt3[i][a] = macierz3[i][a] / ilosc_obrazow
        macierzt4[i][a] = macierz4[i][a] / ilosc_obrazow
        macierzt5[i][a] = macierz5[i][a] / ilosc_obrazow

nazwa_pliku1 = "moja_macierz1.txt"
nazwa_pliku2 = "moja_macierz2.txt"
nazwa_pliku3 = "moja_macierz3.txt"
nazwa_pliku4 = "moja_macierz4.txt"
nazwa_pliku5 = "moja_macierz5.txt"


# Zapisz macierz do pliku txt
np.savetxt(nazwa_pliku1, macierzt1, fmt='%d', delimiter=' ')
np.savetxt(nazwa_pliku2, macierzt2, fmt='%d', delimiter=' ')
np.savetxt(nazwa_pliku3, macierzt3, fmt='%d', delimiter=' ')
np.savetxt(nazwa_pliku4, macierzt4, fmt='%d', delimiter=' ')
np.savetxt(nazwa_pliku5, macierzt5, fmt='%d', delimiter=' ')
#print(f"Macierz 512x512 została zapisana do pliku: {nazwa_pliku}")

'''
ilosc_obrazow_test=len(image_test_values)
for o in range(0, ilosc_obrazow_test):
    sciezka_do_zdjecia = f"zdjecia_uczenie_mini/{image_test_names[o]}"
    # print(sciezka_do_zdjecia)
    # print(sciezka_do_zdjecia[26])
    etykieta = sciezka_do_zdjecia[26]
    # if int(etykieta)==1:
    #    print(etykieta)
    obraz = Image.open(sciezka_do_zdjecia)
    for i in range(0, 5):  # 512
        for a in range(0, 12):  # 512
            # Pobierz kolor pierwszego piksela
            kolor_piksela = obraz.getpixel((i, a))
            kolor_c = kolor_piksela[0] + kolor_piksela[1] + kolor_piksela[2]
            if kolor_c > 384:
                kolor_b = 1
                # print(kolor_b)
            else:
                kolor_b = 0
            sumat1 = sumat1 + kolor_b * macierzt1[i][a]
            sumat2 = sumat2 + kolor_b * macierzt2[i][a]
            sumat3 = sumat3 + kolor_b * macierzt3[i][a]
            sumat4 = sumat4 + kolor_b * macierzt4[i][a]
            sumat5 = sumat5 + kolor_b * macierzt5[i][a]
    #print(etykieta)
    #print("1 et "+str(sumat1))
    #print("2 et " + str(sumat2))
    #print("3 et " + str(sumat3))
    #print("4 et " + str(sumat4))
    #print("5 et " + str(sumat5))
'''
