import os
import csv

# Ścieżka do folderu z zdjęciami
#folder_zdjec = "H:\Politechnika rzeszowska\8 semestr\Projekt badawczy\
#IQA-PyTorch-main\datasets\zdjecia_uczenie"
folder_zdjec = "H:\\Politechnika rzeszowska\\8 semestr\\Projekt badawczy\\IQA-PyTorch-main\\datasets\\uid2013\\distorted_images4\\24"
#folder_zdjec = "H:\\Politechnika rzeszowska\\8 semestr\\Projekt badawczy\\IQA-PyTorch-main\\datasets\\mdid\\MDID\\distortion_images2"
#folder_zdjec = "H:\\Politechnika rzeszowska\\8 semestr\\Projekt badawczy\\IQA-PyTorch-main\\datasets\\ChallengeDB_release\\ChallengeDB_release\\Images2"
#folder_zdjec = "H:\\Politechnika rzeszowska\\8 semestr\\Projekt badawczy\\IQA-PyTorch-main\\datasets\\koniq2"

# Pobranie nazw plików ze zdjęciami
nazwy_zdjec = os.listdir(folder_zdjec)

# Lista ocen
oceny = []

# Przypisanie oceny  dla każdego zdjęcia
for nazwa_zdjecia in nazwy_zdjec:
    #oceny.append({"img_name": nazwa_zdjecia, "mos": (6-int(nazwa_zdjecia[7]))})
    oceny.append({"img_name": nazwa_zdjecia, "mos": (int(nazwa_zdjecia[5]))})
    #oceny.append({"img_name": 'zzzzz'+str((int(nazwa_zdjecia[5])))+'_'+nazwa_zdjecia, "mos": (int(nazwa_zdjecia[5]))})

# Ścieżka do pliku CSV
plik_csv = r"H:\Politechnika rzeszowska\8 semestr\Projekt badawczy\IQA-PyTorch-main\datasets\oceny_zdjec_tid2013_24.csv"
#plik_csv = r"H:\Politechnika rzeszowska\8 semestr\Projekt badawczy\IQA-PyTorch-main\datasets\oceny_zdjec_tid2008.csv"
#plik_csv = r"H:\Politechnika rzeszowska\8 semestr\Projekt badawczy\IQA-PyTorch-main\datasets\oceny_zdjec_mdid.csv"
#plik_csv = r"H:\Politechnika rzeszowska\8 semestr\Projekt badawczy\IQA-PyTorch-main\datasets\oceny_zdjec_challenge.csv"
#plik_csv = r"H:\Politechnika rzeszowska\8 semestr\Projekt badawczy\IQA-PyTorch-main\datasets\oceny_zdjec_konikqx2.csv"

# Zapis ocen do pliku CSV
with open(plik_csv, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["img_name", "mos"])
    writer.writeheader()
    writer.writerows(oceny)

print("Plik CSV został wygenerowany.")

