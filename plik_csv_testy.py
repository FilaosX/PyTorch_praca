import os
import csv

# Ścieżka do folderu z zdjęciami
folder_zdjec = "H:\Projekt badawczy\zdjecia_do_oceny"

# Pobranie nazw plików ze zdjęciami
nazwy_zdjec = os.listdir(folder_zdjec)

# Lista ocen
oceny = []

# Przypisanie oceny 1 dla każdego zdjęcia
for nazwa_zdjecia in nazwy_zdjec:
    oceny.append({"nazwa": nazwa_zdjecia})

# Ścieżka do pliku CSV
plik_csv = "H:\Projekt badawczy\zdjecia_do_oceny\zdjecia_do_oceny.csv"

# Zapis ocen do pliku CSV
with open(plik_csv, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["nazwa"])
    writer.writeheader()
    writer.writerows(oceny)

print("Plik CSV został wygenerowany.")