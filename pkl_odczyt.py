import pickle

# Ścieżka do pliku .pkl
input_file = 'podzial_zbioru_moja_baza.pkl'
input_file2 = 'podzial_zbioru_tid2008_m.pkl'
# Odczyt danych z pliku .pkl
with open(input_file, 'rb') as file:
    data = pickle.load(file)

# Wyświetlenie odczytanych danych
print(data)
with open(input_file2, 'rb') as file:
    data = pickle.load(file)

# Wyświetlenie odczytanych danych
print(data)