import pickle

# Zakres numerów zdjęć
start_num = 1
end_num = 15207
# Obliczenie liczby zdjęć dla każdego zbioru
train_size = 5135
val_size = 5372
test_size = 5000

# Tworzenie listy numerów zdjęć
numbers = list(range(start_num, end_num + 1))

# Podział na zbiory uczące, walidacyjne i testowe
train_numbers = numbers[:train_size]
val_numbers = numbers[train_size:train_size + val_size]
test_numbers = numbers[train_size + val_size:]

# Dane, które chcemy zapisać do pliku .pkl
data = {1: {'train': train_numbers, 'val': val_numbers, 'test': test_numbers}}

# Ścieżka do pliku .pkl
output_file = 'podzial_zbioru_konikq_m.pkl'

# Zapis danych do pliku .pkl
with open(output_file, 'wb') as file:
    pickle.dump(data, file)

print("Plik .pkl został pomyślnie wygenerowany.")
