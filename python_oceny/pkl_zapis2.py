import pickle
import random

# Zakres liczb
start_num = 1
end_num = 5134

# Obliczenie liczby liczb do wylosowania dla każdej tablicy
total_size = end_num - start_num + 1
train_size = int(total_size * 0.8)
val_size = total_size - train_size
test_size = 0

# Wylosowanie losowych liczb
numbers = list(range(start_num, end_num + 1))
random.shuffle(numbers)

# Podział na tablice
train_numbers = numbers[:train_size]
val_numbers = numbers[train_size:train_size + val_size]
test_numbers = numbers[train_size + val_size:]
train_numbers=sorted(train_numbers)
val_numbers=sorted(val_numbers)
test_numbers=sorted(test_numbers)
# Wyświetlenie rozmiarów tablic
print("Rozmiar tablicy treningowej:", len(train_numbers))
print("Rozmiar tablicy walidacyjnej:", len(val_numbers))
print("Rozmiar tablicy testowej:", len(test_numbers))
print("trenowanie:", train_numbers)
print("testowanie:", test_numbers)
print("Walidacja:", val_numbers)
# Dane, które chcemy zapisać do pliku .pkl
data = {1: {'train': train_numbers, 'val': val_numbers, 'test': test_numbers}}
# Ścieżka do pliku .pkl
output_file = 'podzial_zbioru_moja_baza.pkl'

# Zapis danych do pliku .pkl
with open(output_file, 'wb') as file:
    pickle.dump(data, file)

print("Plik .pkl został pomyślnie wygenerowany.")
