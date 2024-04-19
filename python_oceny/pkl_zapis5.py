import pickle

# Dane, które chcemy zapisać do pliku .pkl
data = {1: {'train': [0, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 19, 21, 22, 24, 28, 29, 30, 31, 32, 33, 34, 35, 38, 39, 41, 43, 44, 45, 47, 48, 49, 50, 52, 53, 54, 55, 56, 58, 59, 62, 63, 64, 68, 69, 70, 72, 73, 74, 75, 79, 80, 140], 'val': [20, 25, 26, 27, 36, 40, 57, 65, 96, 97, 99, 105, 106, 135], 'test': [1, 9, 10, 11, 18, 23, 37, 42]}}


# Ścieżka do pliku .pkl
output_file = 'los1.pkl'

# Zapis danych do pliku .pkl
with open(output_file, 'wb') as file:
    pickle.dump(data, file)

print("Plik .pkl został pomyślnie wygenerowany.")
