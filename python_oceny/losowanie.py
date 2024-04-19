import random

# Procent liczb do wylosowania (np. 0.8 dla 80%)
sample_percent = 0.6

# Zakres liczb
start_num = 1
end_num = 5135

# Obliczenie liczby liczb do wylosowania
sample_size = int((end_num - start_num + 1) * sample_percent)

# Wylosowanie losowych liczb
random_numbers = random.sample(range(start_num, end_num + 1), sample_size)

# Wybór niewylosowanych liczb
remaining_numbers = [num for num in range(start_num, end_num + 1) if num not in random_numbers]

# Zapis losowych liczb do tablicy
selected_numbers = random_numbers
sorted_numbers = sorted(selected_numbers)

# Zapis niewylosowanych liczb do drugiej tablicy
unselected_numbers = remaining_numbers

# Wyświetlenie tablic
print("Wylosowane liczby:", sorted_numbers)
print("Niewylosowane liczby:", unselected_numbers)
