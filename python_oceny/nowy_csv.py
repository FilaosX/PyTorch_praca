import csv


def modify_csv(input_csv_file, output_csv_file):
    # Otwarcie pliku CSV do odczytu
    with open(input_csv_file, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = list(csv_reader)

    # Przygotowanie danych do zapisu do nowego pliku CSV
    modified_data = []
    for row_idx, row in enumerate(data):
        modified_row = []
        for col_idx, value in enumerate(row):
            modified_value = f"zzzzz{value}{row_idx},{value}"
            modified_row.append(modified_value)
        modified_data.append(modified_row)

    # Zapisanie danych do nowego pliku CSV
    with open(output_csv_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(modified_data)

    print("Plik CSV został przetworzony i zapisany")


# Nazwy plików wejściowego i wyjściowego
input_csv_file = 'oceny_cha_skala.csv'
output_csv_file = 'oceny_cha_skala3.csv'

# Wywołanie funkcji do modyfikacji
modify_csv(input_csv_file, output_csv_file)
