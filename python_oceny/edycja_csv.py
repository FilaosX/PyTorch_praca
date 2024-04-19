import csv


def add_underscore_to_sixth_position(input_csv_file, output_csv_file):
    # Otwarcie pliku CSV do odczytu
    with open(input_csv_file, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = list(csv_reader)

    # Dodanie znaku '_' na 6 pozycji w pierwszej kolumnie każdego wiersza
    for row in data:
        if row and len(
                row[0]) >= 6:  # Sprawdzenie, czy wiersz nie jest pusty i czy pierwsza kolumna ma co najmniej 6 znaków
            row[0] = row[0][:6] + '_' + row[0][6:]  # Wstawienie znaku '_' na 6 pozycji

    # Zapisanie zmienionych danych do nowego pliku CSV
    with open(output_csv_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)

    print("Plik CSV został przetworzony i zapisany")


# Nazwy plików wejściowego i wyjściowego
input_csv_file = 'meta_info_KonIQ10kDataset_skala_nazwa.csv'
output_csv_file = 'meta_info_KonIQ10kDataset_skala_nazwa_edycja.csv'

# Wywołanie funkcji do modyfikacji
add_underscore_to_sixth_position(input_csv_file, output_csv_file)
