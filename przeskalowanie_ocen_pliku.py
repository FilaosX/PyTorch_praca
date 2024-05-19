import csv

def scale_and_round(value, min_value, max_value, new_min, new_max):
    # Skalowanie wartości do przedziału od 0 do 1
    scaled_value = (value - min_value) / (max_value - min_value)
    # Skalowanie wartości do nowego przedziału
    new_value = new_min + (new_max - new_min) * scaled_value
    # Zaokrąglenie wartości do liczby całkowitej
    rounded_value = round(new_value)
    return rounded_value

def main(input_file, output_file):
    # Otwieramy plik wejściowy w trybie do odczytu
    with open(input_file, 'r', newline='') as csv_in_file:
        # Otwieramy plik wyjściowy w trybie do zapisu
        with open(output_file, 'w', newline='') as csv_out_file:
            # Tworzymy czytnik pliku CSV dla pliku wejściowego
            csv_reader = csv.reader(csv_in_file)
            # Tworzymy pisarza pliku CSV dla pliku wyjściowego
            csv_writer = csv.writer(csv_out_file)

            # Wczytujemy pierwszy wiersz jako nagłówek
            header = next(csv_reader)
            csv_writer.writerow(header)

            # Pobieramy dane dla drugiej kolumny
            column_data = [float(row[1]) for row in csv_reader]

            # Obliczamy minimalną i maksymalną wartość dla drugiej kolumny
            min_value = min(column_data)
            max_value = max(column_data)

            # Ponownie otwieramy plik wejściowy, aby zresetować czytnik
            csv_in_file.seek(0)
            next(csv_reader)  # Pomijamy nagłówek

            # Iterujemy przez pozostałe wiersze
            for row in csv_reader:
                # Przeskalowanie i zaokrąglenie wartości w drugiej kolumnie
                scaled_value = scale_and_round(float(row[1]), min_value, max_value, 1, 5)
                # Aktualizacja wartości w drugiej kolumnie
                row[1] = str(scaled_value)
                # Zapisujemy wiersz do pliku wyjściowego
                csv_writer.writerow(row)

if __name__ == "__main__":
    # Nazwa pliku wejściowego
    input_file = "meta_info_KonIQ10kDataset_nowy.csv"
    # Nazwa pliku wyjściowego
    output_file = "meta_info_KonIQ10kDataset_skala.csv"
    # Wywołujemy funkcję główną
    main(input_file, output_file)
