import csv

def main(input_file, output_file):
    # Otwieramy plik wejściowy w trybie do odczytu
    with open(input_file, 'r', newline='') as csv_in_file:
        # Otwieramy plik wyjściowy w trybie do zapisu
        with open(output_file, 'w', newline='') as csv_out_file:
            # Tworzymy czytnik pliku CSV dla pliku wejściowego
            csv_reader = csv.reader(csv_in_file)
            # Tworzymy pisarza pliku CSV dla pliku wyjściowego
            csv_writer = csv.writer(csv_out_file)

            # Iterujemy przez wiersze pliku wejściowego
            for row in csv_reader:
                # Zapisujemy tylko pierwsze dwie kolumny
                new_row = row[:2]
                # Zapisujemy wiersz do pliku wyjściowego
                csv_writer.writerow(new_row)

if __name__ == "__main__":
    # Nazwa pliku wejściowego
    input_file = "meta_info_KonIQ10kDataset.csv"
    # Nazwa pliku wyjściowego
    output_file = "meta_info_KonIQ10kDataset_nowy.csv"
    # Wywołujemy funkcję główną
    main(input_file, output_file)
