import csv


def main(input_file, output_file):
    # Otwieramy plik wejściowy w trybie do odczytu
    with open(input_file, 'r', newline='') as csv_in_file:
        # Tworzymy czytnik pliku CSV dla pliku wejściowego
        csv_reader = csv.reader(csv_in_file)

        # Tworzymy listę przechowującą dane z drugiej kolumny
        column_data = [row[1] for row in csv_reader]

    # Otwieramy plik wyjściowy w trybie do zapisu
    with open(output_file, 'w') as txt_out_file:
        # Zapisujemy dane z drugiej kolumny do pliku tekstowego
        for item in column_data:
            txt_out_file.write("%s\n" % item)


if __name__ == "__main__":
    # Nazwa pliku wejściowego CSV
    input_file = "meta_info_KonIQ10kDataset_skala_nazwa.csv"
    # Nazwa pliku wyjściowego TXT
    output_file = "mos_skala_konikQ.txt"
    # Wywołujemy funkcję główną
    main(input_file, output_file)
