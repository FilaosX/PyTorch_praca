import csv


def csv_to_txt(input_csv_file, output_txt_file):
    # Otwarcie pliku CSV
    with open(input_csv_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        # Otwarcie pliku TXT do zapisu
        with open(output_txt_file, 'w') as txt_file:
            # Iteracja po wierszach w pliku CSV
            for row in csv_reader:
                # Zapisanie wartości z każdej komórki do kolejnej kolumny w pliku TXT
                for value in row:
                    txt_file.write(value + '\t')  # Oddzielenie wartości tabulatorem
                txt_file.write('\n')  # Przejście do nowej linii po zapisaniu wszystkich wartości z wiersza

    print("Wartości z pliku CSV zostały zapisane w kolejnych kolumnach pliku TXT")


# Nazwy plików wejściowego i wyjściowego
input_csv_file = 'oceny_cha_skala.csv'
output_txt_file = 'mos_skala_challenge.txt'

# Wywołanie funkcji do konwersji
csv_to_txt(input_csv_file, output_txt_file)
