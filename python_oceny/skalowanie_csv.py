import pandas as pd


def rescale_and_round(input_csv_file, output_csv_file):
    # Wczytanie pliku CSV
    df = pd.read_csv(input_csv_file)

    # Znalezienie maksymalnej i minimalnej wartości
    min_val = df.min().min()
    max_val = df.max().max()

    # Przeskalowanie wartości do przedziału od 1 do 5
    scaled_df = ((df - min_val) / (max_val - min_val) * 4) + 1

    # Zaokrąglenie do liczb całkowitych
    scaled_and_rounded_df = scaled_df.round().astype(int)

    # Zapisanie do pliku CSV
    scaled_and_rounded_df.to_csv(output_csv_file, index=False)

    print("Wartości w pliku CSV zostały przeskalowane i zaokrąglone do przedziału od 1 do 5")


# Nazwy plików wejściowego i wyjściowego
input_csv_file = 'oceny_ch.csv'
output_csv_file = 'oceny_cha_skala.csv'

# Wywołanie funkcji do przeskalowania i zaokrąglenia
rescale_and_round(input_csv_file, output_csv_file)
