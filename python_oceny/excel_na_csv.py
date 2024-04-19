import pandas as pd


def excel_to_csv(input_file, output_file):
    # Wczytanie pliku Excel
    df = pd.read_excel(input_file)

    # Zapisanie do pliku CSV
    df.to_csv(output_file, index=False)

    print("Plik Excel został przekształcony na plik CSV")


# Nazwy plików wejściowego i wyjściowego
input_excel_file = 'output_ch2.xlsx'
output_csv_file = 'oceny_ch.csv'

# Wywołanie funkcji do przekształcenia
excel_to_csv(input_excel_file, output_csv_file)
