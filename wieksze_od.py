import pandas as pd

def main():
    # Wprowadź ścieżkę do pliku CSV
    file_path = 'konikq_oceny.csv'

    # Wprowadź nazwę drugiej kolumny
    column2_name = 'mos'

    # Wczytaj dane z pliku CSV
    df = pd.read_csv(file_path)

    # Wybierz wiersze, gdzie druga kolumna ma wartość większą niż 100
    selected_rows = df[df[column2_name] > 4]

    # Wypisz wybrane wiersze
    print(selected_rows)

if __name__ == "__main__":
    main()
