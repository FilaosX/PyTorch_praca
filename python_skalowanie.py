import pandas as pd

def scale_column(data, old_min, old_max, new_min, new_max):
    scaled_data = ((data - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    return scaled_data.round().astype(int)

def main():
    # Wprowadź ścieżkę do pliku CSV
    file_path = 'img_name,mos.csv'

    # Wprowadź nazwy kolumn
    column1_name = 'img_name'
    column2_name = 'mos'

    # Wprowadź nowy przedział skalowania
    new_min = 1
    new_max = 5

    # Wczytaj dane z pliku CSV
    df = pd.read_csv(file_path)

    # Przeskaluj drugą kolumnę i zaokrąglaj do liczb całkowitych
    df[column2_name] = scale_column(df[column2_name], 0, 100, new_min, new_max)

    # Zapisz zmodyfikowane dane do nowego pliku CSV
    df.to_csv('konikq_oceny.csv', index=False)

if __name__ == "__main__":
    main()