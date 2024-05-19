import os
from PIL import Image

def resize_images_in_folder(input_folder, output_folder, target_size):
    # Utworzenie folderu wyjściowego, jeśli nie istnieje
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Przechodzenie przez wszystkie pliki w folderze wejściowym
    for filename in os.listdir(input_folder):
        # Pełna ścieżka do pliku wejściowego
        input_path = os.path.join(input_folder, filename)

        # Pełna ścieżka do pliku wyjściowego
        output_path = os.path.join(output_folder, filename)

        # Przeskalowanie obrazu
        resize_image(input_path, output_path, target_size)

def resize_image(input_path, output_path, target_size):
    # Wczytanie obrazu
    image = Image.open(input_path)

    # Przeskalowanie obrazu
    resized_image = image.resize(target_size)

    # Zapis przeskalowanego obrazu
    resized_image.save(output_path)

    print(f"Przeskalowano obraz: {input_path} -> {output_path}")


# Folder wejściowy z oryginalnymi obrazami
input_folder = 'H:\Politechnika rzeszowska\8 semestr\Projekt badawczy\IQA-PyTorch-main\datasets\ChallengeDB_release\ChallengeDB_release\Images'

# Folder wyjściowy, gdzie będą zapisane przeskalowane obrazy
output_folder = 'H:\Politechnika rzeszowska\8 semestr\Projekt badawczy\IQA-PyTorch-main\datasets\ChallengeDB_release\ChallengeDB_release\Images2'

# Docelowy rozmiar przeskalowanych obrazów
target_size = (512, 512)

# Wywołanie funkcji do przeskalowania obrazów w folderze
resize_images_in_folder(input_folder, output_folder, target_size)

