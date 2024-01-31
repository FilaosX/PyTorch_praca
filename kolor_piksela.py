from PIL import Image

def pobierz_kolor_piksela(sciezka_do_zdjecia):
    try:
        # Otwórz obraz
        obraz = Image.open(sciezka_do_zdjecia)

        # Pobierz kolor pierwszego piksela
        kolor_piksela = obraz.getpixel((0, 1))

        # Wypisz kolor
        print(f"Kolor pierwszego piksela (lewy górny róg): {kolor_piksela}")

    except Exception as e:
        print(f"Wystąpił błąd: {e}")

if __name__ == "__main__":
    # Podaj ścieżkę do pliku ze zdjęciem
    sciezka_do_zdjecia = 'gta6.jpg'

    # Wywołaj funkcję
    pobierz_kolor_piksela(sciezka_do_zdjecia)
