def wypisz_zawartosc_pliku_pth(nazwa_pliku):
    try:
        with open(nazwa_pliku, "r", encoding="utf-8") as plik:
            zawartosc = plik.readlines()
            print("Zawartość pliku '{}':".format(nazwa_pliku))
            for linia in zawartosc:
                print(linia.strip())  # Usuwamy białe znaki na końcu linii
    except FileNotFoundError:
        print("Plik '{}' nie istnieje.".format(nazwa_pliku))
    except Exception as e:
        print("Wystąpił błąd podczas odczytu pliku:", str(e))

# Przykładowe użycie funkcji
nazwa_pliku_pth = "net_latest.pth"
wypisz_zawartosc_pliku_pth(nazwa_pliku_pth)


