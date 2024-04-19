import torch

# Ścieżka do pliku .pth
path = "H:\Politechnika rzeszowska\8 semestr\Projekt badawczy\p_best.pth"

# Odczytanie zawartości pliku .pth
checkpoint = torch.load(path)

# Wyświetlenie dostępnych kluczy
print("Dostępne klucze:")
for key in checkpoint.keys():
    print(key)

# Odczytanie wartości dla konkretnego klucza
# Przykład: klucz 'model'
if 'model' in checkpoint:
    model = checkpoint['model']
    print("Wartość dla klucza 'model':")
    print(model)
