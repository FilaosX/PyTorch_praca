import torch
import zipfile
import pickle

# Ścieżka do pliku .pth
sciezka_do_pliku = r"H:\net_latest_CLIP.pth"


def load_prompt_learner_from_zip(zip_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_contents = zip_ref.namelist()

            # Sprawdzamy, czy plik `.pkl` jest w archiwum
            if 'archive/data.pkl' in zip_contents:
                with zip_ref.open('archive/data.pkl') as pkl_file:
                    data = pickle.load(pkl_file)
                    if 'PromptLearner' in data:
                        prompt_learner_state = data['PromptLearner']
                        if 'ctx' in prompt_learner_state:
                            ctx_data = prompt_learner_state['ctx']
                            return ctx_data
                        else:
                            raise KeyError("Klucz 'ctx' nie istnieje w stanie PromptLearner")
                    else:
                        raise KeyError("Klucz 'PromptLearner' nie istnieje w pliku .pkl")
            else:
                raise FileNotFoundError("Plik 'archive/data.pkl' nie istnieje w archiwum ZIP")
    except Exception as e:
        print("Wystąpił błąd podczas wczytywania:", str(e))


# Przykładowe użycie
ctx_data = load_prompt_learner_from_zip(sciezka_do_pliku)
if ctx_data is not None:
    print("Dane ctx wczytane pomyślnie.")
    # Teraz możemy użyć wczytanych danych ctx
    # self.prompt_learner.ctx.data = ctx_data
else:
    print("Błąd wczytywania danych ctx.")
