import torch

# Ścieżka do pliku .pth
sciezka_do_pliku = r"H:\CLIP-IQA+_learned_prompts-603f3273.pth"

try:
    # Wczytanie pliku .pth
    checkpoint = torch.load(sciezka_do_pliku, map_location=torch.device('cpu'))

    # Sprawdzenie zawartości pliku
    if 'PromptLearner' in checkpoint:
        prompt_learner_state = checkpoint['PromptLearner']
        if isinstance(prompt_learner_state, dict) and 'ctx' in prompt_learner_state:
            ctx_data = prompt_learner_state['ctx']
            print("Dane ctx wczytane pomyślnie.")
            # Tutaj możemy przypisać dane ctx do prompt_learner.ctx.data
            # self.prompt_learner.ctx.data = ctx_data
        else:
            raise KeyError("Klucz 'ctx' nie istnieje w stanie PromptLearner")
    else:
        raise KeyError("Klucz 'PromptLearner' nie istnieje w pliku .pth")
except Exception as e:
    print("Wystąpił błąd podczas wczytywania:", str(e))
