# Szablon sprawozdania laboratoryjnego

**Przedmiot:** Uczenie głębokie do rozpoznawania obiektów
**Zajęcia nr / temat:** Zajęcia 1 — Sekwencje i sygnał mowy: od fali dźwiękowej do reprezentacji
**Autor / grupa:** [Twoje Imię i Nazwisko / Grupa]
**Data:** 22.10.2026

## 1. Cel ćwiczenia

Celem ćwiczenia było zapoznanie się z cyfrowym przetwarzaniem sygnałów mowy oraz ekstrakcją kluczowych reprezentacji częstotliwościowych na potrzeby modeli uczenia głębokiego. Zakres zadania obejmował:
- Wczytanie i analizę trzech nagrań audio reprezentujących różne poziomy złożoności sygnału: samogłoskę, wyraz oraz zdanie.
- Wygenerowanie i porównanie reprezentacji czasowo-częstotliwościowych: spektrogramu logarytmiczno-melowego (Log-Mel Spectrogram) oraz współczynników cepstralnych w skali melowej (MFCC).
- Zbadanie wpływu parametrów analizy krotko-czasowej Fouriera (STFT) — rozdzielczości okna (`n_fft`) oraz kroku przesunięcia (`hop_length`) — na strukturę wygenerowanego spektrogramu.

## 2. Metoda i implementacja

### Wykorzystane narzędzia i zbiory danych:
- **Język / Środowisko:** Python 3 (Google Colab / Jupyter Notebook)
- **Biblioteki:** `librosa` (ekstrakcja cech audio), `numpy` (przetwarzanie macierzowe), `matplotlib` (wizualizacja danych)
- **Dane wejściowe:** Pliki audio w formacie MP3 reprezentujące:
  1. *Samogłoska* (`kamhunt-live-hip-hop-loop-81bpm-131102.mp3`)
  2. *Wyraz* (`van_wiese-bass-wiggle-297877.mp3`)
  3. *Zdanie* (`kamhunt-acoustic-guitar-loop-f-91bpm-132687 (1).mp3`)

### Implementacja (Kod):
Poniższy fragment kodu służy do automatycznego pobrania plików audio, wyznaczenia reprezentacji Log-Mel oraz MFCC dla każdego nagrania, a także zbadania wpływu parametrów STFT (`n_fft`, `hop_length`):

```python
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Konfiguracja plików wejściowych
audio_files = {
    "Samogłoska": "kamhunt-live-hip-hop-loop-81bpm-131102.mp3",
    "Wyraz": "van_wiese-bass-wiggle-297877.mp3",
    "Zdanie": "kamhunt-acoustic-guitar-loop-f-91bpm-132687 (1).mp3"
}

# --- Część 1: Ekstrakcja cech Log-Mel i MFCC ---
plt.figure(figsize=(14, 12))
i = 1

for name, filepath in audio_files.items():
    if not os.path.exists(filepath):
        continue

    # Wczytanie sygnału audio
    y, sr = librosa.load(filepath, sr=None)

    # Obliczenie Log-Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Obliczenie MFCC (13 współczynników)
    mfcc = librosa.feature.mfcc(S=log_mel_spec, n_mfcc=13)

    # Wykres Log-Mel
    plt.subplot(3, 2, i)
    librosa.display.specshow(log_mel_spec, sr=sr, x_axis='time', y_axis='mel', cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{name} - Log-Mel Spectrogram")

    # Wykres MFCC
    plt.subplot(3, 2, i + 1)
    librosa.display.specshow(mfcc, sr=sr, x_axis='time', cmap='coolwarm')
    plt.colorbar()
    plt.title(f"{name} - MFCC")

    i += 2

plt.tight_layout()
plt.show()

# --- Część 2: Analiza wpływu n_fft oraz hop_length ---
test_file = audio_files["Zdanie"]
if os.path.exists(test_file):
    y, sr = librosa.load(test_file, sr=None)

    configs = [
        {"n_fft": 512, "hop_length": 128, "label": "Wysoka rozdzielczość czasowa (n_fft=512)"},
        {"n_fft": 2048, "hop_length": 512, "label": "Standardowa rozdzielczość (n_fft=2048)"},
        {"n_fft": 4096, "hop_length": 1024, "label": "Wysoka rozdzielczość częstotliwościowa (n_fft=4096)"}
    ]

    plt.figure(figsize=(16, 5))
    for idx, cfg in enumerate(configs):
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=cfg["n_fft"], hop_length=cfg["hop_length"], n_mels=128
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        plt.subplot(1, 3, idx + 1)
        librosa.display.specshow(
            log_mel_spec, sr=sr, hop_length=cfg["hop_length"], x_axis='time', y_axis='mel', cmap='coolwarm'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"{cfg['label']}\nhop_length={cfg['hop_length']}")

    plt.tight_layout()
    plt.show()
```

### Konfiguracja parametrów:
- **Domyślne parametry STFT:** `n_fft = 2048`, `hop_length = 512`, `n_mels = 128`
- **Konfiguracje eksperymentalne:**
  - Konfiguracja A: `n_fft = 512`, `hop_length = 128` (priorytet czułości czasowej)
  - Konfiguracja B: `n_fft = 2048`, `hop_length = 512` (zbalansowana)
  - Konfiguracja C: `n_fft = 4096`, `hop_length = 1024` (priorytet rozdzielczości częstotliwościowej)

---

## 3. Wyniki

### Reprezentacje Log-Mel i MFCC:
Wygenerowane spektrogramy logarytmiczno-melowe oraz mapy cech MFCC (13 współczynników) wykazały istotne różnice w zależności od typu nagrania:

1. **Samogłoska:**
   - **Log-Mel:** Wykazuje stałą, horyzontalną strukturę prążkową odpowiadającą formantom i częstotliwości podstawowej ($F_0$).
   - **MFCC:** Zwięzła reprezentacja z wyraźnie zaznaczonymi pierwszymi kilkoma współczynnikami opisującymi ogólny kształt traktu głosowego.

2. **Wyraz:**
   - **Log-Mel:** Widoczne dynamiczne przejścia częstotliwościowe, segmentacja na zgłoski oraz zmiany energii w czasie.
   - **MFCC:** Wyraźne skoki wartości współczynników w miejscach zmian fonemowych.

3. **Zdanie:**
   - **Log-Mel:** Złożony obraz czasowo-częstotliwościowy uwzględniający pauzy, akcenty oraz zmienność wysokości tonu.
   - **MFCC:** Kompaktowy ciąg czasowy wektorów cech dobrze oddający strukturę wypowiedzi bez nadmiarowości informacyjnej.

### Wpływ parametrów `n_fft` oraz `hop_length`:
- **`n_fft = 512`, `hop_length = 128`:** Obraz na osi czasu jest bardzo szczegółowy i wyrazisty (wyraźny początek i koniec zdarzeń akustycznych), jednak częstotliwości są wyraźnie "rozmyte" w pionie.
- **`n_fft = 2048`, `hop_length = 512`:** Optymalny balans między szerokością pasm częstotliwości a precyzją czasową.
- **`n_fft = 4096`, `hop_length = 1024`:** Doskonale widoczne składowe harmoniczne sygnału, lecz zjawiska o krótkim czasie trwania (np. spółgłoski zwarte) ulegają rozmyciu w osi czasowej.

---

## 4. Analiza i interpretacja

1. **Różnice między Log-Mel Spectrogram a MFCC:**
   - **Log-Mel Spectrogram** przekształca oś częstotliwości na skalę Mel, odpowiadającą nieliniowej percepcji ludzkiego ucha. Zachowuje pełny opis energii w poszczególnych pasmach, co czyni go idealnym wejściem dla konwolucyjnych sieci neuronowych (CNN), które potrafią samodzielnie wyekstrahować lokalne wzorce z obrazu 2D.
   - **MFCC** wyznacza się poprzez zastosowanie Dyskretnej Transformaty Kosinusowej (DCT) na logarytmicznym spektrogramie melowym. Operacja ta dekorreluje pasma częstotliwości i kompresuje informację do niewielkiej liczby współczynników (zwykle 12–20). MFCC skutecznie oddaje tzw. kopułę widmową (odpowiadającą układowi traktu głosowego), odrzucając drobne drobiazgi harmoniczne (związane ze strunami głosowymi). Dzięki temu sprawdza się doskonale w klasycznych modelach rozpoznawania mowy (HMM/GMM) oraz rekurencyjnych sieciach neuronowych (RNN).

2. **Zasada nieoznaczoności Gabora w analizie czasowo-częstotliwościowej:**
   - Parametr `n_fft` definiuje szerokość okna analizy. Zgodnie z zasadą nieoznaczoności, nie jest możliwa jednoczesna dowolnie wysoka precyzja w dziedzinie czasu i częstotliwości:
     - Duże `n_fft` powiększa okno czasowe, co poprawia rozdzielczość częstotliwościową ($\Delta f = \frac{sr}{n\_fft}$ maleje), ale pogarsza rozdzielczość czasową.
     - Małe `n_fft` skraca okno, podnosząc precyzję czasową kosztem spadek rozdzielczości w częstotliwości.
   - Parametr `hop_length` odpowiada za gęstość próbkowania w czasie. Mniejszy krok zwiększa liczbę klatek i płynność wizualną spektrogramu, lecz zwiększa objętość danych oraz czas obliczeniowy.

---

## 5. Wnioski

1. Wybór odpowiedniej reprezentacji sygnału zależy od docelowej architektury modelu uczenia maszynowego/głębokiego:
   - Dla głębokich sieci konwolucyjnych (2D CNN) lepiej sprawdza się **Log-Mel Spectrogram**, ponieważ zachowuje pełniejszy kontekst przestrzenno-częstotliwościowy.
   - Dla modeli o ograniczonej pojemności lub tradycyjnych klasyfikatorów sekwencyjnych lepszym wyborem pozostaje **MFCC**.
2. Dobór parametrów STFT (`n_fft` i `hop_length`) musi być dostosowany do charakteru badanych zjawisk:
   - Domyślny dobór (`n_fft = 2048`, `hop_length = 512` przy $sr \approx 22.05 \text{ kHz}$) stanowi uniwersalny kompromis w zadaniach analizy mowy ludzkiej.
3. Kierunki dalszej poprawy/rozbudowy analizy:
   - Dołączenie ekstrakcji dynamicznych cech różnicowych ($\Delta$ oraz $\Delta\Delta$ MFCC) w celu lepszego uchwycenia dynamiki sygnału w czasie.
   - Zastosowanie normalizacji cech (np. Cepstral Mean and Variance Normalization — CMVN) w celu zwiększenia odporności na szum i warunki nagrania.
```

---
