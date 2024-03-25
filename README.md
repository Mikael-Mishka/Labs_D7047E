# D7047E - Labbar

---

## Labbar

Varje lab är i sitt egen map och innehåller en README.md fil som beskriver uppgiften.
Samt en `Main.py` fil som är där uppgiften körs.

- [Lab 0](./Lab%200)
- [Lab 1](./Lab%201)
- [Lab 2](./Lab%202)
- [Lab 3](./Lab%203)

---
# Test kod

---

Skapa en ny mapp på samma nivå
som alla Lab filer som heter `Test`.
I denna fil kan du göra egna tester
utan att påverka din aktiva branch på
git/github vid commits.

---
# Lab Setup Guide

---

Följ stegen för att se till att vi har samma 
versioner av python biblotek på våran dator.

## Skapa ett nytt Conda Environment

1. **Skapa the Environmentet:**

   ```bash
   conda create --name D7047E python
   ```

2. **Aktivera Environmentet:**

   ```bash
   conda activate D7047E
   
3. **Installera `requirements.txt` packages:**

   ```bash
    pip install -r requirements.txt

4. **Testa köra en main fil i en lab mapp:**
   
   ```bash
   python Test.py
   ```
   Om ovan inte fungerar, prova att köra:
   ```bash
   python3 Test.py
   ```

5. **Konfigurera din Python interpreter i din IDE:**
   För PyCharm borde följande gälla i senare versioner:
   ```angular2html
    File -> Settings -> Project Labs D7047E 
    -> Python Interpreter -> Add interpreter
    -> Add local interpreter -> Conda Environment
    -> Use Existing Environment
    ```
   Völj `D7047E` som environmentet.
6. **Klar! (förhoppningsvis)**