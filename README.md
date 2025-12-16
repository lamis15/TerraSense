# TerraSense – Recommandation de Cultures (Domaine Agriculture)

Projet web pour le domaine **Agriculture** du système TerraSense (Agriculture – Qualité de l’air – Qualité de l’eau), avec un thème **vert** et une API Flask pour la recommandation de cultures.

---

## 1. Structure du projet

```text
TerraSense/
│
├── app.py                # Backend Flask (API /predict + page web)
├── index.html            # Interface web verte avec 3 sections
├── requirements.txt      # Dépendances Python
├── Procfile              # Pour Heroku / Render (gunicorn)
│
├── models/               # (à créer) modèles sauvegardés depuis le notebook
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── bounds.pkl        # optionnel
│
└── README.md
```

> **Important** : le dossier `models` doit être créé par vous, et vous devez y déposer les fichiers `.pkl` exportés depuis votre notebook `recommendation_modeling.ipynb`.

---

## 2. Exporter les modèles depuis le notebook

Dans votre notebook (là où le modèle Random Forest est déjà entraîné), ajoutez une cellule comme ceci :

```python
import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(rf_model, "models/random_forest_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

# Si vous avez des bornes (min/max) utilisées pour le clipping
# bounds = {...}  # votre dictionnaire de bornes
joblib.dump(bounds, "models/bounds.pkl")
```

Ensuite, copiez le dossier `models` (et son contenu `.pkl`) dans le même dossier que `app.py` et `index.html` (ici `C:\Users\MSI\Desktop\ML`).

---

## 3. Installation en local

Dans un terminal PowerShell, placez-vous dans le dossier du projet :

```powershell
cd C:\Users\MSI\Desktop\ML

python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Assurez‑vous que le dossier `models/` avec les `.pkl` est présent.

---

## 4. Lancer l’application en local

Toujours dans l’environnement virtuel :

```powershell
python app.py
```

Puis ouvrez votre navigateur sur :

- `http://localhost:5000`

La page TerraSense s’affiche avec :

- **Section Agriculture** : formulaire + prédiction de culture via `/predict`
- **Section Qualité de l’air** : carte explicative (placeholder pour futur modèle)
- **Section Qualité de l’eau** : carte explicative (placeholder pour futur modèle)

---

## 5. Déploiement (exemple Render)

1. Créez un dépôt GitHub avec ces fichiers (`app.py`, `index.html`, `requirements.txt`, `Procfile`, `models/...`).
2. Allez sur `https://render.com` et créez un compte.
3. Cliquez sur **New +** → **Web Service**.
4. Connectez votre dépôt GitHub.
5. Réglez :
   - **Build Command** : `pip install -r requirements.txt`
   - **Start Command** : `gunicorn app:app`
6. Déployez et récupérez l’URL publique générée par Render.

L’interface `index.html` est servie par Flask (`/`), et l’API `/predict` est utilisée par le JavaScript de la page.

---

## 6. Déploiement (exemple PythonAnywhere)

1. Créez un compte sur `https://www.pythonanywhere.com`.
2. Uploadez tous les fichiers du projet (y compris `models/`) dans votre espace.
3. Créez une nouvelle **Web app** → **Flask**.
4. Pointez la web app vers `app.py` (WSGI) et installez les modules de `requirements.txt` dans votre environnement PythonAnywhere.
5. Rechargez la web app et ouvrez l’URL fournie.

---

## 7. Personnalisation / Rapport

- Le thème est majoritairement **vert**, avec des touches rappelant l’environnement (air, eau).
- Le domaine principal (Agriculture) est **fonctionnel** avec appel au modèle.
- Les domaines « Qualité de l’air » et « Qualité de l’eau » sont présentés comme panneaux informatifs harmonisés, que vous pouvez connecter à d’autres notebooks / modèles si besoin.

Vous pouvez adapter les textes pour les aligner avec votre rapport TerraSense (contexte, données utilisées, métriques, etc.).



















