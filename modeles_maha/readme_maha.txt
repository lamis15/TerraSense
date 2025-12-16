# Modèle Air Quality - Random Forest
# Exporté le : 2025-12-16 15:10:21

## 📁 FICHIERS EXPORTÉS

1. **random_forest_maha.pkl**
   - Modèle Random Forest entraîné
   - Test Accuracy: 81.98%
   
2. **scaler_maha.pkl**
   - StandardScaler pour normalisation des données
   - Moyenne = 0, Écart-type = 1

3. **config_maha.json**
   - Configuration complète du modèle
   - Features sélectionnées
   - Paramètres du modèle
   - Métriques de performance

4. **imputation_stats_maha.json**
   - Statistiques pour imputation des valeurs manquantes
   - Moyennes, écarts-types, min, max de chaque colonne

5. **readme_maha.txt**
   - Ce fichier de documentation

## 🎯 VARIABLE CIBLE
- **CO(GT)** : Concentration de CO
- Classes : 0 (Bas), 1 (Moyen), 2 (Élevé)

## 📊 FEATURES UTILISÉES (9 colonnes)
  - PT08.S1(CO)
  - NMHC(GT)
  - C6H6(GT)
  - PT08.S2(NMHC)
  - NOx(GT)
  - PT08.S3(NOx)
  - NO2(GT)
  - PT08.S4(NO2)
  - PT08.S5(O3)

## 🔄 UTILISATION DU MODÈLE

```python
import joblib
import pandas as pd
import numpy as np
import json

# 1. Charger le modèle, le scaler et la config
model = joblib.load('modeles_maha/random_forest_maha.pkl')
scaler = joblib.load('modeles_maha/scaler_maha.pkl')

with open('modeles_maha/config_maha.json', 'r') as f:
    config = json.load(f)
    
selected_features = config['selected_features']

# 2. Préparer vos nouvelles données
# Les colonnes doivent correspondre aux features : PT08.S1(CO), NMHC(GT), C6H6(GT)...
new_data = pd.DataFrame({
    # Exemple de données
    'PT08.S1(CO)': [1200],
    'NMHC(GT)': [150],
    'C6H6(GT)': [10.5],
    # ... autres features
})

# 3. S'assurer que les colonnes sont dans le bon ordre
new_data = new_data[selected_features]

# 4. Normaliser les données
new_data_scaled = scaler.transform(new_data)

# 5. Faire la prédiction
predictions = model.predict(new_data_scaled)
print(f"Prédiction : {predictions[0]}")  # 0 (Bas), 1 (Moyen), 2 (Élevé)

# 6. Obtenir les probabilités de chaque classe
probabilities = model.predict_proba(new_data_scaled)
print(f"Probabilités : Bas={probabilities[0][0]:.2%}, Moyen={probabilities[0][1]:.2%}, Élevé={probabilities[0][2]:.2%}")
```

## 📈 PERFORMANCES
- Train Accuracy: 83.62%
- Test Accuracy: 81.98%
- Écart (overfitting): 1.64%

## ⚙️ PARAMÈTRES DU MODÈLE
- n_estimators: 80
- max_depth: 6
- min_samples_split: 15
- min_samples_leaf: 8
- max_features: 0.6
- random_state: 42

## 📝 NOTES IMPORTANTES
- Le modèle nécessite exactement 9 features dans l'ordre spécifié
- Les données doivent être normalisées avec le scaler fourni
- Les prédictions sont des classes : 0 (Bas), 1 (Moyen), 2 (Élevé)
