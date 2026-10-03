# Projet Data Science — Diabetes 130-US Hospitals (1999–2008)

**Objectif** : aider un hôpital à cibler le suivi des patients diabétiques à risque de **réadmission en moins de 30 jours**.
- **Classification** : prédire la réadmission à la sortie (`target` = 1 si `readmitted = '<30'`).
- **Segmentation** : repérer des profils de patients (K-Means) et voir lesquels sont les plus réadmis.

**Rendu** : Blackboard, **4 octobre** — notebook détaillé, fichier de données nettoyées, présentation de 10 minutes.

---

## Où on en est

| Étape | Notebook | Responsable | État |
|---|---|---|---|
| 0. Business Understanding | `00_business_understanding.ipynb` | Membre 3 | ✅ |
| 1. Exploration | `01_exploration_organise.ipynb` | Membre 1 | ✅ (corrections mineures, voir plus bas) |
| 2a. Valeurs manquantes, qualité | `02_nettoyage_membre1.ipynb` | Membre 1 | ✅ |
| 2b. Outliers, transformations, scaling | `02b_outliers_membre2.ipynb` | Membre 2 | ✅ |
| 2c. Catégorielles, feature engineering | `03_categorielles_membre3.ipynb` | Membre 3 | ✅ |
| 3. Segmentation | à créer (`04_segmentation_membre4.ipynb`) | Membre 4 | ⬜ |
| 4. Classification | à créer (`05_classification_membre5.ipynb`) | Membre 5 | ⬜ |
| Fusion du notebook final | — | Membre 1 | ⬜ |
| Slides et répétition | — | Membre 2 | ⬜ |

## Flux des fichiers

```
diabetic_data.csv (101 766 séjours)
   → data_etape2a.csv      (69 987 séjours, 50 col.)  nettoyage, membre 1
   → data_etape2b.csv      (69 987 séjours, 56 col.)  + 6 colonnes _log, membre 2
   → data_etape2c_lisible.csv   (45 col.)  catégories en texte  → pour INTERPRÉTER
   → data_etape2c_encodee.csv   (107 col.) tout numérique       → pour MODÉLISER
```

Les fichiers `2c` se lisent avec un simple `pd.read_csv(...)`. Seuls `2a` et `2b` exigent `keep_default_na=False, na_values=['']` (pour garder `None` = examen non réalisé).

## Décisions clés

- **Cible** : `<30` = 1, le reste = 0. Déséquilibrée (**9 %** de réadmis) → recall, F1, ROC-AUC, jamais l'accuracy.
- **Un seul séjour par patient** (le premier) : évite la fuite de données entre entraînement et test.
- **Décès et soins palliatifs supprimés** (2 423 séjours) : ils ne peuvent pas être réadmis.
- **Outliers conservés**, atténués par `log1p` (colonnes `_log`) : ce sont des patients réels, souvent les plus à risque.
- **Scaling** : `StandardScaler`, **non appliqué dans les fichiers** → à placer dans le pipeline, ajusté sur l'entraînement.
- **Codes regroupés** : `*_id` par signification, `payer_code` par fréquence, diagnostics ICD-9 en 9 familles.
- **Variables créées** : `frequent_user`, `diabetes_complicated`, `n_changes`, `total_visits` (+ `_log`). Seules `n_changes` et `diabetes_complicated` apportent plus que leurs colonnes sources ; les autres servent à la lisibilité et à la segmentation.
- **Retirées** : `readmitted` (fuite de la cible), `encounter_id`, `patient_nbr`, `weight` (97 % manquant), 15 médicaments quasi jamais prescrits.
- **Signal faible** : aucune variable n'a de lien fort avec la cible (≤ 0,14). Des performances modestes sont normales, à présenter honnêtement.

---

## Ce qu'il reste à faire

**Membre 4 — Segmentation** (`data_etape2c_encodee.csv` + `data_etape2c_lisible.csv`)
- Choisir les variables : plutôt numériques et ordinales (durée, visites en `_log` ou `total_visits_log`, `age`, `n_changes`…). Éviter les 72 colonnes one-hot dans K-Means.
- `StandardScaler`, K-Means, méthode du coude + silhouette, profils des clusters (taux de réadmission par cluster).
- Interpréter avec les valeurs d'origine et le fichier lisible.
- **Rédiger la synthèse** reliant clusters et classification (quels profils sont les plus réadmis).

**Membre 5 — Classification** (`data_etape2c_encodee.csv`)
- `train_test_split(..., stratify=y)`, puis **Pipeline** : `StandardScaler` → modèle (ajusté sur l'entraînement uniquement).
- 2 à 3 modèles (par exemple régression logistique, arbre / forêt), gestion du déséquilibre (`class_weight='balanced'`), comparaison à un modèle naïf.
- Recall, F1, ROC-AUC, matrice de confusion, importance des variables. Objectifs fixés dans `00_business_understanding` : recall ≥ 0,60, ROC-AUC ≥ 0,65.
- Une colonne `_log` **ou** son originale, jamais les deux. Ne pas mettre `total_visits` avec ses trois composantes.

**Membre 1 — Intégration**
- Fusionner les notebooks `00` à `05` en un notebook final, exporter le CSV final.
- Corriger les points listés ci-dessous.

**Membre 2 — Présentation**
- Assembler les slides (modèle commun envoyé avant) et organiser la répétition chronométrée (10 min, 2 min par membre).
- Chacun envoie ses 2 slides **au plus tard samedi matin**.

**Tous** : installer scikit-learn et seaborn dans le `venv` (`venv\Scripts\pip install scikit-learn seaborn`), absents pour l'instant ; `requirements.txt` est vide.

## Points à corriger

| Où | Problème |
|---|---|
| `01`, conclusion « l'âge a peu d'effet » | Calculée sur les données brutes. Après nettoyage, la réadmission passe de ≈ 7 % (20–60 ans) à 10,8 % (80–90 ans) |
| `01`, cellule « 67 630 séjours… » | Cite un tableau qui n'est pas affiché (seul un graphique l'est) |
| `01`, « près de 40 % » | Contredit le 34,6 % du résumé final |
| `01`, « 2 médicaments constants » | Il y en a 3 après nettoyage (`glimepiride-pioglitazone`). Le calculer en code |
| `02_nettoyage`, justification de `weight` | « Même taux de réadmission » jamais affiché (vrai : 11,2 % des deux côtés) ; « sous-ensemble d'hôpitaux » invérifiable (pas d'identifiant d'hôpital) |

Déjà corrigé dans `02_nettoyage` : les pourcentages calculés avant la suppression des patients répétés (79 % au lieu de 84 %, 9,2–9,9 % au lieu de 11,7–11,8 %, race 2,7 % / 7,4 % contre 9,0 %).

## Limites à mentionner

- Pas de dates : l'ordre des `encounter_id` est supposé chronologique.
- Un seul séjour par patient : les patients les plus fréquents sont sous-représentés (le taux de réadmission passe de 11,2 % à 9,0 %).
- Données américaines de 1999–2008, patients diabétiques hospitalisés de 1 à 14 jours : résultats non généralisables.
- Résultats à présenter comme des **associations**, pas des causes.
