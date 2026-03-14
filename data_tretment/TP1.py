import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ============================================================
# ETAPE 1 : Exploration et données manquantes
# ============================================================

# 1. Import dataset
df = pd.read_csv("Titanic-Dataset.csv")
print("=== 5 premières lignes ===")
print(df.head())

# 2. Taille, colonnes, variable cible, valeurs manquantes
print("\n=== Taille du dataset ===")
print("Shape:", df.shape)

print("\n=== Colonnes (features) ===")
print(df.columns.tolist())

print("\n=== Variable cible : Survived ===")
print(df["Survived"].value_counts())

print("\n=== Valeurs manquantes (aperçu global) ===")
print(df.isnull().sum())

# 3. Analyser la colonne Age
age = df["Age"]
print("\n=== Colonne Age ===")
print(age)

missing_age = age[age.isnull()]
print("\n=== Valeurs manquantes dans Age (isnull) ===")
print(missing_age)

print("\n=== Nombre de valeurs manquantes dans Age (len) ===")
print(len(missing_age))

# 4. Nombre de valeurs manquantes pour TOUTES les colonnes
print("\n=== Valeurs manquantes pour toutes les colonnes ===")
print(df.isnull().sum())

# 5. Question de réflexion (réponse en commentaire)
"""
Pourquoi traiter les données manquantes ?
- Les algorithmes ML ne peuvent généralement pas traiter les NaN directement.
- Les valeurs manquantes peuvent biaiser les statistiques et les modèles.
- Elles réduisent la qualité et la fiabilité des prédictions.

Moyens possibles :
- Suppression : supprimer les lignes ou colonnes avec des valeurs manquantes (dropna()).
  → Simple mais perd de l'information.
- Imputation par la moyenne/médiane : remplacer par la valeur centrale.
  → Conserve les données, adapté aux variables numériques.
- Imputation par le mode : pour les variables catégoriques.
- Imputation par modèle (KNN, régression) : plus précis mais complexe.
"""

# ============================================================
# ETAPE 2 : Traitement des valeurs manquantes
# ============================================================

# 1. Vérification et suppression
print("\n=== Nombre de valeurs manquantes dans Age ===")
print(df["Age"].isnull().sum())

df_dropped = df.dropna(subset=["Age"])
print("\n=== Après suppression des lignes avec Age manquant ===")
print("Nouvelle taille:", df_dropped.shape)
print("Valeurs manquantes dans Age:", df_dropped["Age"].isnull().sum())

# 2. Imputation par la moyenne (sur le DataFrame original)
mean_age = df["Age"].mean()
print(f"\n=== Moyenne de Age : {mean_age:.2f} ===")

df["Age"] = df["Age"].fillna(mean_age)

print("=== Valeurs manquantes dans Age après imputation ===")
print(df["Age"].isnull().sum())  # Doit afficher 0

# ============================================================
# ETAPE 3 : Encodage des variables catégoriques
# ============================================================

print("\n=== Colonne Sex avant encodage ===")
print(df["Sex"].value_counts())

# LabelEncoder : male -> 1, female -> 0 (ou inversement)
le = LabelEncoder()
df["Sex_encoded"] = le.fit_transform(df["Sex"])

print("\n=== Colonne Sex après LabelEncoder ===")
print(df[["Sex", "Sex_encoded"]].drop_duplicates())

print("\n=== Aperçu final du DataFrame ===")
print(df[["PassengerId", "Survived", "Pclass", "Sex", "Sex_encoded", "Age"]].head(10))

"""
Justification du choix LabelEncoder pour Sex :
- La variable Sex est binaire (seulement 2 valeurs : male / female).
- LabelEncoder la transforme en 0 et 1, ce qui est suffisant.
- One-Hot Encoding serait redondant ici car il créerait 2 colonnes
  (Sex_male, Sex_female) alors qu'une seule suffit pour représenter
  une variable binaire sans introduire de multicolinéarité.
- LabelEncoder est donc plus simple et plus efficace pour une variable binaire.
"""