
# Mimir : Calculateur d'Options 📈

**Mimir** est un outil Python interactif pour la valorisation d'options financières.  
Il permet de calculer le prix d'options de type **Call** ou **Put** en utilisant deux modèles de pricing robustes.

---

## ✨ Fonctionnalités Clés

- **Choix du modèle de valorisation** :
  - **Européenne (Black-Scholes-Merton - BSM)** : calcul avec rendement de dividende continu.
  - **Américaine (Cox-Ross-Rubinstein - CRR)** : dividendes discrets + exercice anticipé.
  
- **Calcul du prix de l'option** (Call ou Put).

- **Grecs (BSM uniquement)** :
  - Delta (Δ), Gamma (Γ), Vega (ν), Theta (Θ), Rho (Ρ)

- **Gestion des dividendes** :
  - Dividendes **continus** (BSM)
  - Dividendes **discrets (1 à 4 par an)** (Binomial)

- **Validation des entrées** : messages clairs en cas d’erreur (types, valeurs…)

- **Graphique** : visualisation du profit/perte net à l'échéance.

- **Interface CLI** : intuitive en ligne de commande.

---

## 🛠 Prérequis

- Python 3.8+
- Bibliothèques nécessaires :
  - `numpy`
  - `scipy`
  - `matplotlib`
- Pour le développement :
  - `black`
  - `flake8`

---

## 🚀 Installation & Exécution

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre_nom_utilisateur/Mimir.git
cd Mimir
````

> Remplace `votre_nom_utilisateur` par ton identifiant GitHub.

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
```

#### Sous Windows :

```bash
.venv\Scripts\activate
```

#### Sous macOS / Linux :

```bash
source ./.venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer l'application

```bash
python main.py
```

---

## 📘 Mode d'utilisation

Une fois lancé, l’outil te guide pas à pas :

* `EU` ou `US` pour le type d’option
* `C` ou `P` pour Call ou Put
* S, K, T, r, sigma…
* **BSM** : q (dividende continu)
* **CRR** : N, D (montant), T\_div (date)

---

## 🧪 Exemples d’utilisation

### Exemple 1 – Option Européenne (Call) avec dividende continu

```bash
Bienvenue dans Mimir : Le Calculateur d'Options
Quel type d'option ? (EU/US) : EU
Option Call ou Put ? (C/P) : C
S = 100
K = 105
T = 1
r = 0.045
sigma = 0.2
q = 0.01
```

```
--- Résultats Black-Scholes ---
Prix : 5.48 $
d1 = 0.2247 | d2 = 0.0247
Delta : 0.5367 | Vega : 29.35 | Theta : -0.0108
```

➡️ Un graphique est généré automatiquement.

---

### Exemple 2 – Option Américaine (Put) avec dividende discret

```bash
Bienvenue dans Mimir : Le Calculateur d'Options
Type : US
Type d'option : P
S = 100 | K = 100 | T = 1 | r = 0.05 | sigma = 0.2
Nombre de pas : 500
Dividendes ? oui
Nombre : 1
Montant D1 = 2
T_div1 = 0.9
```

```
--- Résultats Binomial Américain ---
Prix de l’option Put : 6.64 $
```

➡️ Graphique généré automatiquement.

---

## 📈 Feuille de route

### 🔧 Code

* Support complet des dividendes discrets multiples (en cours)
* Robustesse accrue des entrées utilisateur

### 📊 Modèles

* Monte Carlo pour options exotiques (barrière, asiatique)
* Volatilité implicite non constante (surface de vol)
* Implémentation partielle en Rust pour accélérer les calculs

### 🖥 Interface & distribution

* GUI (Tkinter)
* Packaging Python
* Tests unitaires avec `pytest`

---