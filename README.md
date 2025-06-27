# Mimir : Calculateur d'Options 📈

Mimir est un outil Python interactif pour la valorisation d'options financières. Il permet de calculer le prix d'options de type Call ou Put en utilisant deux modèles :

* **Black-Scholes-Merton (BSM)** pour les options européennes.
* **Modèle Binomial (Cox-Ross-Rubinstein - CRR)** pour les options européennes et américaines.

Il fournit également les "Grecs" pour le modèle BSM, et génère une visualisation du profit/perte net à l'échéance.

## ✨ Fonctionnalités Clés

* **Choix du modèle de valorisation** selon le type d'option :

  * Européenne (BSM)
  * Américaine (Binomial CRR)

* **Calcul du prix de l'option** (Call ou Put)

* **Grecs (modèle BSM uniquement)** :

  * Delta (Δ) : Sensibilité au sous-jacent
  * Gamma (Γ) : Sensibilité du delta
  * Vega (ν) : Sensibilité à la volatilité
  * Theta (Θ) : Sensibilité au temps
  * Rho (Ρ) : Sensibilité au taux sans risque

* **Visualisation graphique** : Profit/perte net à l'échéance

* **Interface intuitive** : Utilisation en ligne de commande

## 🛠 Prérequis

Python 3.8+ installé et les bibliothèques suivantes :

* `numpy`
* `scipy`
* `matplotlib`

## 🚀 Installation & Exécution

### 1. Clonez le dépôt

```bash
git clone https://github.com/votre_nom_utilisateur/Mimir.git
cd Mimir
```

Remplacez `votre_nom_utilisateur` par votre identifiant GitHub si besoin.

### 2. Créez un environnement virtuel

```bash
python -m venv .venv
```

**Pour Windows** :

```bash
.venv\Scripts\activate
```

**Pour macOS / Linux** :

```bash
source ./.venv/bin/activate
```

### 3. Installez les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancez l'application

```bash
python main.py
```

## 📘 Mode d'utilisation

L'application se lance en terminal. Suivez les instructions pas à pas :

* Type d'option : `EU` pour européenne (BSM), `US` pour américaine (Binomial)
* Type : `C` pour Call, `P` pour Put
* Prix spot (S)
* Prix d'exercice (K)
* Temps jusqu'à l'échéance (T, en années)
* Taux sans risque (r)
* Volatilité (σ)
* Nombre de pas (N) si option américaine

## 📅 Exemple : Option Européenne

```bash
Bienvenue dans Mimir : Le Calculateur d'Options
Quel type d'option souhaitez-vous calculer ? (EU pour Européenne, US pour Américaine) : EU
Voulez-vous calculer le prix d'une option Call (C) ou Put (P) ? C
Entrez le prix spot actuel (S) : 100
Entrez le prix d'exercice (K) : 105
Entrez le temps jusqu'à l'échéance en années (T) : 1
Entrez le taux d'intérêt sans risque (r) : 0.045
Entrez la volatilité (sigma) : 0.2
```

Affichage :

```
--- Modèle utilisé : Black-Scholes-Merton (BSM) ---
Prix de l'option Call : 6.00 $
d1 = 0.2647
d2 = 0.0647
Delta = 0.6044
Gamma = 0.0187
Vega = 29.84
Theta = -0.0090
Rho = 0.4907
```

Un graphique du PnL est généré automatiquement.

## 📅 Exemple : Option Américaine

```bash
Bienvenue dans Mimir : Le Calculateur d'Options
Quel type d'option souhaitez-vous calculer ? (EU pour Européenne, US pour Américaine) : US
Voulez-vous calculer le prix d'une option Call (C) ou Put (P) ? P
Entrez le prix spot actuel (S) : 90
Entrez le prix d'exercice (K) : 100
Entrez le temps jusqu'à l'échéance en années (T) : 0.5
Entrez le taux d'intérêt sans risque (r) : 0.05
Entrez la volatilité (sigma) : 0.2
Entrez le nombre de pas (N) : 500
```

Affichage :

```
--- Modèle utilisé : Binomial (CRR) ---
Prix de l'option Put : 10.67 $
```

Un graphique du PnL est généré automatiquement.

## 🏐 Prochaines évolutions

* Gestion des dividendes
* Options exotiques (barrières, asiatiques...)
* Simulation de Monte Carlo
* Grecs pour le modèle Binomial
* Interface graphique (GUI)
