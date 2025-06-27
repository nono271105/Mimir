# Mimir : Calculateur d'Options 

Ce script Python permet de calculer le prix d'une option (Call ou Put) en utilisant le modèle de Black-Scholes. Il fournit également les principales "Grecs" (Delta, Gamma, Vega, Theta, Rho) qui mesurent la sensibilité de l'option aux différents facteurs du marché. Enfin, il affiche un graphique interactif du profit/perte net de l'option à l'échéance.

## Fonctionnalités

* **Calcul du prix de l'option** (Call ou Put) selon Black-Scholes.
* Affichage des **paramètres intermédiaires** du modèle (d1, d2, N(d1), N(d2)).
* Calcul et affichage des **Grecs** :
    * **Delta**: Sensibilité du prix de l'option au prix du sous-jacent.
    * **Gamma**: Sensibilité du Delta aux variations du prix du sous-jacent.
    * **Vega**: Sensibilité du prix de l'option à la volatilité du sous-jacent.
    * **Theta**: Dépréciation temporelle du prix de l'option (par jour).
    * **Rho**: Sensibilité du prix de l'option au taux d'intérêt sans risque.
* **Visualisation du Profit/Perte net** à l'échéance via un graphique, incluant le prix d'exercice et le point d'équilibre.

## Prérequis

Assurez-vous d'avoir Python installé sur votre système. Vous aurez également besoin des bibliothèques suivantes :

* `numpy`
* `scipy`
* `matplotlib`

## Vous pouvez les installer via pip (dans votre environnement virtuel) :

```bash
pip install numpy scipy matplotlib
```

## Comment utiliser le script
Clonez le dépôt (ou téléchargez le fichier main.py) :

```Bash
git clone [https://github.com/votre_nom_utilisateur/Mimir.git](https://github.com/votre_nom_utilisateur/Mimir.git)
cd Mimir
```
(Remplacez votre_nom_utilisateur par le vôtre)

Configurez votre environnement virtuel (si ce n'est pas déjà fait) :

```Bash
python -m venv .venv
```
Activez l'environnement (selon votre OS): 
Windows: 
```Bash
\.venv\Scripts\activate
```
MacOS/Linux:
```Bash
source ./.venv/bin/activate
pip install -r requirements.txt
```
## Exécutez le script depuis votre terminal (assurez-vous que votre environnement virtuel est activé) :

```Bash
python main.py
```

Suivez les invites dans le terminal pour entrer les paramètres de votre option :

* `Type d'option (C pour Call, P pour Put)`

* `Prix spot actuel (S)`

* `Prix d'exercice (K)`

* `Temps jusqu'à l'échéance en années (T)`

* `Taux d'intérêt sans risque (r, ex: 0.045 pour 4.5%)`

* `Volatilité (sigma, ex: 0.2 pour 20%)`

Le script affichera le prix de l'option, les paramètres intermédiaires, les Grecs, puis ouvrira une fenêtre affichant le graphique de profit/perte.

Exemple d'utilisation : 
```Bash
Voulez-vous calculer le prix d'une option Call (C) ou Put (P) ? C
Entrez le prix spot actuel (S) : 100
Entrez le prix d'exercice (K) : 105
Entrez le temps jusqu'à l'échéance en années (T) : 1
Entrez le taux d'intérêt sans risque (r, ex: 0.045 pour 4.5%) : 0.045
Entrez la volatilité (sigma, ex: 0.2 pour 20%) : 0.2

--- Résultats du Modèle Black-Scholes ---
Le prix de l'option C est : 6.00 $
d1 = 0.2647
d2 = 0.0647
N(d1) = 0.6044
N(d2) = 0.5258

--- Les Grecs ---
Delta = 0.6044
Gamma = 0.0187
Vega = 29.8378 (pour 1% de volatilité)
Theta = -0.0090 (par jour)
Rho = 0.4907 (pour 1% de taux d'intérêt)
```
* (Un graphique s'ouvrira également après ces informations.)

## Amélioration
En cours...🚧 🔨