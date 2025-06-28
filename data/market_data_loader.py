# data/market_data_loader.py
import yfinance as yf
import pandas as pd
from datetime import datetime


def get_current_stock_price(ticker_symbol: str) -> float:
    """
    Récupère le prix actuel du sous-jacent.

    Paramètres:
        ticker_symbol (str): Le symbole boursier (ex: 'AAPL' pour Apple).

    Returns:
        float: Le prix actuel de l'action.
    """
    ticker = yf.Ticker(ticker_symbol)
    todays_data = ticker.history(period="1d")
    if not todays_data.empty:
        return todays_data["Close"].iloc[-1]
    else:
        raise ValueError(
            f"Impossible de récupérer le prix pour {ticker_symbol}. Vérifiez le symbole ou la disponibilité des données."
        )


def get_option_expirations(ticker_symbol: str) -> list[str]:
    """
    Récupère les dates d'expiration des options disponibles pour un ticker.

    Paramètres:
        ticker_symbol (str): Le symbole boursier.

    Returns:
        list[str]: Une liste de dates d'expiration au format 'YYYY-MM-DD'.
    """
    ticker = yf.Ticker(ticker_symbol)
    return ticker.options


def get_option_chain(
    ticker_symbol: str, expiration_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Récupère la chaîne d'options (Calls et Puts) pour une date d'expiration donnée.

    Paramètres:
        ticker_symbol (str): Le symbole boursier.
        expiration_date (str): La date d'expiration au format 'YYYY-MM-DD'.

    Returns:
        tuple[pd.DataFrame, pd.MimiDataFrame]: Un tuple contenant les DataFrames pour les calls et les puts.
                                           Renvoie des DataFrames vides si aucune donnée n'est trouvée.
    """
    ticker = yf.Ticker(ticker_symbol)
    option_chain = ticker.option_chain(expiration_date)
    calls = option_chain.calls if option_chain.calls is not None else pd.DataFrame()
    puts = option_chain.puts if option_chain.puts is not None else pd.DataFrame()
    return calls, puts


def get_risk_free_rate(source: str = "US10Y", days_to_maturity: int = None) -> float:
    """
    Récupère un taux sans risque.

    Paramètres:
        source (str): Source du taux. Actuellement supporte 'US10Y' (via ^TNX).
        days_to_maturity (int): Non utilisé directement pour ^TNX, car c'est un taux fixe à 10 ans.

    Returns:
        float: Le taux sans risque annuel (en décimal, ex: 0.01 pour 1%).
    """
    if source == "US10Y":
        print(
            "Avertissement: La récupération du taux sans risque via yfinance pour ^TNX est un proxy et peut être instable."
        )
        print(
            "Pour des taux précis et maturité-spécifiques, considérez une API dédiée (ex: FRED API, Eikon/Bloomberg, ou source publique)."
        )
        try:
            # Récupérer les données du CBOE Interest Rate 10 Year T-No (^TNX)
            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="1d") # Récupère la dernière clôture

            if not hist.empty:
                # ^TNX est en pourcentage (ex: 4.25 pour 4.25%), le convertir en décimal
                risk_free_rate = hist['Close'].iloc[0] / 100
                print(f"Taux sans risque (via ^TNX) récupéré: {risk_free_rate * 100:.2f}%")
                return risk_free_rate
            else:
                print("Erreur: Impossible de récupérer les données pour ^TNX. Utilisation d'un taux par défaut (4.00%).")
                return 0.04 # Taux par défaut si la récupération échoue
        except Exception as e:
            print(f"Erreur lors de la récupération du taux sans risque via yfinance pour ^TNX: {e}.")
            print("Utilisation d'un taux par défaut (4.00%).")
            return 0.04 # Taux par défaut en cas d'exception (ex: pas de connexion internet)
    else:
        raise ValueError("Source de taux sans risque non supportée.")


def get_current_datetime() -> datetime:
    """
    Retourne la date et l'heure actuelles.
    """
    return datetime.now()


if __name__ == "__main__":
    # Exemple d'utilisation
    ticker = "AAPL"  # Symbole pour Apple

    print(f"--- Données de marché pour {ticker} ---")

    # 1. Prix Spot
    try:
        spot_price = get_current_stock_price(ticker)
        print(f"Prix Spot actuel de {ticker}: {spot_price:.2f}")
    except ValueError as e:
        print(e)
        spot_price = None

    if spot_price:
        # 2. Dates d'expiration
        expirations = get_option_expirations(ticker)
        print("\nDates d'expiration disponibles:")
        if expirations:
            for exp in expirations[:5]:  # Affiche les 5 premières
                print(f"- {exp}")

            # 3. Chaîne d'options pour la première expiration
            first_expiration = expirations[0]
            print(
                f"\nChaîne d'options pour l'expiration la plus proche ({first_expiration}):"
            )
            calls, puts = get_option_chain(ticker, first_expiration)

            if not calls.empty:
                print("\n--- Calls ---")
                print(
                    calls[
                        ["strike", "lastPrice", "bid", "ask", "volume", "openInterest"]
                    ].head()
                )
            else:
                print("\nPas de données de Calls trouvées.")

            if not puts.empty:
                print("\n--- Puts ---")
                print(
                    puts[
                        ["strike", "lastPrice", "bid", "ask", "volume", "openInterest"]
                    ].head()
                )
            else:
                print("\nPas de données de Puts trouvées.")
        else:
            print("Aucune date d'expiration d'option trouvée.")

    # 4. Taux sans risque
    risk_free = get_risk_free_rate()
    print(f"\nTaux sans risque utilisé (proxy): {risk_free * 100:.2f}%")
    current_dt = get_current_datetime()
    print(f"\nDate et heure actuelles: {current_dt.strftime('%Y-%m-%d %H:%M:%S')}")
