# mimir/ui/main_app.py

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading 

# --- Importe tes modules existants ---
from data.market_data_loader import get_option_expirations, get_current_stock_price, get_option_chain, get_risk_free_rate
from calibration.calibrate_heston import run_heston_calibration 

from core.monte_carlo_pricer import run_monte_carlo
from src.models.exotic.payoffs import calculate_asian_payoff, calculate_barrier_payoff, calculate_digital_payoff


class MimirApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mimir: Calculateur d'Options Avancé")
        self.root.geometry("800x650") # Taille de fenêtre initiale légèrement plus grande

        # --- Variables pour les paramètres calibrés de Heston ---
        self.heston_params = {
            "v0": tk.DoubleVar(value=0.0),
            "kappa": tk.DoubleVar(value=0.0),
            "theta": tk.DoubleVar(value=0.0),
            "xi": tk.DoubleVar(value=0.0),
            "rho": tk.DoubleVar(value=0.0),
        }
        self.calibrated_ticker = tk.StringVar(value="")
        self.calibration_message = tk.StringVar(value="Statut: Prêt à calibrer")

        # --- Nouvelles variables pour la barre de progression ---
        self.progress_value = tk.DoubleVar(value=0)
        self.progress_text = tk.StringVar(value="0.00%")

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        model_selection_frame = ttk.LabelFrame(main_frame, text="Sélection du Modèle", padding="10")
        model_selection_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        main_frame.columnconfigure(0, weight=1)

        self.selected_model = tk.StringVar()
        self.selected_model.set("Heston-Monte Carlo")

        rb_vanilla = ttk.Radiobutton(model_selection_frame, text="Options Européennes/Américaines Vanille (BSM/Binomial/B-S)",
                                     variable=self.selected_model, value="Vanilla")
        rb_exotic = ttk.Radiobutton(model_selection_frame, text="Options Exotiques (Heston-Monte Carlo)",
                                    variable=self.selected_model, value="Heston-Monte Carlo")

        rb_vanilla.grid(row=0, column=0, sticky=tk.W, pady=5)
        rb_exotic.grid(row=1, column=0, sticky=tk.W, pady=5)

        self.selected_model.trace_add("write", self.on_model_select)

        self.dynamic_params_frame = ttk.Frame(main_frame, padding="10 10 10 10")
        self.dynamic_params_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        self.dynamic_params_frame.columnconfigure(0, weight=1)

        self.create_heston_monte_carlo_section(self.dynamic_params_frame)
        self.create_vanilla_section(self.dynamic_params_frame)

        self.on_model_select()

        status_label = ttk.Label(main_frame, textvariable=self.calibration_message, foreground="white")
        status_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # --- Barre de progression ---
        self.progress_bar = ttk.Progressbar(main_frame, orient="horizontal", length=200, mode="determinate", variable=self.progress_value)
        self.progress_bar.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=5, pady=5) # Nouvelle ligne

        self.progress_label = ttk.Label(main_frame, textvariable=self.progress_text)
        self.progress_label.grid(row=3, column=1, sticky=(tk.W), padx=5, pady=5) # Nouvelle ligne

    def create_heston_monte_carlo_section(self, parent_frame):
        self.heston_mc_frame = ttk.LabelFrame(parent_frame, text="Paramètres Heston & Monte Carlo", padding="10")

        ttk.Label(self.heston_mc_frame, text="Ticker :").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.ticker_entry = ttk.Entry(self.heston_mc_frame, width=15)
        self.ticker_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        self.ticker_entry.insert(0, "AAPL")

        # Désactive le bouton pendant la calibration
        self.calibrate_button = ttk.Button(self.heston_mc_frame, text="Calibrer Heston", command=self.start_calibration_thread)
        self.calibrate_button.grid(row=0, column=2, padx=10, pady=5)

        ttk.Label(self.heston_mc_frame, text="Ticker calibré :").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Label(self.heston_mc_frame, textvariable=self.calibrated_ticker, foreground="green").grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=5)

        params_labels = ["V0", "Kappa", "Theta", "Xi", "Rho"]
        params_vars = [self.heston_params["v0"], self.heston_params["kappa"], self.heston_params["theta"],
                       self.heston_params["xi"], self.heston_params["rho"]]

        for i, label_text in enumerate(params_labels):
            ttk.Label(self.heston_mc_frame, text=f"{label_text} :").grid(row=2+i, column=0, sticky=tk.W, pady=2)
            entry = ttk.Entry(self.heston_mc_frame, textvariable=params_vars[i], state='readonly', width=15)
            entry.grid(row=2+i, column=1, sticky=(tk.W, tk.E), pady=2)

        ttk.Label(self.heston_mc_frame, text="Type d'Option Exotique :").grid(row=7, column=0, sticky=tk.W, pady=10)
        self.exotic_option_type = ttk.Combobox(self.heston_mc_frame,
                                               values=["Barrière", "Asiatique", "Digitale"],
                                               state="readonly")
        self.exotic_option_type.grid(row=7, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        self.exotic_option_type.set("Barrière")
        self.exotic_option_type.bind("<<ComboboxSelected>>", self.on_exotic_type_select)

        self.exotic_specific_params_frame = ttk.Frame(self.heston_mc_frame, padding="5")
        self.exotic_specific_params_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.create_exotic_specific_fields()

        common_mc_frame = ttk.LabelFrame(self.heston_mc_frame, text="Paramètres Option & MC", padding="10")
        common_mc_frame.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(common_mc_frame, text="Strike (K) :").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.strike_entry = ttk.Entry(common_mc_frame, width=10)
        self.strike_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        self.strike_entry.insert(0, "100")

        ttk.Label(common_mc_frame, text="Maturité (T en années) :").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.maturity_entry = ttk.Entry(common_mc_frame, width=10)
        self.maturity_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        self.maturity_entry.insert(0, "1.0")

        ttk.Label(common_mc_frame, text="Taux sans risque (r) :").grid(row=0, column=2, sticky=tk.W, padx=10, pady=2)
        self.risk_free_rate_entry = ttk.Entry(common_mc_frame, width=10)
        self.risk_free_rate_entry.grid(row=0, column=3, sticky=(tk.W, tk.E), pady=2)
        self.risk_free_rate_entry.insert(0, "0.01")

        ttk.Label(common_mc_frame, text="Rendement Dividende (q) :").grid(row=1, column=2, sticky=tk.W, padx=10, pady=2)
        self.dividend_yield_entry = ttk.Entry(common_mc_frame, width=10)
        self.dividend_yield_entry.grid(row=1, column=3, sticky=(tk.W, tk.E), pady=2)
        self.dividend_yield_entry.insert(0, "0.0")

        ttk.Label(common_mc_frame, text="Nb Simulations :").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.num_simulations_entry = ttk.Entry(common_mc_frame, width=10)
        self.num_simulations_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2)
        self.num_simulations_entry.insert(0, "100000")

        ttk.Label(common_mc_frame, text="Nb Étapes :").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.num_steps_entry = ttk.Entry(common_mc_frame, width=10)
        self.num_steps_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2)
        self.num_steps_entry.insert(0, "252")

        calculate_price_button = ttk.Button(self.heston_mc_frame, text="Calculer Prix de l'Option", command=self.calculate_option_price)
        calculate_price_button.grid(row=10, column=0, columnspan=3, pady=10)

        ttk.Label(self.heston_mc_frame, text="Prix calculé :").grid(row=11, column=0, sticky=tk.W, pady=5)
        self.result_price_label = ttk.Label(self.heston_mc_frame, text="N/A", foreground="white", font=("Arial", 12, "bold"))
        self.result_price_label.grid(row=11, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.heston_mc_frame.columnconfigure(1, weight=1)
        common_mc_frame.columnconfigure(1, weight=1)
        common_mc_frame.columnconfigure(3, weight=1)

    def create_vanilla_section(self, parent_frame):
        self.vanilla_frame = ttk.LabelFrame(parent_frame, text="Paramètres Options Vanille", padding="10")
        ttk.Label(self.vanilla_frame, text="Implémentation BSM/Binomial/B-S à venir ici...").grid(row=0, column=0, sticky=tk.W, pady=10)

    def create_exotic_specific_fields(self, *args):
        for widget in self.exotic_specific_params_frame.winfo_children():
            widget.destroy()

        selected_type = self.exotic_option_type.get()
        current_row = 0

        if selected_type == "Barrière":
            ttk.Label(self.exotic_specific_params_frame, text="Niveau de Barrière :").grid(row=current_row, column=0, sticky=tk.W, pady=2)
            self.barrier_level_entry = ttk.Entry(self.exotic_specific_params_frame, width=10)
            self.barrier_level_entry.grid(row=current_row, column=1, sticky=(tk.W, tk.E), pady=2)
            self.barrier_level_entry.insert(0, "90")

            current_row += 1
            ttk.Label(self.exotic_specific_params_frame, text="Type de Knock :").grid(row=current_row, column=0, sticky=tk.W, pady=2)
            self.knock_type_combobox = ttk.Combobox(self.exotic_specific_params_frame, values=["out", "in"], state="readonly", width=8)
            self.knock_type_combobox.grid(row=current_row, column=1, sticky=(tk.W, tk.E), pady=2)
            self.knock_type_combobox.set("out")

            current_row += 1
            ttk.Label(self.exotic_specific_params_frame, text="Direction Barrière :").grid(row=current_row, column=0, sticky=tk.W, pady=2)
            self.barrier_direction_combobox = ttk.Combobox(self.exotic_specific_params_frame, values=["up", "down"], state="readonly", width=8)
            self.barrier_direction_combobox.grid(row=current_row, column=1, sticky=(tk.W, tk.E), pady=2)
            self.barrier_direction_combobox.set("down")


        elif selected_type == "Asiatique":
            ttk.Label(self.exotic_specific_params_frame, text="Période de Moyenne :").grid(row=current_row, column=0, sticky=tk.W, pady=2)
            self.avg_period_entry = ttk.Entry(self.exotic_specific_params_frame, width=10)
            self.avg_period_entry.grid(row=current_row, column=1, sticky=(tk.W, tk.E), pady=2)
            self.avg_period_entry.insert(0, "all_steps")

        elif selected_type == "Digitale":
            ttk.Label(self.exotic_specific_params_frame, text="Montant de Paiement :").grid(row=current_row, column=0, sticky=tk.W, pady=2)
            self.payoff_amount_entry = ttk.Entry(self.exotic_specific_params_frame, width=10)
            self.payoff_amount_entry.grid(row=current_row, column=1, sticky=(tk.W, tk.E), pady=2)
            self.payoff_amount_entry.insert(0, "1.0")

        current_row += 1
        ttk.Label(self.exotic_specific_params_frame, text="Type (Call/Put) :").grid(row=current_row, column=0, sticky=tk.W, pady=2)
        self.option_type_combobox = ttk.Combobox(self.exotic_specific_params_frame, values=["C", "P"], state="readonly", width=8)
        self.option_type_combobox.grid(row=current_row, column=1, sticky=(tk.W, tk.E), pady=2)
        self.option_type_combobox.set("C")

        self.exotic_specific_params_frame.columnconfigure(1, weight=1)


    def on_model_select(self, *args):
        selected = self.selected_model.get()
        if selected == "Heston-Monte Carlo":
            self.heston_mc_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
            self.vanilla_frame.grid_forget()
        elif selected == "Vanilla":
            self.vanilla_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
            self.heston_mc_frame.grid_forget()

    def on_exotic_type_select(self, *args):
        self.create_exotic_specific_fields()

    def start_calibration_thread(self):
        """ Démarre la calibration dans un thread séparé pour ne pas bloquer l'UI. """
        self.calibrate_button.config(state=tk.DISABLED) # Désactive le bouton
        self.calibration_message.set("Statut: Calibration en cours...")
        self.progress_value.set(0) # Réinitialise la barre de progression
        self.progress_text.set("0.00%")
        self.root.update_idletasks()

        # Lance la calibration dans un thread séparé
        threading.Thread(target=self._run_calibration_threaded, daemon=True).start()

    def _run_calibration_threaded(self):
        """ Fonction de calibration exécutée dans un thread. """
        ticker = self.ticker_entry.get().strip().upper()
        if not ticker:
            messagebox.showwarning("Erreur de Saisie", "Veuillez entrer un symbole boursier (Ticker).")
            self.root.after(0, lambda: self.calibrate_button.config(state=tk.NORMAL)) # Réactive le bouton
            return

        try:
            expirations = get_option_expirations(ticker)
            if not expirations:
                raise ValueError(f"Aucune date d'expiration d'option trouvée pour {ticker}.")

            expiration_date_to_calibrate = sorted(expirations)[0]

            initial_params = (0.04, 1.0, 0.04, 0.1, -0.7)
            bounds = [(1e-5, 2.0), (1e-5, 10.0), (1e-5, 2.0), (1e-5, 2.0), (-0.99, 0.99)]

            # Appel de la fonction de calibration avec un callback pour l'UI
            calibration_result = run_heston_calibration(
                ticker, expiration_date_to_calibrate, initial_params, bounds,
                ui_progress_callback=self.update_progress_bar # Passe le callback
            )

            # Met à jour l'UI sur le thread principal après la fin de la calibration
            self.root.after(0, lambda: self.handle_calibration_result(calibration_result, ticker))

        except Exception as e:
            self.root.after(0, lambda: self.show_calibration_error(e, ticker))

    def handle_calibration_result(self, calibration_result, ticker):
        """ Gère le résultat de la calibration et met à jour l'UI principale. """
        if calibration_result and calibration_result["status"] == "success" and calibration_result["min_error"] < 1000:
            self.heston_params["v0"].set(calibration_result["calibrated_params"][0])
            self.heston_params["kappa"].set(calibration_result["calibrated_params"][1])
            self.heston_params["theta"].set(calibration_result["calibrated_params"][2])
            self.heston_params["xi"].set(calibration_result["calibrated_params"][3])
            self.heston_params["rho"].set(calibration_result["calibrated_params"][4])
            self.calibrated_ticker.set(ticker)
            self.calibration_message.set(f"Statut: Calibration réussie pour {ticker} ! Erreur: {calibration_result['min_error']:.2f}")
            self.progress_value.set(100) # Assure que la barre est à 100%
            self.progress_text.set("100.00%")
            messagebox.showinfo("Calibration Réussie", f"Modèle Heston calibré pour {ticker}.\nErreur Minimale: {calibration_result['min_error']:.2f}")
        else:
            error_message = calibration_result.get("optimizer_message", "Erreur inconnue") if calibration_result else "Aucun résultat de calibration."
            self.calibration_message.set(f"Statut: Échec de la calibration pour {ticker}. {error_message}")
            self.progress_value.set(0) # Remet à 0 si échec
            self.progress_text.set("Échec")
            messagebox.showerror("Échec Calibration", f"Impossible de calibrer le modèle Heston pour {ticker}.\nMessage: {error_message}")
        
        self.calibrate_button.config(state=tk.NORMAL) # Réactive le bouton


    def show_calibration_error(self, e, ticker):
        """ Affiche une erreur de calibration sur l'UI. """
        self.calibration_message.set(f"Statut: Erreur de calibration: {e}")
        messagebox.showerror("Erreur de Calibration", f"Une erreur est survenue lors de la calibration de {ticker}: {e}")
        self.progress_value.set(0) # Remet à 0 si échec
        self.progress_text.set("Échec")
        self.calibrate_button.config(state=tk.NORMAL) # Réactive le bouton


    def update_progress_bar(self, current, total):
        """ Callback appelé par le thread de calibration pour mettre à jour la barre de progression. """
        percentage = (current / total) * 100
        # Utilise root.after pour s'assurer que la mise à jour se fait sur le thread principal de Tkinter
        self.root.after(0, self.progress_value.set, percentage)
        self.root.after(0, self.progress_text.set, f"{percentage:.2f}%")
        self.root.after(0, self.calibration_message.set, f"Statut: Calibration en cours... {current}/{total} itérations")


    def calculate_option_price(self):
        """ Lance le calcul du prix de l'option exotique sélectionnée. """
        if not self.calibrated_ticker.get():
            messagebox.showwarning("Données Manquantes", "Veuillez calibrer le modèle Heston d'abord.")
            return

        try:
            v0 = self.heston_params["v0"].get()
            kappa = self.heston_params["kappa"].get()
            theta = self.heston_params["theta"].get()
            xi = self.heston_params["xi"].get()
            rho = self.heston_params["rho"].get()

            K = float(self.strike_entry.get())
            T = float(self.maturity_entry.get())
            r = float(self.risk_free_rate_entry.get())
            q = float(self.dividend_yield_entry.get())
            N_simulations = int(self.num_simulations_entry.get())
            N_steps = int(self.num_steps_entry.get())

            option_type = self.option_type_combobox.get()
            if not option_type:
                raise ValueError("Veuillez sélectionner le type d'option (Call/Put).")

            S0 = get_current_stock_price(self.calibrated_ticker.get())
            if S0 is None:
                raise ValueError(f"Impossible de récupérer le prix spot pour {self.calibrated_ticker.get()}.")

            selected_exotic_type = self.exotic_option_type.get()
            payoff_func = None
            payoff_kwargs = {"K": K, "option_type": option_type}

            if selected_exotic_type == "Barrière":
                barrier_level = float(self.barrier_level_entry.get())
                knock_type = self.knock_type_combobox.get()
                in_out = self.barrier_direction_combobox.get()
                payoff_func = calculate_barrier_payoff
                payoff_kwargs.update({"barrier_level": barrier_level, "knock_type": knock_type, "in_out": in_out})
            elif selected_exotic_type == "Asiatique":
                payoff_func = calculate_asian_payoff
            elif selected_exotic_type == "Digitale":
                payoff_amount = float(self.payoff_amount_entry.get())
                payoff_func = calculate_digital_payoff
                payoff_kwargs.update({"payoff_amount": payoff_amount})
            else:
                raise ValueError("Type d'option exotique non reconnu.")

            if payoff_func is None:
                raise ValueError("Fonction de payoff non définie.")

            self.calibration_message.set("Statut: Calcul du prix de l'option en cours... (Monte Carlo)")
            self.root.update_idletasks()

            # --- GESTION DES CHEMINS HESTON ---
            # >>> REMPLACE CE BLOC PAR TA VRAIE FONCTION generate_heston_paths !!! <<<
            # from models.heston.process import generate_heston_paths # DÉCOMMENTE QUAND FAIT
            
            print(f"DEBUG: Tentative de génération de chemins avec S0={S0}, V0={v0}, kappa={kappa}, theta={theta}, xi={xi}, rho={rho}, T={T}, N_steps={N_steps}, N_simulations={N_simulations}")
            
            # Placeholder pour generate_heston_paths (à REMPLACER par TA VRAIE SIMULATION HESTON)
            mu = r
            sigma_bsm_placeholder = np.sqrt(v0) # Utilise V0 comme proxy de variance pour BSM simple
            dt_paths = T / N_steps
            dW_placeholder = np.random.normal(size=(N_simulations, N_steps)) * np.sqrt(dt_paths)
            
            paths_S_placeholder = np.zeros((N_simulations, N_steps + 1))
            paths_S_placeholder[:, 0] = S0
            for t in range(N_steps):
                paths_S_placeholder[:, t+1] = paths_S_placeholder[:, t] * np.exp((mu - 0.5 * sigma_bsm_placeholder**2) * dt_paths + sigma_bsm_placeholder * dW_placeholder[:, t])
            
            paths_S = paths_S_placeholder # Tes vrais chemins S de Heston iront ici
            
            # --- Appel du Moteur Monte Carlo ---
            option_price = run_monte_carlo(paths_S, r, T, payoff_func, **payoff_kwargs)

            self.result_price_label.config(text=f"{option_price:.4f}")
            self.calibration_message.set("Statut: Calcul terminé.")
            messagebox.showinfo("Calcul Réussi", f"Le prix de l'option est: {option_price:.4f}")

        except ValueError as ve:
            self.calibration_message.set(f"Statut: Erreur de saisie: {ve}")
            messagebox.showerror("Erreur de Saisie", str(ve))
            self.result_price_label.config(text="ERREUR")
        except Exception as e:
            self.calibration_message.set(f"Statut: Erreur de calcul: {e}")
            messagebox.showerror("Erreur de Calcul", f"Une erreur est survenue lors du calcul: {e}")
            self.result_price_label.config(text="ERREUR")


# --- Point d'entrée de l'application ---
if __name__ == "__main__":
    root = tk.Tk()
    app = MimirApp(root)
    root.mainloop()