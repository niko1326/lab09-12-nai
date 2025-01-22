import numpy as np
import matplotlib.pyplot as plt

# Parametry modelu
np.random.seed(42)  # Dla powtarzalności wyników
num_simulations = 1000
horizon_years = np.random.randint(3, 6, size=num_simulations)  # 3-5 lat

# Założenia dla budżetu, kosztów i przychodów
base_budget = 500000
budget_overruns = np.random.uniform(1.1, 1.3, size=num_simulations)  # 10%-30% przekroczenia
annual_revenue_mean = 300000
annual_revenue_std = 50000
annual_cost_mean = 200000
annual_cost_std = 30000
discount_rate = 0.10

# Symulacja Monte Carlo
npvs = []
for i in range(num_simulations):
    years = horizon_years[i]
    budget = base_budget * budget_overruns[i]

    # Losowanie przychodów i kosztów
    revenues = np.random.normal(annual_revenue_mean, annual_revenue_std, years)
    costs = np.random.normal(annual_cost_mean, annual_cost_std, years)

    # Obliczanie NPV
    cash_flows = revenues - costs
    npv = sum(cash_flows / (1 + discount_rate) ** np.arange(1, years + 1)) - budget
    npvs.append(npv)

# Analiza wyników
npvs = np.array(npvs)
probability_loss = np.mean(npvs < 0)
average_npv = np.mean(npvs)

# Wyniki
print(f"Średni NPV: {average_npv:.2f} PLN")
print(f"Prawdopodobieństwo straty (ujemny NPV): {probability_loss * 100:.2f}%")

# Wizualizacja
plt.hist(npvs, bins=30, color='skyblue', edgecolor='black')
plt.title("Rozkład NPV - Symulacja Monte Carlo")
plt.xlabel("NPV (PLN)")
plt.ylabel("Liczba scenariuszy")
plt.axvline(0, color='red', linestyle='--', label='Break-even')
plt.legend()
plt.show()
