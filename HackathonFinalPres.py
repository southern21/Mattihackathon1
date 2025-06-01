import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

### Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

### Model parameters
N = 1000            
J = 1.0             
h = np.random.uniform(-1, 1, size=N)

def total_energy(spins):
    interaction_energy = -J * np.sum(spins[:-1] * spins[1:])
    field_energy = -np.sum(h * spins)
    return interaction_energy + field_energy


def delta_energy(spins, i):
    left = spins[i-1] if i > 0 else 0
    right = spins[i+1] if i < N-1 else 0
    delta_E = 2 * J * spins[i] * (left + right) + 2 * h[i] * spins[i]
    return delta_E

### Classical Simulated Annealing
def simulated_annealing(max_iter=10000):
    spins = np.random.choice([-1, 1], size=N)
    T_init = 2.0
    T_min = 1e-6
    alpha = (T_min / T_init) ** (1.0 / max_iter)
    T = T_init
    energies = []
    current_energy = total_energy(spins)

    for iteration in range(max_iter):
        i = np.random.randint(0, N)
        dE = delta_energy(spins, i)

        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            spins[i] *= -1
            current_energy += dE

        energies.append(current_energy)
        T *= alpha

    return energies, current_energy

### Classical Tabu Search
def tabu_search(tabu_size=50, max_iter=10000):
    spins = np.random.choice([-1, 1], size=N)
    energies = []
    current_energy = total_energy(spins)
    tabu_list = deque(maxlen=tabu_size)

    for _ in range(max_iter):
        best_move = None
        best_dE = float('inf')

        for i in range(N):
            if i in tabu_list:
                continue
            dE = delta_energy(spins, i)
            if dE < best_dE:
                best_dE = dE
                best_move = i

        if best_move is not None:
            spins[best_move] *= -1
            current_energy += best_dE
            tabu_list.append(best_move)

        energies.append(current_energy)

    return energies, current_energy

### Quantum-Inspired Tabu Search
def quantum_inspired_tabu(tabu_size=50, diversification_prob=0.01, max_iter=10000):
    spins = np.random.choice([-1, 1], size=N)
    energies = []
    current_energy = total_energy(spins)
    tabu_list = deque(maxlen=tabu_size)

    for _ in range(max_iter):
        if np.random.rand() < diversification_prob:
            i = np.random.randint(0, N)
            dE = delta_energy(spins, i)
            spins[i] *= -1
            current_energy += dE
            tabu_list.append(i)
        else:
            best_move = None
            best_dE = float('inf')

            for i in range(N):
                if i in tabu_list:
                    continue
                dE = delta_energy(spins, i)
                if dE < best_dE:
                    best_dE = dE
                    best_move = i

            if best_move is not None:
                spins[best_move] *= -1
                current_energy += best_dE
                tabu_list.append(best_move)

        energies.append(current_energy)

    return energies, current_energy

### Quantum Simulated Annealing (Santoro-Tosatti Model)
def quantum_simulated_annealing(P=30, gamma_init=1.5, T_init=5.0, T_min=0.01, max_iter=10000):
    beta_init = 1 / T_init
    spins = np.random.choice([-1, 1], size=(P, N))

    alpha_T = (T_min / T_init) ** (1.0 / max_iter)
    gamma_min = 0.01
    alpha_gamma = (gamma_min / gamma_init) ** (1.0 / max_iter)

    T = T_init
    gamma = gamma_init
    energies = []

    for iteration in range(max_iter):
        beta = 1 / T
        J_perp = -0.5 / beta * np.log(np.tanh(beta * gamma / P))

        for _ in range(N):
            p = np.random.randint(0, P)
            i = np.random.randint(0, N)

            left = spins[p, i-1] if i > 0 else 0
            right = spins[p, i+1] if i < N-1 else 0

            delta_E_classical = 2 * J * spins[p, i] * (left + right) + 2 * h[i] * spins[p, i]

            prev_p = (p - 1) % P
            next_p = (p + 1) % P
            delta_E_quantum = 2 * J_perp * spins[p, i] * (spins[prev_p, i] + spins[next_p, i])

            delta_E_total = delta_E_classical + delta_E_quantum

            if delta_E_total < 0 or np.random.rand() < np.exp(-beta * delta_E_total):
                spins[p, i] *= -1

        avg_energy = np.mean([total_energy(spins[p_r, :]) for p_r in range(P)])
        energies.append(avg_energy)

        T *= alpha_T
        gamma *= alpha_gamma

    final_energy = np.mean([total_energy(spins[p_r, :]) for p_r in range(P)])
    return energies, final_energy

### Run all algorithms
max_iter = 5000

sa_energies, sa_final = simulated_annealing(max_iter)
tabu_energies, tabu_final = tabu_search(max_iter=max_iter)
q_tabu_energies, q_tabu_final = quantum_inspired_tabu(max_iter=max_iter)
qsa_energies, qsa_final = quantum_simulated_annealing(max_iter=max_iter)

### Plotting
plt.figure(figsize=(12, 6))
plt.plot(sa_energies, label='Simulated Annealing')
plt.plot(tabu_energies, label='Classical Tabu Search')
plt.plot(q_tabu_energies, label='Quantum-Inspired Tabu Search')
plt.plot(qsa_energies, label='Quantum Simulated Annealing')
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.legend()
plt.grid(True)
plt.show()

# Print final energies
print(f"Final SA energy: {sa_final}")
print(f"Final Tabu energy: {tabu_final}")
print(f"Final Quantum-Inspired Tabu energy: {q_tabu_final}")
print(f"Final Quantum SA energy: {qsa_final}")
