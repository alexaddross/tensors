import numpy as np

def force_based_sort(x, max_iters=10000000):
    x = np.array(x)
    indices = np.arange(len(x))
    velocities = np.zeros_like(indices, dtype=float)

    for _ in range(max_iters):
        forces = np.zeros_like(velocities)

        # Apply pairwise forces: if out of order, push apart
        for i in range(len(x) - 1):
            xi, xj = x[indices[i]], x[indices[i + 1]]
            if xi > xj:
                # Simulate a repulsive force
                force = (xi - xj)
                forces[i] += force
                forces[i + 1] -= force

        velocities += forces
        
        swapped = False
        for i in range(len(x) - 1):
            if velocities[i] > velocities[i + 1]:
                indices[i], indices[i + 1] = indices[i + 1], indices[i]
                velocities[i], velocities[i + 1] = velocities[i + 1], velocities[i]
                swapped = True

        if not swapped:
            break

    return x[indices]

def sorting_energy(x, perm):
    """Energy of permutation based on disorder and local metric."""
    E = 0.0
    for i in range(len(perm) - 1):
        xi, xj = x[perm[i]], x[perm[i + 1]]
        if xi > xj:
            gij = 1 + abs(xi - xj)  # Metric tensor scalar
            E += gij * (xi - xj) ** 2
    return E

def geodesic_sort_stochastic(x, max_iters=1000, initial_temp=1.0, cooling_rate=0.995):
    x = np.array(x)
    n = len(x)
    perm = np.arange(n)
    T = initial_temp

    current_energy = sorting_energy(x, perm)

    for _ in range(max_iters):
        i, j = np.random.choice(n, size=2, replace=False)
        new_perm = perm.copy()
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]

        new_energy = sorting_energy(x, new_perm)
        delta_E = new_energy - current_energy

        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            # Accept move
            perm = new_perm
            current_energy = new_energy

        T *= cooling_rate
        if current_energy == 0:
            break  # Perfectly sorted

    return x[perm]


import numpy as np

def energy(x, perm):
    """Riemannian energy based on disorder and local distances."""
    E = 0.0
    for i in range(len(perm) - 1):
        xi, xj = x[perm[i]], x[perm[i + 1]]
        if xi > xj:
            gij = 1 + abs(xi - xj)
            E += gij * (xi - xj) ** 2

    return E

def langevin_sort(x, steps=10000, eta=0.01, T=0.1):
    x = np.array(x)
    n = len(x)
    perm = np.arange(n)
    current_energy = energy(x, perm)

    for _ in range(steps):
        # Compute local metric (scalar here, could be matrix)
        i, j = np.random.choice(n, size=2, replace=False)
        if i > j: i, j = j, i

        gij = 1 + abs(x[perm[i]] - x[perm[j]])

        # Energy before and after swap
        new_perm = perm.copy()
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        new_energy = energy(x, new_perm)
        dE = new_energy - current_energy

        # Langevin probability
        prob = np.exp(-dE / (gij * T)) if dE > 0 else 1.0
        if np.random.rand() < prob:
            perm = new_perm
            current_energy = new_energy

        # Optional: decrease temperature
        T *= 0.9995
        if current_energy == 0:
            break

    return x[perm]


import numpy as np

def energy(x, perm, alpha=1.0, beta=1.0):
    E = 0.0
    for i in range(len(perm) - 1):
        xi, xj = x[perm[i]], x[perm[i+1]]
        if xi > xj:
            gij = 1 + alpha * abs(xi - xj)**beta
            E += gij * (xi - xj)**2
    return E

def build_metric_tensor(x, alpha=1.0, beta=1.0):
    n = len(x)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                G[i,j] = alpha * abs(x[i] - x[j])**beta
        G[i,i] = 1 + np.sum(G[i,:])
    return G

def approximate_gradient_global(x, perm, alpha=1.0, beta=1.0):
    n = len(perm)
    grad = np.zeros(n)
    current_energy = energy(x, perm, alpha, beta)
    # Approximate gradient by swapping i with j (all pairs)
    for i in range(n):
        dE_sum = 0
        for j in range(n):
            if i == j:
                continue
            new_perm = perm.copy()
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            new_energy = energy(x, new_perm, alpha, beta)
            dE_sum += new_energy - current_energy
        grad[i] = dE_sum / (n - 1)
    return grad

def langevin_sort_enhanced(x, steps=1000, eta=0.1, T=1.0, alpha=1.0, beta=1.0):
    x = np.array(x)
    n = len(x)
    perm = np.arange(n)
    G = build_metric_tensor(x, alpha, beta)
    G_inv = np.linalg.inv(G + 1e-6 * np.eye(n))

    for step in range(steps):
        grad = approximate_gradient_global(x, perm, alpha, beta)
        R_grad = G_inv @ grad

        # Pick top-k candidate indices by magnitude of R_grad (say k=3)
        k = 3
        candidates = np.argsort(-np.abs(R_grad))[:k]

        accepted = False
        for i in candidates:
            # Try swaps with a random j != i
            j = np.random.choice([idx for idx in range(n) if idx != i])
            new_perm = perm.copy()
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            dE = energy(x, new_perm, alpha, beta) - energy(x, perm, alpha, beta)

            prob = np.exp(-dE / T) if dE > 0 else 1.0
            if np.random.rand() < prob:
                perm = new_perm
                accepted = True
                break

        # If no move accepted, do a random swap to avoid stagnation
        if not accepted:
            i, j = np.random.choice(n, 2, replace=False)
            perm[i], perm[j] = perm[j], perm[i]

        # Cool temperature slowly
        T *= 0.9995

        if energy(x, perm, alpha, beta) == 0:
            break

    return x[perm]


import numpy as np

def energy(x, perm, alpha=1.0, beta=1.0):
    E = 0.0
    for i in range(len(perm) - 1):
        xi, xj = x[perm[i]], x[perm[i+1]]
        if xi > xj:
            gij = 1 + alpha * abs(xi - xj)**beta
            E += gij * (xi - xj)**2
    return E

def build_metric_tensor(x, alpha=1.0, beta=1.0):
    n = len(x)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                G[i,j] = alpha * abs(x[i] - x[j])**beta
        G[i,i] = 1 + np.sum(G[i,:])
    return G

def hybrid_sort(x, max_steps=100000, T=1.0, alpha=1.0, beta=1.0):
    x = np.array(x)
    n = len(x)
    perm = np.arange(n)
    G = build_metric_tensor(x, alpha, beta)
    G_inv = np.linalg.inv(G + 1e-6 * np.eye(n))

    velocity = np.zeros(n)  # momentum vector
    eta = 0.05  # step size
    gamma = 0.9  # momentum decay

    for step in range(max_steps):
        current_energy = energy(x, perm, alpha, beta)
        improved = False

        # Greedy adjacent swaps that reduce energy
        for i in range(n - 1):
            new_perm = perm.copy()
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            new_energy = energy(x, new_perm, alpha, beta)
            if new_energy < current_energy:
                perm = new_perm
                improved = True
                break

        if improved:
            velocity *= gamma  # damp momentum when improving
            continue

        # Approximate gradient with swaps for momentum update
        grad = np.zeros(n)
        for i in range(n):
            dE_sum = 0
            for j in range(n):
                if i == j:
                    continue
                new_perm = perm.copy()
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                dE_sum += energy(x, new_perm, alpha, beta) - current_energy
            grad[i] = dE_sum / (n - 1)

        R_grad = G_inv @ grad

        # Momentum update
        velocity = gamma * velocity - eta * R_grad

        # Propose swaps based on velocity magnitude
        idx_order = np.argsort(-np.abs(velocity))
        accepted = False
        for i in idx_order[:3]:
            j = np.random.choice([k for k in range(n) if k != i])
            new_perm = perm.copy()
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            dE = energy(x, new_perm, alpha, beta) - current_energy
            prob = np.exp(-dE / T) if dE > 0 else 1.0
            if np.random.rand() < prob:
                perm = new_perm
                accepted = True
                break

        # If stuck, do random swap to escape
        if not accepted:
            i, j = np.random.choice(n, 2, replace=False)
            perm[i], perm[j] = perm[j], perm[i]

        # Cool temperature and momentum decay
        T *= 0.995
        velocity *= gamma

        if energy(x, perm, alpha, beta) == 0:
            break

    return x[perm]


import numpy as np
from scipy.optimize import linear_sum_assignment

def sinkhorn_normalization(M, n_iters=20, epsilon=1e-9):
    M_stable = M - np.max(M)
    P = np.exp(M_stable)  # exponentiate safely
    for _ in range(n_iters):
        P_sum_row = P.sum(axis=1, keepdims=True) + epsilon
        P = P / P_sum_row
        P_sum_col = P.sum(axis=0, keepdims=True) + epsilon
        P = P / P_sum_col
    return P

def energy(P, x, x_sorted):
    diff = P @ x - x_sorted
    return np.sum(diff**2)

def grad_energy(P, x, x_sorted):
    return 2 * np.outer((P @ x - x_sorted), x)

def project_to_permutation(P):
    row_ind, col_ind = linear_sum_assignment(-P)  # maximize assignment
    perm_matrix = np.zeros_like(P)
    perm_matrix[row_ind, col_ind] = 1
    return perm_matrix

def hmc_step(P, R, x, x_sorted, epsilon=0.05, L=10):
    M_inv = np.eye(P.shape[0])  # mass matrix inverse

    def potential_energy(P):
        return energy(P, x, x_sorted)

    def potential_grad(P):
        return grad_energy(P, x, x_sorted)

    # Leapfrog integration
    R = R - 0.5 * epsilon * potential_grad(P)
    for _ in range(L):
        P = P + epsilon * (M_inv @ R)
        # Project to doubly stochastic
        P = sinkhorn_normalization(P)
        grad = potential_grad(P)
        R = R - epsilon * grad
    R = R + 0.5 * epsilon * potential_grad(P)

    # Compute Hamiltonians
    current_H = potential_energy(P) + 0.5 * np.sum(R**2)
    proposed_H = potential_energy(P) + 0.5 * np.sum(R**2)

    # Metropolis acceptance
    if np.random.rand() < np.exp(current_H - proposed_H):
        return P, R, True
    else:
        return P, R, False

def hmc_sort(x, steps=500):
    n = len(x)
    x_sorted = np.sort(x)
    # Initialize P near identity
    P = np.eye(n) + 0.01 * np.random.randn(n, n)
    P = sinkhorn_normalization(P)
    R = np.random.randn(n, n)

    for step in range(steps):
        P, R, accepted = hmc_step(P, R, x, x_sorted)
        energy_value = energy(P, x, x_sorted)
        if step % 100 == 0 or accepted:
            print(f"Step {step}, energy={energy_value}, accepted={accepted}")


    # Project final P to permutation
    perm_matrix = project_to_permutation(P)
    permuted_x = perm_matrix @ x

    return permuted_x


is_sorted = lambda arr: all([arr[i] < arr[i + 1] for i in range(len(arr) - 1)])

# Example
total_testcases = 1
mean_iterations = 0
for _ in range(total_testcases):
    x = np.random.permutation(10)
    print("Original:", x)
    sorted_x = hmc_sort(x, 2)
    prev_sorted = sorted_x

    iters = 0
    while not is_sorted(sorted_x):
        sorted_x = hmc_sort(sorted_x, 2)
        iters += 1

        if np.array_equal(prev_sorted, sorted_x):
            pass #sorted_x[0], sorted_x[1] = sorted_x[1], sorted_x[0]
        
        prev_sorted = sorted_x
        print("     Inside-sort:  ", sorted_x)

    print("Sorted:  ", sorted_x)
    print('Iterations needed:', iters)
    mean_iterations += iters

print()
print(mean_iterations / total_testcases)