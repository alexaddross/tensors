from qiskit.visualization import plot_bloch_vector
from qiskit.quantum_info import Statevector, Operator
import matplotlib.pyplot as plt
import numpy as np


def form_metric_theta_tensor(theta: int) -> Operator:
    '''
    This function generates metric-like tensor for polarization filter  , based on given polarization angle.
    '''
        
    theta = theta * np.pi / 180 # Convert angle to radians, since NumPy's sin and cos works with radians

    return Operator(
      np.array(
          [
              [np.cos(theta) ** 2,  np.cos(theta) * np.sin(theta)],
              [np.cos(theta) * np.sin(theta), np.sin(theta) ** 2]
          ]
      )
    )

# QuBit State |ψ> = α|0⟩ + β|1⟩
alpha = 1 / np.sqrt(2)
beta  = 1 / np.sqrt(2)

# Create state vector for quant
psi = Statevector([alpha, beta])
psi_vector = np.array([alpha, beta])
print("Initial state |ψ⟩:", psi)

# Create polarizator tensor and dot it with Psi
theta_tensor = form_metric_theta_tensor(90)
result = theta_tensor @ psi
print("\nProjected state (after measurement tensor):\n", result)

# Double tensor contraction method
double_contraction = np.vdot(psi_vector, theta_tensor @ psi_vector).real

print(f"\nDouble tensor contraction method output:\n{double_contraction}")

# Calculate probability of the projections
probability = np.abs(result.data) ** 2
print("\nProjection probability (⟨ψ|M|ψ⟩):", probability.sum())

# Get neccessary data for plotting the Bloch sphere and qubit state
delta0 = np.angle(alpha)
delta1 = np.angle(beta)
r0 = np.abs(alpha)
r1 = np.abs(beta)

theta = 2 * np.arccos(r0)
phi = delta1 - delta0

x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

bloch_vector = [x, y, z]

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
plot_bloch_vector(bloch_vector, ax=ax)
plt.show()
