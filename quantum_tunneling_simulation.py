import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# Параметры
hbar = 1.0  # редуцированная постоянная Планка (условная единица)
m = 1.0     # масса частицы (условная единица)
L = 10.0    # длина области по x
N = 1000    # число точек сетки
dx = L / N  # шаг по x

x = np.linspace(0, L, N)

# Потенциальный барьер
V0 = 6.0       # высота барьера
a = 2.0        # ширина барьера
barrier_start = L/2 - a/2
barrier_end = L/2 + a/2

V = np.zeros(N)
for i in range(N):
    if barrier_start <= x[i] <= barrier_end:
        V[i] = V0

# Формируем матрицу гамильтониана в дискретном виде (трёхдиагональная)
# Диагональ
main_diag = (hbar**2 / (m * dx**2)) + V
# Побочные диагонали
off_diag = np.full(N-1, -hbar**2 / (2 * m * dx**2))

# Решаем задачу на собственные значения: H psi = E psi
E, psi = eigh_tridiagonal(main_diag, off_diag)

# Выберем первый собственный уровень выше потенциала (ищем туннельный эффект)
idx = np.where(E > 0)[0][0]  # первый уровень с энергией > 0

wavefunc = psi[:, idx]
# Нормируем волновую функцию
wavefunc = wavefunc / np.sqrt(np.sum(np.abs(wavefunc)**2)*dx)

# Визуализация
plt.figure(figsize=(10,6))
plt.plot(x, wavefunc**2, label=r'$|\psi(x)|^2$ (вероятность)')
plt.plot(x, V / V0 * np.max(wavefunc**2), 'r--', label='Потенциал $V(x)$ (норм.)')
plt.title(f'Квантовое туннелирование: уровень энергии E={E[idx]:.2f}')
plt.xlabel('x')
plt.ylabel(r'$|\psi(x)|^2$')
plt.legend()
plt.grid()
plt.show()
