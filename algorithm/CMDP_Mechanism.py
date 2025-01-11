import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
from math import sqrt
import copy

class CMDPMechanism:
    def __init__(self, epsilon, delta2_f, batch_size, m=1,
                 lambda_range=(0.01, 2.0),
                 tol=1e-6, max_iter=100):

        self.epsilon = epsilon
        self.delta2_f = delta2_f
        self.batch_size = batch_size
        self.m = m
        self.tol = tol
        self.max_iter = max_iter

        self.gamma = 1 + 1 / self.batch_size

        self.lambda_min, self.lambda_max = lambda_range

        self.lamda = None
        self.N = None
        self.B = None
        self.L = None
        self.T = None


        self._find_lambda()


    def _solve_N_B_reference(self, lamda):
        gamma = self.gamma
        L = gamma * self.delta2_f
        T = (lamda * self.epsilon) / L

        def system(X):
            N, B = X

            F1 = N * L - L - 2 * B - self.delta2_f

            Z, _ = integrate.quad(lambda x: np.exp(-T ** 2 * x ** 2), -L, N * L)

            I, _ = integrate.quad(lambda x: x * np.exp(-T ** 2 * x ** 2), -L, N * L)

            if Z == 0:
                F2 = 1e6
            else:
                F2 = (I / Z) - B

            return [F1, F2]

        X_initial = [5.0, 0.2]

        sol, infodict, ier, mesg = optimize.fsolve(system, X_initial, xtol=self.tol, full_output=True)
        N_sol, B_sol = sol

        if ier != 1:
            print(f"fsolve did not converge: {mesg}")

        return N_sol, B_sol

    def _check_lambda_constraint(self, lamda, N, B):
        gamma = self.gamma
        L = self.delta2_f * gamma
        W = (N + 1) + 2 * abs(B) / L
        lambda_max_constraint = sqrt(1 / (2 * self.epsilon * self.m * W))

        if lamda > lambda_max_constraint:
            return False, lambda_max_constraint
        return True, lambda_max_constraint

    def _find_lambda(self):
        lamda_low = self.lambda_min
        lamda_high = self.lambda_max
        best_lamda = None

        for _ in range(self.max_iter):
            lamda_mid = (lamda_low + lamda_high) / 2
            try:
                N, B = self._solve_N_B_reference(lamda_mid)
            except Exception as e:
                print(f"Error solving for lambda={lamda_mid}: {e}")
                break

            valid, lambda_max_constraint = self._check_lambda_constraint(lamda_mid, N, B)
            if valid:
                best_lamda = lamda_mid
                lamda_low = lamda_mid
            else:
                lamda_high = lambda_max_constraint

        if best_lamda is not None:
            self.lamda = best_lamda
            self.L = self.gamma * self.delta2_f
            self.T = (self.lamda * self.epsilon) / self.L
            self.N, self.B = self._solve_N_B_reference(self.lamda)
        else:
            raise ValueError("Cannot find a lambda。")

    def _compute_normalization_factor(self, I: int) -> float:
        def density(y):
            return np.exp(-self.T ** 2 * y ** 2)

        if I == 1:
            integral, _ = integrate.quad(density, -self.L - self.B, self.N * self.L - self.B)
        elif I == -1:
            integral, _ = integrate.quad(density, -self.N * self.L + self.B, self.L + self.B)
        else:
            raise ValueError("I must be 1 or -1.")

        C = 1.0 / integral
        return C

    def _density_function(self, x: float, I: int) -> float:
        C = self._compute_normalization_factor(I)

        if I == 1:

            density = C * np.exp(-self.T ** 2 * (x + self.B) ** 2)

            mask = (-self.L - self.B <= x) & (x <= self.N * self.L - self.B)
        elif I == -1:

            density = C * np.exp(-self.T ** 2 * (x - self.B) ** 2)

            mask = (-self.N * self.L + self.B <= x) & (x <= self.L + self.B)
        else:
            raise ValueError("I must be 1 or -1.")

        density = np.where(mask, density, 0.0)
        return density

    def mixed_density_function(self, x: float) -> float:

        rho1 = self._density_function(x, 1)
        rho2 = self._density_function(x, -1)
        return 0.5 * rho1 + 0.5 * rho2

    def mixed_density_function_vectorized(self, x: np.ndarray) -> np.ndarray:

        rho1 = self._density_function(x, 1)
        rho2 = self._density_function(x, -1)
        return 0.5 * rho1 + 0.5 * rho2

    def generate_random_samples(self, num):
        x_min = -self.N * self.L + self.B
        x_max = self.N * self.L - self.B

        x_grid = np.linspace(x_min, x_max, 1000)
        y_max = np.max(self.mixed_density_function_vectorized(x_grid))

        samples = []

        batch_size = num * 2

        while len(samples) < num:

            x_rand = np.random.uniform(x_min, x_max, batch_size)
            y_rand = np.random.uniform(0, y_max, batch_size)


            rho_rand = self.mixed_density_function_vectorized(x_rand)


            accept = y_rand <= rho_rand
            accepted_samples = x_rand[accept]


            samples.extend(accepted_samples.tolist())


        return np.array(samples[:num])

    def plot_samples_histogram(self, num=10000, bins=50):

        samples = self.generate_random_samples(num)


        plt.figure(figsize=(10, 6))
        count, bins_edges, _ = plt.hist(samples, bins=bins, density=True, alpha=0.6, color='g', label='Sampled Data')


        x = np.linspace(-self.N * self.L + self.B, self.N * self.L - self.B, 1000)
        rho_mix = self.mixed_density_function_vectorized(x)
        plt.plot(x, rho_mix, 'r-', label='Mixed Density Function')

        plt.title('Histogram of Generated Samples vs Mixed Density Function')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

    def compute_var_rho1(self):

        x_min = -self.L - self.B
        x_max = self.N * self.L - self.B

        # E[X]
        def integrand_x(x):
            return x * self._density_function(x, 1)

        E_x, _ = integrate.quad(integrand_x, x_min, x_max)

        # E[X^2]
        def integrand_x2(x):
            return x ** 2 * self._density_function(x, 1)

        E_x2, _ = integrate.quad(integrand_x2, x_min, x_max)

        var_rho1 = E_x2 - (E_x ** 2)
        return var_rho1

    def compute_var_mixed(self):
        x_min = min(-self.L - self.B, -self.N * self.L + self.B)
        x_max = max(self.N * self.L - self.B, self.L + self.B)

        def integrand_x_mixed(x):
            return x * self.mixed_density_function(x)

        def integrand_x2_mixed(x):
            return x ** 2 * self.mixed_density_function(x)

        # 可尝试增大limit、放宽精度要求
        E_x_mixed, _ = integrate.quad(integrand_x_mixed, x_min, x_max, limit=200, epsabs=1e-9, epsrel=1e-9)
        E_x2_mixed, _ = integrate.quad(integrand_x2_mixed, x_min, x_max, limit=200, epsabs=1e-9, epsrel=1e-9)

        var_mixed = E_x2_mixed - (E_x_mixed ** 2)
        return var_mixed



class hyper_para_setup:
    def __init__(self, lr_ini, lr_T_gain, lr_min, lr_max, epsilon_T_acc):
        self.lr = lr_ini
        self.epsilon = 2

        self.update_index = 0
        self.acc_ave_current = 0
        self.acc_ave_last = 0
        self.acc_ave = 0

        self.lr_T_gain = lr_T_gain
        self.rate_lr = 0.1
        self.lr_min = lr_min
        self.lr_max = lr_max

        self.epsilon_T_acc = epsilon_T_acc
        self.rate_epsilon = 0.1
        self.epsilon_min = 2
        self.epsilon_max = 5

        self.lr_new = 0
        self.epsilon_new = 0

    def hyper_parameter_update(self, test_accuracy, test_loss, lr_used, epsilon_used, iter_num, N_epoch):
        if (test_loss <= 0.2322):
            lr_new = 0.1
            epsilon_new = 5
            return lr_new, epsilon_new

        if (iter_num % N_epoch) == 0:
            self.update_index = self.update_index + 1
            if self.update_index == 1:
                self.acc_ave_current = self.acc_ave/N_epoch
                self.acc_ave = 0
            else:
                self.acc_ave_last = copy.deepcopy(self.acc_ave_current)
                self.acc_ave_current = self.acc_ave/N_epoch
                self.acc_ave = 0

                # adjusting the learning rate
                if (self.acc_ave_current/self.acc_ave_last <= self.lr_T_gain):
                    self.lr_new = max(lr_used * (1 - self.rate_lr), self.lr_min)
                else:
                    self.lr_new = min(lr_used * (1 + self.rate_lr), self.lr_max)

                if (self.acc_ave_current <= self.epsilon_T_acc):
                    self.epsilon_new = max(epsilon_used * (1 - self.rate_epsilon), self.epsilon_min)
                else:
                    self.epsilon_new = min(epsilon_used * (1 + self.rate_epsilon), self.epsilon_max)

                self.acc_ave += test_accuracy
                return self.lr_new, self.epsilon_new

        self.acc_ave += test_accuracy

        return None, None
