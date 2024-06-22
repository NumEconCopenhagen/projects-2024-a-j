import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D


class Problem1():
    def __init__(self, **kwargs):
        '''
        Initialize the model with default parameters
        kwargs allow any parameter in the par namespace to be overridden
        '''

        self.par = par = SimpleNamespace() # Create a namespace object for parameters

        # Set default parameters
        self.setup()

        # Update parameters with user input
        for key, value in kwargs.items():
            setattr(par, key, value)

    def setup(self):
        '''
        Set default parameters
        '''
        par = self.par
        par.A = 1.0
        par.gamma = 0.5
        par.alpha = 0.3
        par.nu = 1.0
        par.epsilon = 2.0
        par.tau = 0.0
        par.T = 0.0
        par.kappa = 0.1
    
    def optimal_labor(self, w, p):
        '''
        Defining the optimal labor supply in the economy
        '''
        par = self.par
        return (p * par.A * par.gamma / w) ** (1 / (1 - par.gamma))
    
    def optimal_output(self, w, p):
        '''
        Defining the optimal output in the economy
        '''
        par = self.par
        ell_star = self.optimal_labor(w, p)
        return par.A * (ell_star ** par.gamma)
    
    def optimal_profits(self, w, p):
        '''
        Defining the optimal profits in the economy
        '''
        par = self.par
        return (1 - par.gamma) / par.gamma * w * (self.optimal_labor(w, p) ** (1 / (1 - par.gamma)))
    
    def consumer_behavior(self, w, p1, p2):
        '''
        Defining consumption behavior for a given price and income
        '''
        par = self.par
        pi1 = self.optimal_profits(w, p1)
        pi2 = self.optimal_profits(w, p2)
        income = w * self.optimal_labor(w, p1) + par.T + pi1 + pi2
        c1 = par.alpha * income / p1
        c2 = (1 - par.alpha) * income / (p2 + par.tau)
        return c1, c2

    def utility(self, w, p1, p2):
        '''
        Defining the consumer utility with adjustment to avoid invalid values
        '''
        par = self.par
        c1, c2 = self.consumer_behavior(w, p1, p2)
        ell = self.optimal_labor(w, p1)
        c1 = max(c1, 1e-8)
        c2 = max(c2, 1e-8)
        utility = np.log(c1 ** par.alpha * c2 ** (1 - par.alpha)) - par.nu * (ell ** (1 + par.epsilon)) / (1 + par.epsilon)
        return utility

    def market_clearing(self, w, p1_range, p2_range):
        '''
        Check market clearing for given price ranges
        '''
        p1_vals = np.linspace(p1_range[0], p1_range[1], p1_range[2])
        p2_vals = np.linspace(p2_range[0], p2_range[1], p2_range[2])

        results = []
        clearing_conditions = []

        for p1 in p1_vals:
            for p2 in p2_vals:
                c1, c2 = self.consumer_behavior(w, p1, p2)
                y1 = self.optimal_output(w, p1)
                y2 = self.optimal_output(w, p2)
                market_clearing_p1 = np.isclose(c1, y1, atol=1e-2)
                market_clearing_p2 = np.isclose(c2, y2, atol=1e-2)
                results.append({
                    'p1': round(p1, 2),
                    'p2': round(p2, 2),
                    'Check market clearing p1': market_clearing_p1,
                    'Check market clearing p2': market_clearing_p2
                })
                if market_clearing_p1 or market_clearing_p2:
                    clearing_conditions.append({
                        'Price p1': round(p1, 2),
                        'Price p2': round(p2, 2),
                        'Check market clearing p1': market_clearing_p1,
                        'Check market clearing p2': market_clearing_p2
                    })
        
        df = pd.DataFrame(results)
        df_clearing = pd.DataFrame(clearing_conditions)

        return df, df_clearing
    
    def excess_demand(self, prices):
        '''
        Calculating excess demand for a given set of prices
        '''
        p1, p2 = prices
        w = 1.0  # numeraire
        c1, c2 = self.consumer_behavior(w, p1, p2)
        y1 = self.optimal_output(w, p1)
        y2 = self.optimal_output(w, p2)
        excess_demand_p1 = c1 - y1
        excess_demand_p2 = c2 - y2
        return [excess_demand_p1, excess_demand_p2]
    
    def objective_function(self, prices):
        '''
        Objective function used to minimize excess demand
        '''
        excess_demand = self.excess_demand(prices)
        return sum(ed**2 for ed in excess_demand)

    def find_equilibrium_prices(self, verbose=True):
        '''
        Minimization algorithm used to find the equilibrium prices by minimizing the excess demand
        '''
        initial_guess = [2.0, 2.0]
        solution = minimize(self.objective_function, initial_guess, method='Nelder-Mead')
        if solution.success:
            equilibrium_prices = solution.x
            final_excess_demand = self.excess_demand(equilibrium_prices)
            if verbose:
                print(f"Equilibrium prices found: \n p1 = {equilibrium_prices[0]:.2f}, p2 = {equilibrium_prices[1]:.2f}")
                print(f"Check if good markets have cleared: \n c1 - y1 = {final_excess_demand[0]:.5f}, c2 - y2 = {final_excess_demand[1]:.5f}")
            return equilibrium_prices
        else:
            raise ValueError("Equilibrium prices not found.")
    
    def swf(self, w, p1, p2):
        '''
        Defining the Social Welfare Function (SWF) with the cost term -kappa y_2
        '''
        par = self.par
        utility = self.utility(w, p1, p2)
        y2 = self.optimal_output(w, p2)
        swf = utility - par.kappa * y2
        return swf

    def objective_swf(self, params):
        '''
        Objective function to maximize SWF
        '''
        tau, T = params
        self.par.tau = tau
        self.par.T = T
        w = 1.0  # numeraire
        equilibrium_prices = self.find_equilibrium_prices(verbose=False)
        p1, p2 = equilibrium_prices
        return -self.swf(w, p1, p2)  # Negative because we minimize in scipy

    def maximize_swf(self):
        '''
        Maximizing algorithm which chooses the optimal tau and T values that maximize the SWF
        '''
        initial_guess = [3, 3]
        solution = minimize(self.objective_swf, initial_guess, method='Nelder-Mead')
        if solution.success:
            optimal_tau, optimal_T = solution.x
            print(f"Optimal tau: {optimal_tau:.2f}, Optimal T: {optimal_T:.2f}")
            return optimal_tau, optimal_T
        else:
            raise ValueError("Optimal tau and T not found.")
    
    def plot_swf(self, tau_range, T_range):
        tau_vals = np.linspace(tau_range[0], tau_range[1], tau_range[2])
        T_vals = np.linspace(T_range[0], T_range[1], T_range[2])
        swf_vals = np.zeros((tau_range[2], T_range[2]))

        for i, tau in enumerate(tau_vals):
            for j, T in enumerate(T_vals):
                swf_vals[i, j] = -self.objective_swf([tau, T])

        tau_grid, T_grid = np.meshgrid(tau_vals, T_vals)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(tau_grid, T_grid, swf_vals, cmap='viridis')
        ax.set_xlabel('tau')
        ax.set_ylabel('T')
        ax.set_zlabel('SWF')
        plt.show()

class Problem3():
    def __init__(self, random_seed=2024):
        self.rng = np.random.default_rng(random_seed)
        self.X = self.rng.uniform(size=(50, 2))
        self.y = self.rng.uniform(size=(2, ))
        self.A = None
        self.B = None
        self.C = None
        self.D = None

    def find_points(self):
        y = self.y
        X = self.X

        def safe_min(sequence):
            try:
                return min(sequence, key=lambda x: np.linalg.norm(x - y))
            except ValueError:
                return None

        self.A = safe_min((x for x in X if x[0] > y[0] and x[1] > y[1]))
        self.B = safe_min((x for x in X if x[0] > y[0] and x[1] < y[1]))
        self.C = safe_min((x for x in X if x[0] < y[0] and x[1] < y[1]))
        self.D = safe_min((x for x in X if x[0] < y[0] and x[1] > y[1]))

    def plot_points_and_triangles(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.X[:, 0], self.X[:, 1], color='blue', label='Points in X')
        plt.scatter(self.y[0], self.y[1], color='red', label='Point y', zorder=5)

        if self.A is not None:
            plt.scatter(*self.A, color='green', label='Point A', zorder=5)
        if self.B is not None:
            plt.scatter(*self.B, color='purple', label='Point B', zorder=5)
        if self.C is not None:
            plt.scatter(*self.C, color='orange', label='Point C', zorder=5)
        if self.D is not None:
            plt.scatter(*self.D, color='brown', label='Point D', zorder=5)

        if self.A is not None and self.B is not None and self.C is not None:
            plt.plot([self.A[0], self.B[0]], [self.A[1], self.B[1]], 'k-')
            plt.plot([self.B[0], self.C[0]], [self.B[1], self.C[1]], 'k-')
            plt.plot([self.C[0], self.A[0]], [self.C[1], self.A[1]], 'k-', label='Triangle ABC')

        if self.C is not None and self.D is not None and self.A is not None:
            plt.plot([self.C[0], self.D[0]], [self.C[1], self.D[1]], 'k--')
            plt.plot([self.D[0], self.A[0]], [self.D[1], self.A[1]], 'k--', label='Triangle CDA')
            
        plt.style.use('fivethirtyeight')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.title('Points and Triangles')
        plt.grid(True)
        plt.show()

    def barycentric_coordinates(self, y, A, B, C):
        denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r1 = ((B[1] - C[1]) * (y[0] - C[0]) + (C[0] - B[0]) * (y[1] - C[1])) / denom
        r2 = ((C[1] - A[1]) * (y[0] - C[0]) + (A[0] - C[0]) * (y[1] - C[1])) / denom
        r3 = 1 - r1 - r2
        return r1, r2, r3

    def check_triangle(self, r1, r2, r3):
        return 0 <= r1 <= 1 and 0 <= r2 <= 1 and 0 <= r3 <= 1

    def solve(self):
        self.find_points()
        if self.A is None or self.B is None or self.C is None or self.D is None:
            return {"error": "Not enough points to form triangles."}

        rABC1, rABC2, rABC3 = self.barycentric_coordinates(self.y, self.A, self.B, self.C)
        rCDA1, rCDA2, rCDA3 = self.barycentric_coordinates(self.y, self.C, self.D, self.A)

        if self.check_triangle(rABC1, rABC2, rABC3):
            containing_triangle = 'ABC'
        elif self.check_triangle(rCDA1, rCDA2, rCDA3):
            containing_triangle = 'CDA'
        else:
            containing_triangle = 'None'

        self.plot_points_and_triangles()

        result = {
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "D": self.D,
            "y": self.y,
            "rABC": (rABC1, rABC2, rABC3),
            "rCDA": (rCDA1, rCDA2, rCDA3),
            "containing_triangle": containing_triangle
        }

        return result

    def compute_approximation(self, f):
        # First, solve to find the points and barycentric coordinates
        result = self.solve()

        if "error" in result:
            return None, None

        # Get the barycentric coordinates and the containing triangle
        rABC = result["rABC"]
        rCDA = result["rCDA"]
        containing_triangle = result["containing_triangle"]

        if containing_triangle == "ABC":
            # Calculate the approximation of f(y) using triangle ABC
            fA = f(self.A)
            fB = f(self.B)
            fC = f(self.C)
            f_approx = rABC[0] * fA + rABC[1] * fB + rABC[2] * fC
        elif containing_triangle == "CDA":
            # Calculate the approximation of f(y) using triangle CDA
            fC = f(self.C)
            fD = f(self.D)
            fA = f(self.A)
            f_approx = rCDA[0] * fC + rCDA[1] * fD + rCDA[2] * fA
        else:
            return None, None

        # Calculate the true value of f(y)
        f_true = f(self.y)

        return f_approx, f_true

    def compute_approximations_for_set(self, Y, f):
        results = []
        for point in Y:
            self.y = np.array(point)
            f_approx, f_true = self.compute_approximation(f)
            if f_approx is not None and f_true is not None:
                results.append((point, f_approx, f_true))
            else:
                results.append((point, None, None))
        return results

