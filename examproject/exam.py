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
        # Create a namespace object for parameters
        self.par = par = SimpleNamespace()

        # Set default parameters
        self.setup()

        # Update parameters with user input
        for key, value in kwargs.items():
            setattr(par, key, value)

    def setup(self):
        '''
        Set default parameters
        '''
        # Parameters as given in the problem description
        par = self.par
        par.A = 1.0
        par.gamma = 0.5
        par.alpha = 0.3
        par.nu = 1.0
        par.epsilon = 2.0
        par.tau = 0.0
        par.T = 0.0
        par.kappa = 0.1
    
    def reset_parameters(self):
        '''
        Method for resetting parameters
        '''
        # Resetting parameters to ensure reliable results
        self.setup()


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
        # Define optimal ell as given in the problem description
        ell_star = self.optimal_labor(w, p)
        return par.A * (ell_star ** par.gamma)
    
    def optimal_profits(self, w, p):
        '''
        Defining the optimal profits in the economy
        '''
        par = self.par
        # Define optimal profits as given in the problem description
        return (1 - par.gamma) / par.gamma * w * (self.optimal_labor(w, p) ** (1 / (1 - par.gamma)))
    
    def consumer_behavior(self, w, p1, p2):
        '''
        Defining consumption behavior for a given price and income
        '''
        par = self.par
        pi1 = self.optimal_profits(w, p1)
        pi2 = self.optimal_profits(w, p2)
        # Initial guess for c2 to calculate T
        c2_initial_guess = 1.0  
        par.T = par.tau * c2_initial_guess
        income = w * self.optimal_labor(w, p1) + par.T + pi1 + pi2
        c1 = par.alpha * income / p1
        c2 = (1 - par.alpha) * income / (p2 + par.tau)
        # Update T to match the government constraint T = tau * c2
        par.T = par.tau * c2 
        return c1, c2

    def utility(self, w, p1, p2):
        '''
        Defining the consumer utility with adjustment to avoid invalid values
        '''
        par = self.par
        c1, c2 = self.consumer_behavior(w, p1, p2)
        ell = self.optimal_labor(w, p1)
        # Ensure non-zero consumption
        c1 = max(c1, 1e-8)
        c2 = max(c2, 1e-8)
        # Define househould utility as given in the problem description
        utility = np.log(c1 ** par.alpha * c2 ** (1 - par.alpha)) - par.nu * (ell ** (1 + par.epsilon)) / (1 + par.epsilon)
        return utility

    def market_clearing(self, w, p1_range, p2_range):
        '''
        Check market clearing for given price ranges
        '''
        # Define price values within the linspace given in the problem description
        p1_vals = np.linspace(p1_range[0], p1_range[1], p1_range[2])
        p2_vals = np.linspace(p2_range[0], p2_range[1], p2_range[2])
        # Initialize an empty list for results
        results = []
        clearing_conditions = []
        # Loop over the price values and check market clearing
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
        # Store results in a pandas dataframe for neat output in notebook
        df = pd.DataFrame(results)
        df_clearing = pd.DataFrame(clearing_conditions)

        return df, df_clearing
    
    def excess_demand(self, prices):
        '''
        Calculating excess demand for a given set of prices
        '''
        p1, p2 = prices
        # w is numeraire
        w = 1.0
        c1, c2 = self.consumer_behavior(w, p1, p2)
        y1 = self.optimal_output(w, p1)
        y2 = self.optimal_output(w, p2)
        # Calculate excess demand to check market clearing
        excess_demand_p1 = c1 - y1
        excess_demand_p2 = c2 - y2
        return [excess_demand_p1, excess_demand_p2]
    
    def objective_function(self, prices):
        '''
        Objective function used to minimize excess demand
        '''
        excess_demand = self.excess_demand(prices)
        # Return sum of squared excess demands to ensure deviations contributing positively to the objective function
        return sum(ed**2 for ed in excess_demand)

    def find_equilibrium_prices(self, verbose=True):
        '''
        Minimization algorithm used to find the equilibrium prices by minimizing the excess demand
        '''
        # Define initial guesses for the prices
        initial_guess = [2.0, 2.0]
        # Calling a solver to minimize the objective function
        solution = minimize(self.objective_function, initial_guess, method='Nelder-Mead')
        # Printing results if solution has been achieved
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
        Defining the Social Welfare Function (SWF)
        '''
        par = self.par
        utility = self.utility(w, p1, p2)
        y2 = self.optimal_output(w, p2)
        # Defining SWF as given in the problem description
        swf = utility - par.kappa * y2
        return swf

    def objective_swf(self, params):
        '''
        Objective function to maximize SWF
        '''
        tau = params[0]
        self.par.tau = tau
        # numeraire
        w = 1.0 
        equilibrium_prices = self.find_equilibrium_prices(verbose=False)
        p1, p2 = equilibrium_prices
        _, c2 = self.consumer_behavior(w, p1, p2)
        # Ensure the government constraint T = tau * c2 is satisfied
        self.par.T = tau * c2 
        # Return negative objective function because solver is minimizing
        return -self.swf(w, p1, p2)

    def maximize_swf(self):
        '''
        Maximizing algorithm which chooses the optimal tau value that maximizes the SWF
        '''
        # Initial guess for optimal tau
        initial_guess = [3]
        # Calling a solver to maximize the objective function
        solution = minimize(self.objective_swf, initial_guess, method='Nelder-Mead')
        # Print results if solution has been found
        if solution.success:
            optimal_tau = solution.x[0]
            _, c2 = self.consumer_behavior(1.0, *self.find_equilibrium_prices(verbose=False))
            # Calculating T from the government constraint
            optimal_T = optimal_tau * c2
            print(f"Optimal tau: {optimal_tau:.2f}, Optimal T: {optimal_T:.2f}")
            return optimal_tau, optimal_T
        else:
            raise ValueError("Optimal tau and T not found.")

class Problem2:
    def __init__(self):
        '''
        Initialize the model with default parameters, seed and epsilon.
        '''
        # Initialize parameters
        self.par = SimpleNamespace()
        self.par.J = 3
        self.par.N = 10
        self.par.K = 10000
        self.par.F = np.arange(1, self.par.N + 1)
        self.par.sigma = 2
        self.par.v = np.array([1, 2, 3])
        self.par.c = 1

        # Set random seed for reproducibility
        np.random.seed(1919)
        # Generate epsilon values
        self.epsilon = np.random.normal(0, self.par.sigma, (self.par.J, self.par.K))

    def calculate_expected_utility(self):
        '''
        Calculating and displaying the expected utility and the average realized utility
        '''
        # Calculate expected utility
        self.expected_utility = self.par.v + np.mean(self.epsilon)
        # Calculate realized utility
        self.realized_utility = self.par.v[:, None] + self.epsilon
        self.average_realized_utility = np.mean(self.realized_utility, axis=1)
        # Store results in a DataFrame
        self.results = pd.DataFrame({
            'Career Track': np.arange(1, self.par.J + 1),
            'Expected Utility': self.expected_utility,
            'Average Realized Utility': self.average_realized_utility
        })
        # Display results
        display(self.results)
    
    def calculate_prior_expected_utility(self, v_j, F_i, sigma, K):
        '''
        Calculating the prior given values - returns a numpy-array.
        '''
        # Generate noise
        epsilon_friend = np.random.normal(0, sigma, (self.par.J, F_i, K))
        # Calculate expected utility
        prior_expected_utility = v_j[:, None] + np.mean(epsilon_friend, axis=1)
        return prior_expected_utility
    
    def simulate_process(self):
        '''
        Simulating the process for all F, and calculating career choices,
        Furthermore, calculating individual expected utilities, and realized utilities.
        '''

        # List to store value
        self.collected_prior_expected_utilities = []

        # Calculate prior expected utilities for all F
        for i in self.par.F:
            prior_expected_utility = self.calculate_prior_expected_utility(self.par.v, i, self.par.sigma, self.par.K)
            self.collected_prior_expected_utilities.append(np.mean(prior_expected_utility, axis=1))

        # Initialize arrays for values
        self.career_choices = np.zeros((self.par.N, self.par.K), dtype=int)
        self.ind_expected_utilities = np.zeros((self.par.N, self.par.K))
        self.realized_utilities = np.zeros((self.par.N, self.par.K))

        # Simulate career choices
        for i in range(self.par.N):
            F_i = i + 1
            prior_expected_utility = self.calculate_prior_expected_utility(self.par.v, F_i, self.par.sigma, self.par.K)
            epsilon_graduate = np.random.normal(0, self.par.sigma, (self.par.J, self.par.K))
            
            # Choose highest utility
            for k in range(self.par.K):
                j_star = np.argmax(prior_expected_utility[:, k])
                self.career_choices[i, k] = j_star
                self.ind_expected_utilities[i, k] = prior_expected_utility[j_star, k]
                self.realized_utilities[i, k] = self.par.v[j_star] + epsilon_graduate[j_star, k]
        
        # Initialize arrays for values
        self.career_share = np.zeros((self.par.N, self.par.J))
        self.average_ind_utility = np.zeros(self.par.N)
        self.average_realized_utility = np.zeros(self.par.N)

        # Shares and utilities
        for i in range(self.par.N):
            for j in range(self.par.J):
                self.career_share[i, j] = np.mean(self.career_choices[i] == j)
            self.average_ind_utility[i] = np.mean(self.ind_expected_utilities[i])
            self.average_realized_utility[i] = np.mean(self.realized_utilities[i])
    
    def plot_results(self):
        '''
        Plotting the results of the simulation
        '''
        # Subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

        # Shares - careers
        width = 0.25
        x = np.arange(self.par.N) + 1
        axes[0].bar(x - width, self.career_share[:, 0], width, label='j = 1')
        axes[0].bar(x, self.career_share[:, 1], width, label='j = 2')
        axes[0].bar(x + width, self.career_share[:, 2], width, label='j = 3')
        axes[0].set_xlabel('i')
        axes[0].set_ylabel('Share Choosing Career')
        axes[0].set_title('Share of Graduates Choosing Each Career')
        axes[0].legend()

        # Utilities
        axes[1].plot(x, self.average_ind_utility, marker='o', label='Average Individual Expected Utility')
        axes[1].plot(x, self.average_realized_utility, marker='o', label='Average Realized Utility')
        axes[1].set_xlabel('i')
        axes[1].set_ylabel('Utility')
        axes[1].set_title('Average Utility')
        axes[1].legend()
        axes[1].grid(True)

        plt.style.use('fivethirtyeight')
        plt.tight_layout()
        plt.show()
    
    def simulate_new_optimal_career_choice(self):
        '''
        Simulating the new optimal career choice after one year, including the cost of switching careers.
        '''
        # Create arrays
        self.new_career_choices = np.zeros((self.par.N, self.par.K), dtype=int)
        self.new_ind_expected_utilities = np.zeros((self.par.N, self.par.K))
        self.new_realized_utilities = np.zeros((self.par.N, self.par.K))

        # Simulate new career choice process
        for i in range(self.par.N):
            F_i = i + 1
            prior_expected_utility = self.calculate_prior_expected_utility(self.par.v, F_i, self.par.sigma, self.par.K)
            epsilon_graduate = np.random.normal(0, self.par.sigma, (self.par.J, self.par.K))
            
            for k in range(self.par.K):
                #Initial career
                j_star = np.argmax(prior_expected_utility[:, k])
                initial_utility = self.par.v[j_star] + epsilon_graduate[j_star, k]
                
                # prior utility incl. costs
                new_prior_utility = prior_expected_utility - self.par.c
                new_prior_utility[j_star] = initial_utility
                
                # Choose new career by highest utility
                new_j_star = np.argmax(new_prior_utility[:, k])
                self.new_career_choices[i, k] = new_j_star
                self.new_ind_expected_utilities[i, k] = new_prior_utility[new_j_star, k]
                self.new_realized_utilities[i, k] = self.par.v[new_j_star] + epsilon_graduate[new_j_star, k] - (self.par.c if new_j_star != j_star else 0)
        
        # Arrays for values
        self.new_career_share = np.zeros((self.par.N, self.par.J))
        self.new_average_ind_utility = np.zeros(self.par.N)
        self.new_average_realized_utility = np.zeros(self.par.N)
        self.switching_share = np.zeros((self.par.N, self.par.J))

        # Calculate shares and utility
        for i in range(self.par.N):
            for j in range(self.par.J):
                self.new_career_share[i, j] = np.mean(self.new_career_choices[i] == j)
            self.new_average_ind_utility[i] = np.mean(self.new_ind_expected_utilities[i])
            self.new_average_realized_utility[i] = np.mean(self.new_realized_utilities[i])
            
            # Switching shares
            for j in range(self.par.J):
                self.switching_share[i, j] = np.mean((self.career_choices[i] == j) & (self.new_career_choices[i] != j))

    def plot_new_results(self):
        '''
        Plotting the results of the new simulation
        '''
        # Subplots
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))

        width = 0.25
        x = np.arange(self.par.N) + 1

        # New shares
        axes[0].bar(x - width, self.new_career_share[:, 0], width, label='j = 1')
        axes[0].bar(x, self.new_career_share[:, 1], width, label='j = 2')
        axes[0].bar(x + width, self.new_career_share[:, 2], width, label='j = 3')
        axes[0].set_xlabel('i')
        axes[0].set_ylabel('Share Choosing Career')
        axes[0].set_title('Share of Graduates Choosing Each Career After One Year')
        axes[0].legend()

        # New utilities
        axes[1].plot(x, self.new_average_ind_utility, marker='o', label='Average Individual Expected Utility')
        axes[1].plot(x, self.new_average_realized_utility, marker='o', label='Average Realized Utility')
        axes[1].set_xlabel('i')
        axes[1].set_ylabel('Utility')
        axes[1].set_title('Average Utility After One Year')
        axes[1].legend()
        axes[1].grid(True)

        # Shares switching
        for j in range(self.par.J):
            axes[2].plot(x, self.switching_share[:, j], marker='o', label=f'Switched from j = {j + 1}')
        axes[2].set_xlabel('i')
        axes[2].set_ylabel('Switching Share')
        axes[2].set_title('Share of Graduates Switching Careers After One Year')
        axes[2].legend()
        axes[2].grid(True)

        plt.style.use('fivethirtyeight')
        plt.tight_layout()
        plt.show()

class Problem3():
    def __init__(self, random_seed=2024):
        '''
        Initialize the class with the seed and the sample points given given in the problem description
        '''
        # Define X and y as given by the sample code in the problem description
        self.rng = np.random.default_rng(random_seed)
        self.X = self.rng.uniform(size=(50, 2))
        self.y = self.rng.uniform(size=(2, ))
        self.A = None
        self.B = None
        self.C = None
        self.D = None

    def find_points(self):
        '''
        Finds the sets of points in X that minimizes the Euclidean distance between x and y points
        '''
        y = self.y
        X = self.X

        def distance(x, y):
            '''
            Defines the Euclidean distance between x and y points
            '''
            return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

        def safe_min(sequence):
            '''
            Minimizes the Euclidean distance, here called "sequence"
            '''
            sequence = list(sequence)
            if sequence:
                return min(sequence, key=lambda x: distance(x, y))
            else:
                return None

        # find points A, B, C, D using the given minimization conditions
        self.A = safe_min((x for x in X if x[0] > y[0] and x[1] > y[1]))
        self.B = safe_min((x for x in X if x[0] > y[0] and x[1] < y[1]))
        self.C = safe_min((x for x in X if x[0] < y[0] and x[1] < y[1]))
        self.D = safe_min((x for x in X if x[0] < y[0] and x[1] > y[1]))

    def plot_points_and_triangles(self):
        '''
        Plots all points and triangles in a combined plot
        '''
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
        plt.grid(True)
        plt.show()

    def barycentric_coordinates(self, y, A, B, C):
        '''
        Calculates the barycentric coordinates for given points
        '''
        # Barycentric coodinates calculated using the formulas given in the problem description
        denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r1 = ((B[1] - C[1]) * (y[0] - C[0]) + (C[0] - B[0]) * (y[1] - C[1])) / denom
        r2 = ((C[1] - A[1]) * (y[0] - C[0]) + (A[0] - C[0]) * (y[1] - C[1])) / denom
        r3 = 1 - r1 - r2
        return r1, r2, r3

    def check_triangle(self, r1, r2, r3):
        '''
        Checks whether a set of barycentric coordinates corresponds to a point that lies inside a triangle.
        '''
        return 0 <= r1 <= 1 and 0 <= r2 <= 1 and 0 <= r3 <= 1

    def solve(self):
        '''
        Calculates the barycentric coordinates of point y w.r.t. the triangles ABC and CDA, 
        and returns which triangle y lies within
        '''
        # Ensuring all points exist
        self.find_points()
        if self.A is None or self.B is None or self.C is None or self.D is None:
            return {"error": "Not enough points to form triangles."}

        rABC1, rABC2, rABC3 = self.barycentric_coordinates(self.y, self.A, self.B, self.C)
        rCDA1, rCDA2, rCDA3 = self.barycentric_coordinates(self.y, self.C, self.D, self.A)
        # Checks which triangle y lies within
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
        '''
        Calculates an approximation of f(y)
        '''
        # Solve to find the points and barycentric coordinates
        result = self.solve()

        if "error" in result:
            return None, None

        # Get the barycentric coordinates and the containing triangle
        rABC = result["rABC"]
        rCDA = result["rCDA"]
        containing_triangle = result["containing_triangle"]
        # Calculate the approximation of f(y) using triangle ABC
        if containing_triangle == "ABC":
            fA = f(self.A)
            fB = f(self.B)
            fC = f(self.C)
            f_approx = rABC[0] * fA + rABC[1] * fB + rABC[2] * fC
        # Calculate the approximation of f(y) using triangle CDA
        elif containing_triangle == "CDA":
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
        '''
        Calculates approximations of f(y) for all y in Y
        '''
        results = []
        for point in Y:
            self.y = np.array(point)
            f_approx, f_true = self.compute_approximation(f)
            if f_approx is not None and f_true is not None:
                results.append((point, f_approx, f_true))
            else:
                results.append((point, None, None))
        return results

