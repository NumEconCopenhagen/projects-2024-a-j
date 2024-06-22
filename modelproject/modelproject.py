
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy import optimize
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})


# Defining the model as a class
class NewKeynesianModel():

    def __init__(self, **kwargs):
        '''
        Initialize the model with default parameters
        kwargs allow any parameter in the par namespace to be overridden by the user
        '''

        self.par = par = SimpleNamespace() # Create a namespace object for parameters
        self.sol = sol = SimpleNamespace() # Create a namespace object for solution results
        self.sim = sim = SimpleNamespace() # Create a namespace object for simulation results (not always neccesary)

        # Set default parameters
        self.setup()

        # Update parameters with user input
        for key, value in kwargs.items():
            setattr(par, key, value)

        # Allocate arrays simulation (if needed)
        self.allocate()

        # Draw shocks
        self.simulate()


    def setup(self):
        '''
        Set default parameters
        '''
        par = self.par

        par.sigma = 1.5  # Intertemporal elasticity of substitution
        par.beta = 0.99  # Discount rate
        par.rho = 0.9  # Persistence of the exogenous shock
        par.phi = 1.5  # Taylor rule coefficient
        par.phi_y = 0.5  # Response coefficient to output
        par.alpha = 0.75  # Share parameter in Phillips curve
        par.kappa = (par.alpha / (1 - par.alpha)) * (1 - par.beta * (1 - par.alpha)) * par.phi  # Sensitivity of inflation to output

        # Simulation options
        par.T = 10  # Number of time periods to simulate

    def allocate(self):
        '''
        Allocate arrays for simulation
        '''
        par = self.par
        sim = self.sim

        # Ensure all variables including shocks are initialized
        simvarnames = ['y', 'pi', 'i', 'a', 'epsilon_a', 'epsilon_m']  # Including both state variables and shock variables

        for varname in simvarnames:
            sim.__dict__[varname] = np.nan * np.ones(par.T)  # Allocate the size of the arrays

    def simulate(self, seed=1000):
        '''
        Simulation of the baseline model
        '''
        par = self.par
        sim = self.sim

        np.random.seed(seed)

        sim.epsilon_a[:] = np.random.normal(scale=1, size=par.T)  # Shocks to the exogenous variable
        sim.epsilon_m[:] = np.random.normal(scale=0.25, size=par.T)  # Monetary policy shocks

        # Set initial conditions
        sim.y[0] = 0.05  # Arbitrary output starting value
        sim.pi[0] = 0.02  # 2% initial inflation rate
        sim.i[0] = 0.025  # 2.5% initial interest rate
        sim.a[0] = 0.0  # No initial shock

        for t in range(1, par.T):
            sim.a[t] = par.rho * sim.a[t-1] + sim.epsilon_a[t]  # AR(1) process for exogenous monetary shock
            expected_y_next = sim.y[t-1]  # "Naive" expectations for y
            expected_pi_next = sim.pi[t-1]  # "Naive" expectations for pi

            # New-Keynesian model equations
            sim.y[t] = expected_y_next - (1 / par.sigma) * (sim.i[t-1] - expected_pi_next) # Dynamic IS-curve
            sim.pi[t] = par.beta * expected_pi_next + par.kappa * (sim.y[t] - sim.a[t]) # New-Keynesian Phillips Curve
            sim.i[t] = par.phi * sim.pi[t] + sim.epsilon_m[t]  # Taylor-Rule 

    def plot_results(self):
        '''
        Plot the results of the simulation
        '''
        sim = self.sim
        plt.figure(figsize=(12, 8))
        plt.plot(sim.y, label='Output ($y_t$)')
        plt.plot(sim.pi, label='Inflation ($\pi_t$)')
        plt.plot(sim.i, label='Interest Rate ($i_t$)')
        plt.legend()
        plt.title('Simulation of the New-Keynesian Model')
        plt.show()


    def simulate_extended(self, seed=1000):
        '''
        Simulation of the extended model
        '''
        par = self.par
        sim = self.sim

        np.random.seed(seed)
        sim.epsilon_a[:] = np.random.normal(scale=1, size=par.T)
        sim.epsilon_m[:] = np.random.normal(scale=0.25, size=par.T)
        
        #Same initial values
        sim.y[0] = 0.05
        sim.pi[0] = 0.02
        sim.i[0] = par.phi * sim.pi[0] + par.phi_y * sim.y[0] + sim.epsilon_m[0]
        sim.a[0] = 0.0

        for t in range(1, par.T):
            sim.a[t] = par.rho * sim.a[t-1] + sim.epsilon_a[t]
            expected_y_next = sim.y[t-1]
            expected_pi_next = sim.pi[t-1]

            sim.y[t] = expected_y_next - (1 / par.sigma) * (sim.i[t-1] - expected_pi_next)
            sim.pi[t] = par.beta * expected_pi_next + par.kappa * (sim.y[t] - sim.a[t])
            sim.i[t] = par.phi * sim.pi[t] + par.phi_y * sim.y[t] + sim.epsilon_m[t]

    def plot_results_extended(self):
        '''
        Plot the results of the simulation of the extended model
        '''
        sim = self.sim
        plt.figure(figsize=(12, 8))
        plt.plot(sim.y, label='Output ($y_t$)', color='blue')
        plt.plot(sim.pi, label='Inflation ($\pi_t$)', color='red')
        plt.plot(sim.i, label='Interest Rate ($i_t$)', color='green')
        plt.title('Extended Simulation of the New-Keynesian Model')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.show()