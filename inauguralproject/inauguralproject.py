from types import SimpleNamespace
from scipy.optimize import minimize_scalar
import numpy as np

class ExchangeEconomyClass:
    '''
    A class defining an Edgeworth economy
    '''
    def __init__(self):
        '''
        Initialization of the class
        '''

        # parameter dictionary
        par = self.par = SimpleNamespace()

        # preferences in parameters
        par.alpha = 1/3
        par.beta = 2/3

        # endowments in parameters
        par.w1A = 0.8
        par.w2A = 0.3
        par.w1B = 1 - par.w1A
        par.w2B = 1 - par.w2A
        par.w1bar = 1.0
        par.w2bar = 1.0

    def baseline_utility_A(self):
        '''
        Returns consumer A's baseline utility
        '''

        # define consumer A's baseline utility
        utility_A_baseline = self.utility_A(self.par.w1A, self.par.w2A)
        return utility_A_baseline

    def baseline_utility_B(self):
        '''
        Returns consumer B's baseline utility
        '''

        # define consumer B's baseline utility
        utility_B_baseline = self.utility_B(self.par.w1B, self.par.w2B)

        return utility_B_baseline
    
    def utility_A(self,x1A,x2A):
        '''
        Returns the utility of consumer A for a given allocation
        '''

        # utility of consumer A is defined
        return x1A ** self.par.alpha * x2A ** (1 - self.par.alpha)

    def utility_B(self,x1B,x2B):
        '''
        Returns the utility of consumer B for a given allocation
        '''

        # utility of consumer B is defined
        return x1B ** self.par.beta * x2B ** (1 - self.par.beta)
    
    def utility_A_p1(self,p1):
        '''
        Returns the utility of consumer A for a given p1
        '''

        # consumer B's demand at price p1
        x1B, x2B = self.demand_B(p1)

        # consumer A's consumption after B's consumption
        x1A, x2A = 1 - x1B, 1 - x2B

        # utility of A given this consumption
        utility_A = self.utility_A(x1A, x2A)

        # returns negative utility as it is used in a minimization call
        return -utility_A
    
    def objective_function(self, x):
        '''
        Objective function used to maximize utility of consumer A
        '''

        x1A, x2A = x

        # returns negative utility as it is used in a minimization call
        return -self.utility_A(x1A, x2A)
    
    def aggregate_utility_objective(self,x):
        '''
        Returns the aggregate utility of the two consumers
        '''

        x1A, x2A = x

        # returns the negative utility as it is used in a minimization call
        return -(self.utility_A(x1A, x2A) + self.utility_B(1 - x1A, 1 - x2A))
    
    def demand_A(self,p1):
        '''
        Returns consumer A's demand for good x1A and x2A for a given p1
        '''

        # define budget constraint of consumer A
        budget_a = p1 * self.par.w1A + self.par.w2A

        # define and return demand of consumer A for each good
        x1A = self.par.alpha * (budget_a / p1)
        x2A = (1 - self.par.alpha) * budget_a
        return x1A, x2A

    def demand_B(self,p1):
        '''
        Returns consumer B's demand for good x1B and x2B for a given p1
        '''

        # define budget constraint of consumer B
        budget_b = p1 * self.par.w1B + self.par.w2B

        # define and return demand of consumer B for each good
        x1B = self.par.beta * (budget_b / p1)
        x2B = (1 - self.par.beta) * budget_b

        return x1B, x2B

    def check_market_clearing(self,p1):
        '''
        Returns the errors in the market clearing condition for a given p1
        '''

        # define the demands
        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        # define the errors
        eps1 = x1A-self.par.w1A + x1B-self.par.w1B
        eps2 = x2A-self.par.w2A + x2B-self.par.w2B

        return eps1,eps2
    
    def find_market_clearing_price(self, w1A, w2A):
        '''
        Finds the market-clearing price given new endowments for consumer A
        '''

        # set new endowments for consumer A
        self.par.w1A = w1A
        self.par.w2A = w2A
        self.par.w1B = self.par.w1bar - w1A
        self.par.w2B = self.par.w2bar - w2A

        # find the price that minimizes the sum of absolute excess demands
        res = minimize_scalar(self.error_sum, bounds=(0, 10), method='bounded')
        return res.x
    
    def error_sum(self, p1):
        '''
        Returns the absolute sum of the errors for a given p1
        '''
    
        eps1, eps2 = self.check_market_clearing(p1)
        return abs(eps1) + abs(eps2)