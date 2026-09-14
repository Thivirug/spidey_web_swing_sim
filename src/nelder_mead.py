import numpy as np
from collections.abc import Callable

# https://alexdowad.github.io/visualizing-nelder-mead/  - ref
class NelderMead:
    def __init__(self, f: Callable, initial_point: np.ndarray, tol=1e-6, alpha=1.0, gamma=2.0, beta=0.5, sigma=0.5):
        self.f = f # policy cost func

        if np.allclose(initial_point, np.zeros_like(initial_point)):
            raise ValueError("Initial point cannot be all zeros")
        
        self.initial_point = initial_point
        self.ndims = len(self.initial_point)
        self.tol = tol

        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if gamma <= 1:
            raise ValueError("gamma must be greater than 1")

        if not (0 < beta < 1):
            raise ValueError("beta must be between 0 and 1")

        if not (0 < sigma < 1):
            raise ValueError("sigma must be between 0 and 1")

        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.sigma = sigma

        # create simplex
        self.simplex = self._simplex()

    def _simplex(self, vary_frac=0.05):
        """
            Perturb the initial point by vary_frac make ndims more points
        """
        simplex = [self.initial_point] # initialise

        for n in range(self.ndims):
            new = self.initial_point.copy()
            new[n] += new[n]*vary_frac # vary the nth entry of initial point by vary_frac

            simplex.append(new)

        return np.array(simplex)

    def order_vertices(self):
        self.f_values = np.array([
            self.f(vertex) for vertex in self.simplex
        ])

        # get the indices for sorting in asc order
        order = np.argsort(self.f_values)

        # sort f vals and simplex
        self.f_values = self.f_values[order]
        self.simplex = self.simplex[order]

        # get the centroid after ordering
        self._centroid()

    def _centroid(self):
        self.centroid = np.mean(self.simplex[:-1,:], axis=0) # the mean across rows except N+1 th row (the worst after ordering)
    
    def _reflect(self):
        """
            Move the worst point thru the middle of the other 2
            Return the reflected point and its func value
        """
        v_R = self.centroid + self.alpha * (self.centroid - self.simplex[-1])
        f_R = self.f(v_R)

        return v_R, f_R

    def _expand(self, v_R):
        """
            Similar to reflect but move the reflected point further
            - v_R is the reflected point
        """
        v_E = self.centroid + self.gamma * (v_R - self.centroid)
        f_E = self.f(v_E)

        return v_E, f_E

    def _contract(self, contract_type: str):
        """
            Move the worst point towards the middle of the other two. 
            Depending on the situation, contract can 
            - either stop short of the opposite side (contract_type=ci), - contract inside
            - or move slightly past it (contract_type=co). - contract outside
        """
        if contract_type not in ("ci", "co"):
            raise ValueError("Invalid contract_type arg")

        w = self.beta if contract_type=='co' else -self.beta

        v_C = self.centroid + w * (self.centroid - self.simplex[-1])
        f_C = self.f(v_C)

        return v_C, f_C

    def _shrink(self):
        """
            Shrink the whole triangle towards the best point, maintaining its angles.
        """
        best_point = self.simplex[0].copy()
        new_simplex = [i + self.sigma * (best_point - i) for i in self.simplex]

        self.simplex = np.array(new_simplex)

    def __call__(self):
        # main algorithm 
        pass
        

if __name__ == '__main__':
    # generic test
    def f(t):
        x,y = t
        return x**2 + y**2

    a = np.array([1.0, 1.5])
    n = NelderMead(f, a)
    print(n.simplex)
    n.order_vertices()
    print(n.simplex)
    print(n.f_values)
    print(n.centroid)

    print()
    r, fr = n._reflect()
    print("centroid:", n.centroid)
    print("reflected:", r)
    print("f(reflected):", fr)

    print()
    e, fe = n._expand(r)
    print("centroid:", n.centroid)
    print("expanded:", e)
    print("f(expanded):", fe)

    print()
    c, fc = n._contract("ci")
    print("centroid:", n.centroid)
    print("contracted inside:", c)
    print("f(contracted inside):", fc)

    c, fc = n._contract("co")
    print("centroid:", n.centroid)
    print("contracted outside:", c)
    print("f(contracted outside):", fc)

    print()
    n._shrink()
    print("centroid:", n.centroid)
    print("shrinked simplex:\n", n.simplex)

    

