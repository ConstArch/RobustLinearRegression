import numpy as np
import statsmodels.api as sm


class LogisticMLE(sm.robust.norms.RobustNorm):
    
    def __init__(self):
        super().__init__()
    
    def rho(self, z):
        return np.log(np.cosh(0.5 * z))
    
    def psi(self, z):
        return 0.5 * np.tanh(0.5 * z)
    
    def weights(self, z):
        return 0.5 * np.tanh(0.5 * z) / z
    
    def psi_deriv(self, z):
        ch = np.cosh(z)
        return 1 / (ch * ch)


class OLSModelFactory:
    
    def __init__(self):
        pass
    
    def __call__(self, X, Y):
        return sm.OLS(Y, sm.add_constant(X)).fit()


class LADModelFactory:
    
    def __init__(self):
        pass
    
    def __call__(self, X, Y):
        return sm.QuantReg(Y, sm.add_constant(X)).fit()


class LogModelFactory:
    
    def __init__(self):
        pass
    
    def __call__(self, X, Y):
        return sm.RLM(Y, sm.add_constant(X), M=LogisticMLE()).fit()


class HuberModelFactory:
    
    def __init__(self):
        pass
    
    def __call__(self, X, Y):
        return sm.RLM(Y, sm.add_constant(X), M=sm.robust.norms.HuberT()).fit()


class TukeyModelFactory:
    
    def __init__(self):
        pass
    
    def __call__(self, X, Y):
        return sm.RLM(Y, sm.add_constant(X), M=sm.robust.norms.TukeyBiweight()).fit()
