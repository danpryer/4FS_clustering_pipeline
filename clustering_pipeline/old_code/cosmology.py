from project_libs import *
import allvars as av


# function to get H(z)
def hubble_param(z, H0, omega_m):
    return H0 * pow(omega_m*(1+z)**3 + (1-omega_m),0.5)



# function to get D(z)
def growth_factor(z, H0, omega_m):

    def growth_integrand(z, H0, omega_m):
        return (1 + z) * pow(hubble_param(z, H0, omega_m)**3, -1)

    norm, norm_err = integrate.quad(growth_integrand, 0, np.inf, args=(H0, omega_m))
    res, err = integrate.quad(growth_integrand, z, np.inf, args=(H0, omega_m))
#     res = integrate.quad(growth_integrand, z, np.inf, args=(H0, omega_m))[0]
    return hubble_param(z, H0, omega_m)*pow(H0, -1)*pow(norm, -1)*res



# model of evolving bias
def bias_model(z):
    return av.bias_1 + (av.bias_0 - av.bias_1) * pow(av.spl_growth_fact(z), -av.bias_alpha)



# linear RSD factor
def RSD_linear(z):
    beta = av.spl_growth_rate(z) / (av.spl_bias(z)*1.)
    return 1 + ((2./3)*beta) + ((1./5.)*beta*beta)

# nonlinear factor, to be integrated for -1 < mu < 1
def RSD_nonlin_integrand(mu, z, k):
    beta = av.spl_growth_rate(z) / (av.spl_bias(z)*1.)
    return pow(1 + beta*mu*mu, 2) * pow(1 + pow(k*mu*av.bias_sigma,2),-1)

# integrate the above
def RSD_nonlin(z, k):
    return integrate.quad(RSD_nonlin_integrand, -1., 1., args=(z, k), limit = 10000, epsrel=1e-3)[0]
