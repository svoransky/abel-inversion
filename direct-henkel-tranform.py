from scipy.special import jn
import numpy as np

#############################################################
#
# @stggh - test of
# discrete Hankel transform by
# @steven-murray 
# 29 Aug 2017
#############################################################

def construct_dht_matrix(N, nu  = 0, b = 1):
    m = np.arange(0,N).astype('float')
    n = np.arange(0,N).astype('float')
    return b * jn(nu, np.outer(b*m, n/N))

def dht(X, d = 1, nu = 0, axis=-1, b = 1):
    N = X.shape[axis]
    
    prefac = d**2
    m = np.arange(0,N)
    freq = m/(float(d)*N)
    
    F = construct_dht_matrix(N,nu,b)*m
    return prefac * np.tensordot(F,X, axes=([1],[axis])) #, freq


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    r = np.linspace(0, 1001, 1001)
    n = r.shape[-1]

    # Gaussian function
    a = 0.01
    f = np.exp(-(r*a)**2/2)

    # discrete Hankle transform
    g = dht(f)*np.sqrt(a)/n

    # plot results
    fig, ax = plt.subplots()

    ax.plot(r, f, label=r'$e^{-\frac{1}{2}a^2 r^2}$')
    ax.plot(r/np.sqrt(a), g, label=r'$\frac{1}{a^2}e^{-\frac{k^2}{2a^2}}$')
    ax.set_xlabel(r'$r$ or $k$')
    ax.set_ylabel(r'$f(r)$ or $F_0(k)$')
    ax.axis(xmin=0, xmax=500)
    ax.legend(fontsize='larger')

    plt.savefig("Hankel.png", dpi=75)
    plt.show()
