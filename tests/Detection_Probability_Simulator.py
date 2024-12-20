"""
using Code by Lorenzo/Dent this script will 
test PDEt for mass and redhsifts
"""
##################import libraries ############################
import o123_class_found_inj_general as u_pdet
mport numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import h5py as h5
import scipy
from matplotlib import rcParams
from matplotlib.colors import LogNorm, Normalize
import matplotlib.colors as colors
from matplotlib import cm
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import matplotlib.patches
from matplotlib.patches import Rectangle
import glob
import deepdish as dd
rcParams["text.usetex"] = True
rcParams["font.serif"] = "Computer Modern"
rcParams["font.family"] = "Serif"
rcParams["xtick.labelsize"]=18
rcParams["ytick.labelsize"]=18
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=18
rcParams["axes.labelsize"]=18
rcParams["axes.grid"] = True
rcParams["grid.color"] = 'grey'
rcParams["grid.linewidth"] = 1.
rcParams["grid.linestyle"] = ':'
rcParams["grid.alpha"] = 0.6

############# Call FoundInjection ####################3333
g = u_pdet.Found_injections(dmid_fun = 'Dmid_mchirp_fdmid_fspin', emax_fun='emax_exp', alpha_vary = None, ini_files = None, thr_far = 1, thr_snr = 10)
##################### Need Cosmology for redshift #########################3
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value, Planck15
# Define the cosmology (can be adjusted if needed)
H0 = 67.9 #km/sMpc
c = 299792.458 #3e5 #km/s
omega_m = 0.3065
cosmo = FlatLambdaCDM(H0=H0, Om0=omega_m)
def calculate_pdet_m1m2dL(m1i, m2i, dLiMpc, classcall=g, mass_frame=None):
    """
    Calculate the probability of detection (PDET) given primary mass (m1),
    secondary mass (m2), and luminosity distance (dL).

    Parameters:
    m1i : float
        Primary mass (source or detector frame).
    m2i : float
        Secondary mass (source or detector frame).
    dLiMpc : float
        Luminosity distance in megaparsecs.
    classcall : object, optional
        Class or function to calculate PDET (default is `g`).
    mass_frame : str, optional
        Frame of the masses: 'detector' or 'source'. If not specified,
        the function assumes source frame and converts to detector frame.
    Returns:
    float
        Probability of detection (PDET).
    """
    if mass_frame == 'detector':
        print("Masses are provided in the detector frame.")
        m1_det, m2_det = m1i, m2i
    else:
        print("Masses are assumed to be in the source frame. Converting to detector frame.")
        redshift = z_at_value(cosmo.luminosity_distance, dLiMpc * u.Mpc)
        m1_det = m1i * (1 + redshift)
        m2_det = m2i * (1 + redshift)

    # Call the PDET calculation method from the provided classcall
    return classcall.run_pdet(dLiMpc, m1_det, m2_det, 'o3', chieff = 0,rescale_o3=True)
  
def pdfm2_powerlaw(m2, mmin, m1i, beta=1.26):
    """
    Normalized PDF for power-law distribution (eq.B4 in arXiv:2010.14533)
    """
    normfactor = (m1i ** (beta+1) - mmin ** (beta+1)) / ((beta+1) * m1i ** beta)
    if mmin <= m2 < m1i:
        return (m2/m1i) ** beta / normfactor
    else:
        return 0

def pdet_of_m1_dL_powerlawm2(m1i, m_min, dLi, beta=1.26, classcall=g):
    """
    Evaluate VT for the given array of m1 values marginalizing over
    m2 using a power law distribution with parameters m_min, beta.

    Parameters:
    m1i : float
        Primary mass.
    m_min : float
        Minimum mass for the power-law distribution.
    dLi : float
        Luminosity distance in megaparsecs.
    beta : float, optional
        Slope of the power-law distribution (default is 1.26).
    classcall : object, optional
        Class or function to calculate PDET (default is `g`).

    Returns:
    float
        Marginalized PDET over m2.
    """
    def pdet_integrand_powerlawm2(m2):
        return calculate_pdet_m1m2dL(m1i, m2, dLi, classcall=classcall) * pdfm2_powerlaw(m2, m_min, m1i, beta=beta)

    return integrate.quad(pdet_integrand_powerlawm2, m_min, m1i, epsrel=1.5e-4)[0]

def pdet_of_m1_dL_marginalized(m1i, m_min, dLi, classcall=g):
    """
    Evaluate VT for the given array of m1 values marginalizing over
    m2 with the condition that m2 <= m1.

    Parameters:
    m1i : float
        Primary mass.
    m_min : float
        Minimum mass for integration.
    dLi : float
        Luminosity distance in megaparsecs.
    classcall : object, optional
        Class or function to calculate PDET (default is `g`).

    Returns:
    float
        Marginalized PDET over m2 with m2 <= m1.
    """
    def pdet_integrand_m2_conditioned(m2):
        if m2 > m1i:
            return 0  # Mask values where m2 > m1
        return calculate_pdet_m1m2dL(m1i, m2, dLi, classcall=classcall)

    return integrate.quad(pdet_integrand_m2_conditioned, m_min, m1i, epsrel=1.5e-4)[0]

m1_vals = np.linspace(3, 150, 50)
m2_vals = np.linspace(3, 150, 50)
D_L_vals = np.logspace(1, 4, 50)  # From 10 Mpc to 10,000 Mpc
m1src_grid, m2src_grid, DLsrc_grid = np.meshgrid(m1_vals, m2_vals, D_L_vals, indexing='ij')
m1src_flat, m2src_flat, DLsrc_flat = m1src_grid.flatten(), m2src_grid.flatten(), DLsrc_grid.flatten()
#get detectoor frame mass
redshift = z_at_value(cosmo.luminosity_distance, D_L_vals* u.Mpc)
m1_det = m1_vals * (1 + redshift)
m2_det = m2_vals * (1 + redshift)
m1_grid, m2_grid, DL_grid = np.meshgrid(m1_det, m2_det, D_L_vals, indexing='ij')

PDET = calculate_pdet_m1m2dL(m1_grid, m2_grid, DL_grid, classcall=g, mass_frame='detector')
# Flatten the data for 3D scatter plot
m1_flat, m2_flat, DL_flat, PDET_flat = m1_grid.flatten(), m2_grid.flatten(), DL_grid.flatten(), PDET.flatten()

# Create the 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(m1src_flat, m2src_flat, DL_flat, c=PDET_flat, cmap='viridis', s=5, norm=colors.LogNorm(vmin=1e-5))
ax.set_xlabel('$m_{1, \mathrm{source}}$')
ax.set_ylabel('$m_{2, \mathrm{source}}$')
ax.set_zlabel('$D_L$ [Mpc]')
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('$p_\mathrm{det}$')
plt.tight_layout()
plt.savefig('ThreeD_ScatterPdet.png')
plt.close()
