"""
using Code by Ana this script will 
test PDEt for mass and redhsifts
"""
# we need Ana's main code
import sys
sys.append('../cbc_pdet/')
import o123_class_found_inj_general as u_pdet
import numpy as np
import json
import h5py as h5
import scipy
from scipy import integrate
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value, Planck15


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm, Normalize
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import glob
import deepdish as dd
from matplotlib import rcParams
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
rcParams["grid.alpha"] = 0.8

######## Some function for getting PDET ###############
#run_fit = 'o3'
g = u_pdet.Found_injections(dmid_fun = 'Dmid_mchirp_fdmid_fspin', emax_fun='emax_exp', alpha_vary = None, ini_files = None, thr_far = 1, thr_snr = 10)

# Define the cosmology (can be adjusted if needed)
H0 = 67.9 #km/sMpc
c = 299792.458 #3e5 #km/s
omega_m = 0.3065
cosmo = FlatLambdaCDM(H0=H0, Om0=omega_m)

pathplot = '/home/jam.sadiq/public_html/m1_dL_Analysis_with_SelectionEffects/selection_effects_plots/del_'
def calculate_pdet_m1m2dL(m1i, m2i, dLiMpc, classcall=g, mass_frame=None):
    """
    Calculate the probability of detection (PDET) given primary mass (m1),
    secondary mass (m2), and luminosity distance (dL).

    Parameters:
    m1i : float/array
        Primary mass (source or detector frame).
    m2i : float/array
        Secondary mass (source or detector frame).
    dLiMpc : float/array
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
        redshift = z_at_value(cosmo.luminosity_distance, dLiMpc * u.Mpc).value
        m1_det = m1i * (1 + redshift)
        m2_det = m2i * (1 + redshift)

    # Call the PDET calculation method from the provided classcall
    return classcall.run_pdet(dLiMpc, m1_det, m2_det, 'o3', chieff = 0,rescale_o3=True)


def pdfm2_powerlaw(m2, mmin, m1i, beta=1.26):
    """
    Probability density function for m2 based on a power law.
    Works with scalars or multidimensional arrays as input.
    """
    normfactor = (m1i ** (beta + 1) - mmin ** (beta + 1)) / ((beta + 1) * m1i ** beta)
    # Use np.where for conditional array calculations
    pdf = np.where((m2 >= mmin) & (m2 < m1i), (m2 / m1i) ** beta / normfactor, 0)
    return pdf

def pdet_of_m1_dL_powerlawm2(m1i, m_min, dLi, beta=1.26, classcall=g, mass_frame=None):
    """
    Calculates detection probability for a multidimensional input.
    """
    # Helper function to handle single m2 values (inner integrand)
    def pdet_integrand_powerlawm2(m2, m1i, dLi, m_min, beta, classcall, mass_frame= mass_frame):
        pdf_m2 = pdfm2_powerlaw(m2, m_min, m1i, beta)
        return calculate_pdet_m1m2dL(m1i, m2, dLi, classcall=classcall, mass_frame=mass_frame) * pdf_m2

    # Define a vectorized wrapper around the integrand
    vectorized_integrand = np.vectorize(
        lambda m1i, dLi: integrate.quad(
            pdet_integrand_powerlawm2,
            m_min,
            m1i,
            args=(m1i, dLi, m_min, beta, classcall, mass_frame),
            epsrel=1.5e-4
        )[0]
    )

    # Vectorize over inputs using numpy's broadcasting
    return vectorized_integrand(m1i, dLi)


def save_data_h5(fname, m1src_grid, DLsrc_grid, pdet_m1dLpowerlawm2, pdet_masses='correctframe'):
    """
    save expensive pdet when power-law on m2 is used
    """
    fh5 = h5.File(pdet_masses+fname, "w")
    fh5.create_dataset("m1_mesh", data=m1src_grid)
    fh5.create_dataset("dL_mesh", data=DLsrc_grid)
    fh5.create_dataset("pdet_mesh", data=pdet_m1dLpowerlawm2)
    #pdet_m1dLpowerlawm2 = fh5["pdet_mesh"][:]
    fh5.close()
    return 0

def pdet_of_m1_dL_marginalized_over_m2(m1i, m_min, dLi, classcall=None):
    """
    Computes the marginalized detection probability over m2 for arrays of m1i and dLi.
    """
    def pdet_integrand_m2_conditioned(m2, m1i, dLi, classcall):
        """
        Inner integrand function for a single m2 value.
        """
        return np.where(
            m2 > m1i,
            0,  # Return 0 where m2 > m1i
            calculate_pdet_m1m2dL(m1i, m2, dLi, classcall=classcall)
        )

    def normalization_integrand(m2, m1i):
        """
        Integrand for the normalization factor (PDF over m2).
        """
        return np.where(
            m2 > m1i,
            0,  # Return 0 where m2 > m1i
            1  # Constant density; modify if a specific PDF for m2 exists
        )

    # Define a vectorized wrapper around the integration
    vectorized_integrand = np.vectorize(
        lambda m1i, dLi: integrate.quad(
            pdet_integrand_m2_conditioned,
            m_min,
            m1i,
            args=(m1i, dLi, classcall),
            epsrel=1.5e-4
        )[0]
    )

    # Define a vectorized wrapper for normalization factor
    vectorized_normalization = np.vectorize(
        lambda m1i: integrate.quad(
            normalization_integrand,
            m_min,
            m1i,
            args=(m1i,),
            epsrel=1.5e-4
        )[0]
    )

    pdet_integral = vectorized_integrand(m1i, dLi)
    normalization_factor = vectorized_normalization(m1i)
    # Normalize the result
    return pdet_integral / normalization_factor

def pdet_of_m1_dL_marginalized_over_m2_Efficient_with_simpson(m1_det_grid, m2_det_grid, DL_det_grid, classcall=g, mass_frame='detector'):
    m2_vals = m2_det_grid[0, :, 0]
    PDET = calculate_pdet_m1m2dL(m1_det_grid, m2_det_grid, DL_det_grid, classcall=g, mass_frame='detector')
    normalization_factor = integrate.simpson(np.ones_like(PDET), m2_vals, axis=1) 
    pdet_marginalized_over_m2 = integrate.simpson(PDET, m2_vals, axis=1)/normalization_factor
    return pdet_marginalized_over_m2

########################PLOTS ##################################################
def m1_dLplot(m1src_grid, DLsrc_grid, pdet_m1_dL, plot_tag='power_law', save_tag='correct', pathplot='./'):
    """
    give 2D grid data for m1, dL and pdet 
    plot_tag for case power law of m2 or marginalization
    save_tag for pdet on correct det frame mass ior not
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(m1src_grid, DLsrc_grid, pdet_m1_dL, shading='auto', cmap='viridis')
# Add contour lines with specified levels
    contour_levels = [0.01,  0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0, 1.001]
    contours = ax.contour(m1src_grid, DLsrc_grid, pdet_m1_dL, levels=contour_levels,colors='white', linewidths=1.5)
    ax.clabel(contours, fmt='%0.2f', colors='white', fontsize=15)
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label(r'$p_\mathrm{det}$', fontsize=20)
    ax.set_xlabel('$m_{1, \mathrm{source}}\, [M_\odot]$')
    ax.set_ylabel('$d_L$  [Mpc]')
    #note that if we chnage this power in analysis we need to change it in  title  
    ax.set_title(r'$q^{1.26}$', fontsize=20)
    if plot_tag =='marginalize':
        ax.set_title(r'Marginalize over $m_2$', fontsize=20)
    # Display the plot
    plt.semilogx()
    plt.ylim(ymax=7000)
    plt.tight_layout()
    plt.savefig(pathplot+plot_tag+'_m2.png')
    plt.close()
    return 0

def for_fixed_dL_slicedata(m1_3Dgrid, m2_3Dgrid, dL_3Dgrid, pdet_3Dgrid, fix_dLval=500, save_tag='correct', pathplot='./'):
    """
    assuming that grids are computed m1, m2,dL 
    np.meshgrid(m1_vals, m2_vals, D_L_vals, indexing='ij')
    way and pdet is computed on those meshgrid
    """
    dL_1Dvals = dL_3Dgrid[0, 0, :]
    dL_index = np.searchsorted(dL_1Dvals, fix_dLval)
    print(dL_index)
    m1slice = m1_3Dgrid[:, :, dL_index]
    m2slice = m2_3Dgrid[:, :, dL_index]
    pdetslice = pdet_3Dgrid[:, :, dL_index]
    print(pdetslice.shape, np.min(pdetslice), np.max(pdetslice))
    #twoD plot
    fig, ax = plt.subplots(figsize=(8, 6))
    masked_pdet = np.ma.masked_where(m2slice > m1slice, pdetslice)
    mesh = ax.pcolormesh(m1slice, m2slice, masked_pdet, shading='auto', cmap='viridis')
    #mesh = ax.pcolormesh(m1slice, m2slice, pdetslice, shading='auto', cmap='viridis')
    # Add contour lines with specified levels
    contour_levels = [0.01,  0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0, 1.001]
    contours = ax.contour(m1slice, m2slice, masked_pdet, levels=contour_levels,colors='white', linewidths=1.5)
    #contours = ax.contour(m1slice, m2slice, pdetslice, levels=contour_levels,colors='white', linewidths=1.5)
    
    # Add labels to contour lines
    ax.clabel(contours, fmt='%0.2f', colors='white', fontsize=15)
    
    # Add colorbar for the colormesh
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label(r'$p_\mathrm{det}$')
    
    # Set axis labels
    ax.set_xlabel('$m_{1, \mathrm{source}}$')
    ax.set_ylabel('$m_{2, \mathrm{source}}$')
    # Add a title
    ax.set_title(r'$d_L={0}$ [Mpc]'.format(fix_dLval), fontsize=18)    
    # Display the plot
    plt.loglog()
    plt.tight_layout()
    
    plt.savefig(pathplot+save_tag+'_slicedL{0}_pdetm1m2.png'.format(fix_dLval))
    plt.close()
    
    return m1slice, m2slice, pdetslice

################ testing pdet for m1-dL intgrating on m2 #####################
m1_source = np.logspace(np.log10(3), np.log10(120), 100)
m2_source = np.logspace(np.log10(3), np.log10(120), 100)
dL_vals = np.linspace(10, 10000, 100)  # From 10 Mpc to 10,000 Mpc

#get detector frame-mass
redshift_vals = z_at_value(cosmo.luminosity_distance, dL_vals* u.Mpc).value
m1_det = m1_source * (1. + redshift_vals)
m2_det = m2_source * (1. + redshift_vals)

m1src_3Dgrid, m2src_3Dgrid, DLsrc_3Dgrid = np.meshgrid(m1_source, m2_source, dL_vals, indexing='ij')
m1_det_3Dgrid, m2_det_3Dgrid, DL_det_3Dgrid = np.meshgrid(m1_det, m2_det, dL_vals, indexing='ij')
#################### Case I m1-dL power law for m2 #################################################
#we need to use 2D grid for power-law case
m1src_2Dgrid, dLsrc_2Dgrid = m1src_3Dgrid[:, 0, :], DLsrc_3Dgrid[:, 0, :] 
#m1src_2Dgrid, dLsrc_2Dgrid = np.meshgrid(m1_source, dL_vals, indexing='ij')

m1_det_2Dgrid, dL_2Dgrid = m1_det_3Dgrid[:, 0, :], DL_det_3Dgrid[:, 0, :]
#m1_det_2Dgrid, dL_2Dgrid = np.meshgrid(m1_det, dL_vals, indexing='ij')

##### correct way to get pdet is either use mass_frame='detector' or use source frame removing mass_frame
mmin = 2.999 #min of integration for m2
beta_index = 1.26  #index of power law for q 
#pdet_m1dLpowerlawm2 = pdet_of_m1_dL_powerlawm2(m1_det_2Dgrid, mmin, dLsrc_2Dgrid, beta=1.26, classcall=g, mass_frame='detector')
#to check the difference if we use masses in wrong frame
#Incorrect_pdet_m1dLpowerlawm2 = pdet_of_m1_dL_powerlawm2(m1src_2Dgrid, mmin,dDLsrc_2Dgrid, beta=1.26, classcall=g, mass_frame='detector')
###########its very expensive so save the data so plot later
#save_data_h5('data_m1src_dL_Pdet_2D.h5', m1src_2Dgrid,, pdet_m1dLpowerlawm2, pdet_masses='correctframe')
##########plot 
#m1_dLplot(m1src_2Dgrid, dLsrc_2Dgrid, pdet_m1dLpowerlawm2, plot_tag='power_law', pathplot=pathplot)
#m1_dLplot(m1src_2Dgrid, dLsrc_2Dgrid, Incorrect_pdet_m1dLpowerlawm2, plot_tag='power_law', save_tag='Incorrect', pathplot=pathplot)

#################### Case II m1-dLMarginalize over m2 #################################################
##### correct way to get pdet is either use detector frame mass #if use source frame mass remove mass_frame arg
pdet_m1m2dL_3D = calculate_pdet_m1m2dL(m1_det_3Dgrid, m2_det_3Dgrid, DL_det_3Dgrid, classcall=g, mass_frame='detector')
pdet_3Dflat = pdet_m1m2dL_3D.flatten()
#incorrect_pdet_m1m2dL_3D= calculate_pdet_m1m2dL(m1src_3Dgrid, m2src_3Dgrid, DLsrc_3Dgrid, classcall=g, mass_frame='detector')
#incorrect_pdet_3Dflat = incorrect_pdet_m1m2dL_3D.flatten()

pdet_marginalized_over_m2 = pdet_of_m1_dL_marginalized_over_m2_Efficient_with_simpson(m1_det_3Dgrid, m2_det_3Dgrid, DL_det_3Dgrid, classcall=g, mass_frame='detector')
m1_dLplot(m1src_2Dgrid, dLsrc_2Dgrid, pdet_marginalized_over_m2, plot_tag='marginalize', pathplot=pathplot)
#################### Case III m1-m2 pdet at fixed dL or z val ########################################
for dLslice in [500, 900, 1500]:
    for_fixed_dL_slicedata(m1src_3Dgrid, m2src_3Dgrid, DLsrc_3Dgrid, pdet_m1m2dL_3D, fix_dLval=dLslice, save_tag='correct', pathplot=pathplot)
    #for_fixed_dL_slicedata(m1src_3Dgrid, m2src_3Dgrid, DLsrc_3Dgrid, incorrect_pdet_m1m2dL_3D, fix_dLval=dLslice, save_tag='Incorrect', plotplot=pathplot)
# Create the 3D scatter plot
# Flatten the data for 3D scatter plot
m1src_flat, m2src_flat, DLsrc_flat = m1src_3Dgrid.flatten(), m2src_3Dgrid.flatten(), DLsrc_3Dgrid.flatten()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(m1src_flat, m2src_flat, DLsrc_flat, c=pdet_3Dflat, cmap='viridis', s=5, norm=LogNorm(vmin=1e-5))
ax.set_xlabel('$m_{1, \mathrm{source}}$')
ax.set_ylabel('$m_{2, \mathrm{source}}$')
ax.set_zlabel('$D_L$ [Mpc]')
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('$p_\mathrm{det}$')
plt.tight_layout()
plt.savefig(pathplot+'ThreeDpdet_m1m2dL.png')
plt.close()


################## Final Posterior Samples PDET m1-m2-dL scatter #############################
