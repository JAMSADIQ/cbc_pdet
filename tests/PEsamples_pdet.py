"""
using Code by Ana this script will 
test PDEt for mass and redhsifts
"""
# we need Ana's main code
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

pathplot = '/home/jam.sadiq/public_html/m1_dL_Analysis_with_SelectionEffects/selection_effects_plots/'

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

injection_file = "endo3_bbhpop-LIGO-T2100113-v12.hdf5"
with h5.File(injection_file, 'r') as f:
    T_obs = f.attrs['analysis_time_s']/(365.25*24*3600) # years
    N_draw = f.attrs['total_generated']

    m1 = f['injections/mass1_source'][:]
    m2 = f['injections/mass2_source'][:]
    s1x = f['injections/spin1x'][:]
    s1y = f['injections/spin1y'][:]
    s1z = f['injections/spin1z'][:]
    s2x = f['injections/spin2x'][:]
    s2y = f['injections/spin2y'][:]
    s2z = f['injections/spin2z'][:]
    z = f['injections/redshift'][:]
    dLp = f["injections/distance"][:]
    m1_det = m1#*(1.0 +  z)
    p_draw = f['injections/sampling_pdf'][:]
    pastro_pycbc_bbh = f['injections/pastro_pycbc_bbh'][:]

# Calculate min and max for dLp
min_dLp, max_dLp = min(dLp), max(dLp)
fz = h5.File('Final_noncosmo_GWTC3_redshift_datafile.h5', 'r')
dz = fz['randdata']
f1 = h5.File('Final_noncosmo_GWTC3_m1srcdatafile.h5', 'r')#m1
d1 = f1['randdata']
f2 = h5.File('Final_noncosmo_GWTC3_m2srcdatafile.h5', 'r')#m2
d2 = f2['randdata']
f3 = h5.File('Final_noncosmo_GWTC3_dL_datafile.h5', 'r')#dL
d3 = f3['randdata']
print(d1.keys())
sampleslists1 = []
eventlist = []
sampleslists2 = []
sampleslists3 = []
pdetlists = []
for k in d1.keys():
    eventlist.append(k)
    #if (k  == 'GW190719_215514_mixed-nocosmo' or k == 'GW190805_211137_mixed-nocosmo'):
    m1_source = d1[k][...]
    m2_source = d2[k][...]
    m1_det = d1[k][...]*(1.0 + dz[k][...])
    m2_det = d2[k][...]*(1.0 + dz[k][...])
    d_Lvalues = d3[k][...]
    #find unreasonable dL vals in PE samples

    dL_indices = [i for i, dL in enumerate(d_Lvalues) if (dL < min_dLp  or dL > max_dLp)]
    m1_source = [m for i, m in enumerate(m1_source) if i not in  dL_indices]
    m2_source = [m for i, m in enumerate(m2_source) if i not in  dL_indices]
    m1_det = [m for i, m in enumerate(m1_det) if i not in  dL_indices]
    m2_det = [m for i, m in enumerate(m2_det) if i not in  dL_indices] 
    d_Lvalues = [dL for i, dL in enumerate(d_Lvalues) if i not in dL_indices]
    #pdet_values = u_pdet.get_pdet_m1m2dL(np.array(m1_det), np.array(m2_det), np.array(d_Lvalues), classcall=g)
    pdet_values = u_pdet.get_pdet_m1m2dL(np.array(m1_det), np.array(m2_det), np.array(d_Lvalues), classcall=g)
        #fpdet.create_dataset(k, data=pdet_values)
        #pdet_values = [pdet for i, pdet in enumerate(pdet_values) if i not in dL_indices]
        #still some bad indices
    pdetminIndex = np.where(np.array(pdet_values) < 5e-4)[0]
    m1_source = np.delete(m1_source, pdetminIndex).tolist()
    m2_source = np.delete(m2_source, pdetminIndex).tolist()
    m1_det = np.delete(m1_det, pdetminIndex).tolist()
    m2_det = np.delete(m2_det, pdetminIndex).tolist()
    d_Lvalues = np.delete(d_Lvalues, pdetminIndex).tolist()
    pdet_values = np.delete(pdet_values, pdetminIndex).tolist()

   # else:
   #     m1_values = d1[k][...]#*(1.0 + dz1[k][...])
   #     m2_values = d2[k][...]#*(1.0 + dz1[k][...])
   #     d_Lvalues = d3[k][...]
        # if we want to compute pdet use line after below line 
  #      pdet_values =  np.zeros(len(d_Lvalues))
  #      for i in range(len(d_Lvalues)):
        #    print(i)
  #         pdet_values[i] = u_pdet.get_pdet_m1m2dL(d_Lvalues[i], m1_values[i], m2_values[i], classcall=g)
    pdetlists.append(pdet_values)
    print(k, len(pdet_values))
    sampleslists1.append(m1_source)
    sampleslists2.append(m2_source)
    sampleslists3.append(d_Lvalues)

f1.close()
f2.close()
f3.close()
fz.close()
flat_samples1 = np.concatenate(sampleslists1).flatten()
flat_samples2 = np.concatenate(sampleslists2).flatten()
flat_samples3 = np.concatenate(sampleslists3).flatten()
flat_pdetlist = np.concatenate(pdetlists).flatten()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(flat_samples1, flat_samples2, flat_samples3, c=flat_pdetlist, cmap='viridis', s=10, norm=LogNorm())
plt.colorbar(sc, label=r'$p_\mathrm{det}(m_1, m_2, d_L)$')
ax.set_xlabel(r'$m_{1, source} [M_\odot]$', fontsize=20)
ax.set_ylabel(r'$m_{2, source} [M_\odot]$', fontsize=20)
#ax.set_xlim(min(flat_samples1), max(flat_samples1))
#ax.set_ylim(min(flat_samples2), max(flat_samples2))
#ax.set_zlim(min(flat_samples3), max(flat_samples3))
ax.set_zlabel(r'$d_L [Mpc]$', fontsize=20)
plt.tight_layout()
plt.savefig(pathplot+"clean_PEsamplespdet3Dscatter.png")
plt.close()


plt.figure(figsize=(8,6))
plt.scatter(flat_samples1, flat_samples3, c=flat_pdetlist, cmap='viridis', norm=LogNorm())
cbar = plt.colorbar(label=r'$p_\mathrm{det}$')
cbar.set_label(r'$p_\mathrm{det}$', fontsize=20)
#plt.xlim(min(flat_samples1), max(flat_samples1))
#plt.ylim(min(flat_samples3), max(flat_samples3))
#plt.xlabel(r'$m_{1, detector} [M_\odot]$', fontsize=20)
plt.xlabel(r'$m_{1, source} [M_\odot]$', fontsize=20)
plt.ylabel(r'$d_L [Mpc]$', fontsize=20)
plt.loglog()
plt.title(r'$p_\mathrm{det}$', fontsize=20)
plt.tight_layout()
plt.savefig(pathplot+"clean_PEsampls2Dm1dLpdetscatter.png")
plt.close()
