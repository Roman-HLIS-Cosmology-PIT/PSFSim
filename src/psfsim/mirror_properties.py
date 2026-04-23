#mirror_properties class
#Created 16-Apr-2026
#Developer: Anthony Harbo Torres
#  with technical guidance by Christopher Hirata
#   and Charuhas Shiveshwarkar
#version:0.1a


# imports
import argparse
import numpy as np

# optical functions
def n_medium(epsilon, mu):
    """ Compute n of this medium
        Params:
            epsilon, mu: complex
        Output:
            n_medium: complex
    """
    return np.emath.sqrt(epsilon*mu)

def cosine_theta_medium(theta_inc, n_inc, n_medium):
    """ Compute cos(theta_medium) from conserved transverse wavevector,
            according to n_inc * sin(theta_inc) = n_med * sin(theta_med)
        Params:
            theta_inc: float
            n_inc, n_medium : complex
        Output:
            cosine(theta_medium): complex
    """
    return np.emath.sqrt( 1 - ( (n_inc/n_medium) * np.sin(theta_inc) )**2 )


def tilted_optical_admittance(cos_theta_medium, epsilon, mu, polarisation_mode):
    """ Tilted optical admittance, Q(z) = [U(z),V(z)]
            For TE: Q = (1/z) * cos(theta_medium)
            For TM: Q = z * cos(theta_medium)
        Params:
            cos_theta_medium: float
                Cosine of the angle theta in that layer
            epsilon, mu: complex
                Elec. permittivity and mag. permeability of the layer
            polarisation_mode: str
                Can pass either {TM or P} or {TE or S} as choices
        Output:
            Q(z): complex
                Form depends on polarisation_mode
    """
    z = np.emath.sqrt(mu/epsilon)

    polarisation_mode = polarisation_mode.lower()
    if polarisation_mode in ("te","s"):
        return cos_theta_medium / z
    elif polarisation_mode in ("tm", "p"):
        return cos_theta_medium * z

def thin_film_characteristic_matrix(d, k_0, n_inc, theta_inc, epsilon, mu, polarisation_mode):
    """ Characteristic matrix for a single thin film layer
        Params:
            d: float
                Thickness d of the thin film in nm
            k_0: float
                Vacuum wavevector in inverse cm
            n_inc: complex
                Refrc. index of the incident medium, used to
                define the conserved transverse wavevector
            theta_inc: float
                Angle of incidence rel. to normal in radians
            epsilon, mu: complex
                Relative elec. permittivity and mag. permeability
                of this layer
            polarisation_mode: str
                Which polarisation mode is being solved for, TE or TM

        Output:
            matrix: np.array
                Characteristic matrix for this layer, (2x2)
    """
    #change incoming d in nm to cm
    d = d*(1e-7)

    #compute n of this medium
    index_of_medium = n_medium(epsilon=epsilon, mu=mu)

    #compute cos(theta_medium) from conserved transverse wavevector
    cos_theta_med = cosine_theta_medium(theta_inc=theta_inc,
                                        n_inc = n_inc,
                                        n_medium = index_of_medium
                                        )

    #calculate the corresponding Q in this layer, picking the right branch
    Q_layer = tilted_optical_admittance(
        cos_theta_medium=cos_theta_med,
        epsilon=epsilon,
        mu=mu,
        polarisation_mode=polarisation_mode)

    #precalculate the argument of the sines and cosines
    argument = k_0 * d * index_of_medium * cos_theta_med

    #compute the matrix
    matrix = np.array( [ [np.cos(argument), np.sin(argument) * (1/Q_layer) * (-1j)], [np.sin(argument) * (Q_layer) * (-1j), np.cos(argument)] ] )

    return matrix

def effective_admittance(matrix, Q_0):
    """Effective admittance seen at the entrace of the layer stack, given substrate
    admittance Q_0 and the characteristic matrix for the thin film above it.
        Params:
            matrix: np.array
                Characteristic matrix of thin film above substrate
            Q_0: complex
                The optical admittance of the substrate
        Output:
            Q: complex
                Effective admittance for the layer stack
    """
    A, B, C, D = matrix[0,0], matrix[0,1], matrix[1,0], matrix[1,1]

    return (C + D*Q_0) / (A + B*Q_0)

def reflection_coefficient(Q_vacuum, Q_medium):
    """ Reflection coefficient for an incident beam going from vacuum into a medium
        Params:
            Q_vacuum: complex
                Admittance in vacuum
            Q_medium: complex
                Optical admittance of the medium
    """
    return (Q_vacuum - Q_medium) / (Q_vacuum + Q_medium)

#end of optical functions--------------------------------



#------------------------------------------------------
# wavelength specific epsilon functions

def malitson_sellmeier_sio2_eps(wavelength:float):
    """
    RefractiveIndex.INFO 'formula 1' Sellmeier form.

    Parameters
    ----------
    wavelength : float
        Wavelength in microns.

    Returns
    -------
    epsilon : float
        SiO2 electric permittivity.
    """
    n_squared = (
        1 + ( ((0.6961663)*wavelength**2)/ (wavelength**2 - (0.0684043)**2) )
        + ( ((0.4079426)*wavelength**2)/ (wavelength**2 - (0.1162414)**2) )
        + ( ((0.8974794)*wavelength**2)/ (wavelength**2 - (9.896161)**2) )
        )

    return n_squared

def yang_ag_eps(wavelength:float,):
    """ Computes the Yang et al 2015 dielectric function,
    in natural units???
        Params:
            wavelength: float
                wavelength in microns
    """
    h_bar = 6.582119569e-16 # hbar in units of eV*s
    hc = 1239.841984 # hc in units of eV*nm
    tau = 17 # femtoseconds
    epsilon_infinity = 5  #unitless
    h_bar_omega_plasma = 8.9 # units of eV

    wvlngh_energy = (hc/(wavelength*1e3))
    denom = wvlngh_energy**2 + 1j * (wvlngh_energy * (h_bar/(tau*1e-15)))

    return epsilon_infinity - (h_bar_omega_plasma**2)/denom
#end of epsilon functions------------------------------------------


def reflect_RB_off_mirror(thetas:np.array,wavelength:float, thickness:float = 0.0):
    """
    Mirror class that calculates the S and P reflectances for a set
    of angles theta, for a given wavelength
        Params:
            thetas: np.array
                Angles in radians
            wavelength: float
                wavelength in cm
        Output:
            te_reflectance, tm_reflectance
    
    """
    #eventually set these as params, todo

    ag_mag_permeability = 1 + 0j
    sio2_mag_permeability = 1 + 0j
    vacuum_mag_permeability = 1 + 0j
    vacuum_elec_permittivity = 1 + 0j


    # Refractive index of incident medium (vacuum)
    n_vacuum = 1 + 0j

    # Refractive index of substrate (silver)
    eps_ag = yang_ag_eps(wavelength=wavelength*1e4) #wavelength is in cm, sent as microns

    # Refractive index of thin film coating (single layer, SiO2)
    eps_sio2 = malitson_sellmeier_sio2_eps(wavelength=wavelength*1e4) #wavelength is in cm, sent as microns

    te_coefs = []
    tm_coefs = []

    for theta in thetas:
        # Vacuum wave number in inverse cm
        current_k_0 = 2 * np.pi / wavelength

        # --- vacuum admittances ---
        # Kept here for clarity and symmetry, but could be just set by fiat
        cos_theta_vacuum = cosine_theta_medium(
            theta_inc=theta,
            n_inc=n_vacuum,
            n_medium=n_vacuum
        )

        Q_vacuum_TE = tilted_optical_admittance(
            cos_theta_medium=cos_theta_vacuum,
            epsilon=vacuum_elec_permittivity,
            mu=vacuum_mag_permeability,
            polarisation_mode="TE"
        )

        Q_vacuum_TM = tilted_optical_admittance(
            cos_theta_medium=cos_theta_vacuum,
            epsilon=vacuum_elec_permittivity,
            mu=vacuum_mag_permeability,
            polarisation_mode="TM"
        )

        cos_theta_ag = cosine_theta_medium(
            theta_inc=theta,
            n_inc=n_vacuum,
            n_medium=np.emath.sqrt(eps_ag)
        )

        Q_substrate_TE = tilted_optical_admittance(
            cos_theta_medium=cos_theta_ag,
            epsilon=eps_ag,
            mu=ag_mag_permeability,
            polarisation_mode="TE"
        )

        Q_substrate_TM = tilted_optical_admittance(
            cos_theta_medium=cos_theta_ag,
            epsilon=eps_ag,
            mu=ag_mag_permeability,
            polarisation_mode="TM"
        )

        # --- Thin-film characteristic matrices (SiO2) ---
        matrix_TE = thin_film_characteristic_matrix(
            d=thickness,                  # thickness in nm
            k_0=current_k_0,             # inverse cm
            n_inc=n_vacuum,
            theta_inc=theta,
            epsilon=eps_sio2,
            mu=sio2_mag_permeability,
            polarisation_mode="TE"
        )

        matrix_TM = thin_film_characteristic_matrix(
            d=thickness,
            k_0=current_k_0,
            n_inc=n_vacuum,
            theta_inc=theta,
            epsilon=eps_sio2,
            mu=sio2_mag_permeability,
            polarisation_mode="TM"
        )

        # --- Effective admittances seen from vacuum ---
        Q_final_TE = effective_admittance(
            matrix=matrix_TE,
            Q_0=Q_substrate_TE
        )

        Q_final_TM = effective_admittance(
            matrix=matrix_TM,
            Q_0=Q_substrate_TM
        )

        # --- Reflection coefficients ---
        r_coef_TE = reflection_coefficient(
            Q_vacuum=Q_vacuum_TE,
            Q_medium=Q_final_TE
        )

        r_coef_TM = reflection_coefficient(
            Q_vacuum=Q_vacuum_TM,
            Q_medium=Q_final_TM
        )

        te_coefs.append(r_coef_TE)
        tm_coefs.append(r_coef_TM)

    # Convert to arrays 
    te_coefs = np.array(te_coefs)
    tm_coefs = np.array(tm_coefs)

    return te_coefs, -1*tm_coefs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reflectance for TE and TM modes")
    parser.add_argument("thetas", help="np.array of angles theta, in radians")
    parser.add_argument("wavelength", help="wavelength of interst, in cm")

    args = parser.parse_args()

    reflectance = reflect_RB_off_mirror(args.thetas, args.wavelength)
