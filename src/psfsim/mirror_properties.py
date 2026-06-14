# mirror_properties class
# For use with RomanRayBundle objects to calculate
# reflectance coefficients for s and p polarization modes
#
# Created 16-Apr-2026
# Developer: Anthony Harbo Torres
# with technical guidance by Christopher Hirata
# and Charuhas Shiveshwarkar
# Model follows Sec 1.6 of Principles of Optics by Born & Wolf
# version:0.2

# imports
import argparse
from importlib.resources import files

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


# optical functions
def n_medium(epsilon, mu):
    """
    Compute n of this medium.

    Parameters
    -----------
    epsilon, mu: complex
        The dielectric and diamagnetic constants.

    Returns
    --------
    complex
        The complex index of refraction.

    """

    return np.emath.sqrt(epsilon * mu)


def cosine_theta_medium(theta_inc, n_inc, n_medium):
    """
    Compute cos(theta_medium) from conserved transverse wavevector.

    This is ``n_inc * sin(theta_inc) = n_med * sin(theta_med)``

    Parameters
    ----------
    theta_inc: float
        Incident angle of ``RomanRayBundle`` onto the surface of the medium
    n_inc, n_medium : complex
        Indices of refraction for the incident (usually vacuum) and medium

    Returns
    -------
    complex
        Cosine of the reduced angle in the medium. May be complex if the medium
        is lossy.

    """

    return np.emath.sqrt(1 - ((n_inc / n_medium) * np.sin(theta_inc)) ** 2)


def tilted_optical_admittance(cos_theta_medium, epsilon, mu, polarisation_mode):
    """
    Tilted optical admittance, Q(z) = [U(z),V(z)]

    For TE: Q = (1/z) * cos(theta_medium)
    For TM: Q = z * cos(theta_medium)

    Parameters
    ----------
    cos_theta_medium : float or np.ndarray of float
        Cosine of the angle theta in that layer
    epsilon, mu : complex
        Elec. permittivity and mag. permeability of the layer
    polarisation_mode : str
        Can pass either {TM or P} or {TE or S} as choices

    Returns
    -------
    complex
        Q(z); form depends on polarisation_mode.

    """
    z = np.emath.sqrt(mu / epsilon)

    polarisation_mode = polarisation_mode.lower()
    if polarisation_mode in ("te", "s"):
        return cos_theta_medium / z
    elif polarisation_mode in ("tm", "p"):
        return 1.0 / (cos_theta_medium * z)


def thin_film_characteristic_matrix(thickness, k_0, n_inc, theta_inc, epsilon, mu, polarisation_mode):
    """
    Characteristic matrix for a single thin film layer.

    Parameters
    ----------
    thickness: float
        Thickness of the thin film in nm
    k_0: float
        Vacuum wavevector in inverse mm
    n_inc: complex
            Refractive index of the incident medium, used to
        define the conserved transverse wavevector
    theta_inc: float or np.ndarray of float
        Angle of incidence rel. to normal in radians
    epsilon, mu: complex
        Relative elec. permittivity and mag. permeability of this layer
    polarisation_mode: str
        Which polarisation mode is being solved for, TE or TM

    Returns
    -------
    np.array of complex
        Characteristic matrix for this layer, (2x2)

    """

    # change incoming d in nm to mm
    thickness = thickness * 1.0e-6

    # compute n of this medium
    index_of_medium = n_medium(epsilon=epsilon, mu=mu)

    # compute cos(theta_medium) from conserved transverse wavevector
    cos_theta_med = cosine_theta_medium(theta_inc=theta_inc, n_inc=n_inc, n_medium=index_of_medium)

    # factor in the characteristic matrix
    if polarisation_mode.lower() in ("te", "s"):
        gamma = index_of_medium * cos_theta_med
    elif polarisation_mode.lower() in ("tm", "p"):
        gamma = index_of_medium / cos_theta_med
    else:
        raise ValueError(f"Invalid polarisation mode: {polarisation_mode:s}")

    # precalculate the argument of the sines and cosines
    argument = k_0 * thickness * index_of_medium * cos_theta_med

    # compute the matrix
    matrix = np.array(
        [
            [np.cos(argument), -np.sin(argument) * 1j / gamma],
            [-np.sin(argument) * 1j * gamma, np.cos(argument)],
        ]
    )

    return matrix


def effective_admittance(matrix, Q_0):
    """
    Effective admittance seen at the entrance of the layer stack, given substrate
    admittance Q_0 and the characteristic matrix for the thin film above it.

    Parameters
    ----------
    matrix: np.array
        Characteristic matrix of thin film above substrate
    Q_0: complex
        The optical admittance of the substrate

    Returns
    -------
    Q: complex
        Effective admittance for the layer stack

    """

    A, B, C, D = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]

    Q = (C + D * Q_0) / (A + B * Q_0)
    return Q


def reflection_coefficient(Q_vacuum, Q_medium):
    """
    Reflection coefficient for an incident beam going from vacuum into a medium

    Parameters
    ----------
    Q_vacuum: complex
        Admittance in vacuum
    Q_medium: complex or np.ndarray of complex
        Optical admittance of the medium

    Returns
    -------
    complex or np.ndarray of complex
        Complex-valued reflection coefficient

    """

    r_coef = (Q_vacuum - Q_medium) / (Q_vacuum + Q_medium)

    return r_coef


# end of optical functions


# wavelength dependent epsilon functions
def sio2_epsilon(wavelength: float):
    """
    Computes the Sellmeier formula for n^2 from Malitson, 1965.

    Parameters
    ----------
    wavelength : float
        Wavelength in microns.

    Returns
    -------
    epsilon : float
        SiO2 electric permittivity (real valued).

    """

    n_squared = (
        1
        + (((0.6961663) * wavelength**2) / (wavelength**2 - (0.0684043) ** 2))
        + (((0.4079426) * wavelength**2) / (wavelength**2 - (0.1162414) ** 2))
        + (((0.8974794) * wavelength**2) / (wavelength**2 - (9.896161) ** 2))
    )
    return n_squared


def ag_epsilon(wavelength: float):
    """
    Computes the Yang et al 2015 dielectric function.

    Parameters
    ----------
    wavelength: float
        wavelength in microns

    Returns
    -------
    complex
        Complex-valued dielectric constant of silver.

    """

    datafile = files("psfsim.data").joinpath("mirror_Ag_C_corrected.csv")  # reads in data from directory
    data = pd.read_csv(datafile, sep=",")

    duplicates = data["Wavelength (um)"].duplicated(keep="first")
    duplicates_indices = data["Wavelength (um)"].index[duplicates].tolist()
    data_reduced = data.drop(index=duplicates_indices)

    wavelengths = np.array(data_reduced["Wavelength (um)"]).flatten()

    real_eps = np.array(data_reduced["ep1"]).flatten()
    imag_eps = np.array(data_reduced["ep2"]).flatten()
    total_eps = real_eps + 1j * imag_eps

    interpolation = CubicSpline(wavelengths, total_eps)
    ag_epsilon = interpolation(wavelength)
    return ag_epsilon


# end of epsilon functions


# main script
def reflect_RB_off_mirror(thetas, wavelength, epsilon_coat=2.1, thickness=110.0, reduce=0.0):
    """Mirror class that calculates the S and P reflectances for a set
    of angles theta, for a given wavelength

    Parameters
    ----------
    thetas : np.array
        Angles in radians
    wavelength : float
        wavelength in mm
    epsilon_coat : float or list, optional
        The dielectric constant of the coating layer (set to ``None`` for SiO2).
    thickness: float or list, optional
        Thickness of the coating in nm.
    reduce : float, optional
        If specified, adjust the epsilon_2 (lossy part) of the silver at short wavelengths.
        (This varies by sample, so we allow it when doing an empirical fit.)

    Returns
    -------
    te_ceofs, tm_coefs: complex
        Complex-valued reflection coefficients for the TE & TM modes

    Notes
    -----
    `epsilon_coat` and `thickness` should be either both floats (single layer) or
    lists (multi-layer, in order from the silver toward vacuum).

    """
    # eventually set these as params, todo

    ag_mag_permeability = 1 + 0j
    sio2_mag_permeability = 1 + 0j
    vacuum_mag_permeability = 1 + 0j
    vacuum_elec_permittivity = 1 + 0j

    # Refractive index of incident medium (vacuum)
    n_vacuum = 1 + 0j

    # Refractive index of substrate (silver)
    eps_ag = ag_epsilon(wavelength=wavelength * 1e3)  # wavelength is in mm, sent as microns
    eps_ag -= reduce * 1j

    # Refractive index of thin film coating (single layer, SiO2)
    eps_sio2 = sio2_epsilon(wavelength=wavelength * 1e3)  # wavelength is in mm, sent as microns
    eps_coat = eps_sio2 if epsilon_coat is None else epsilon_coat
    mu_coat = sio2_mag_permeability if epsilon_coat is None else 1 + 0j

    # Convert to list
    ec = np.array(eps_coat).ravel()
    d = np.array(thickness).ravel()
    nlayer = np.size(eps_coat)

    te_coefs = []
    tm_coefs = []

    # Vacuum wave number in inverse cm
    current_k_0 = 2 * np.pi / wavelength

    # vacuum admittances
    # Kept for now, but could be just set by fiat
    cos_theta_vacuum = cosine_theta_medium(theta_inc=thetas, n_inc=n_vacuum, n_medium=n_vacuum)

    Q_vacuum_TE = tilted_optical_admittance(
        cos_theta_medium=cos_theta_vacuum,
        epsilon=vacuum_elec_permittivity,
        mu=vacuum_mag_permeability,
        polarisation_mode="TE",
    )

    Q_vacuum_TM = tilted_optical_admittance(
        cos_theta_medium=cos_theta_vacuum,
        epsilon=vacuum_elec_permittivity,
        mu=vacuum_mag_permeability,
        polarisation_mode="TM",
    )

    cos_theta_ag = cosine_theta_medium(theta_inc=thetas, n_inc=n_vacuum, n_medium=np.emath.sqrt(eps_ag))

    Q_substrate_TE = tilted_optical_admittance(
        cos_theta_medium=cos_theta_ag, epsilon=eps_ag, mu=ag_mag_permeability, polarisation_mode="TE"
    )

    Q_substrate_TM = tilted_optical_admittance(
        cos_theta_medium=cos_theta_ag, epsilon=eps_ag, mu=ag_mag_permeability, polarisation_mode="TM"
    )

    # Thin-film characteristic matrices (SiO2)
    matrix_TE = np.identity(2).astype(np.complex128)
    matrix_TM = np.identity(2).astype(np.complex128)
    for ilayer in range(nlayer):
        smatrix_TE = thin_film_characteristic_matrix(
            thickness=d[ilayer],  # thickness in nm
            k_0=current_k_0,  # inverse mm
            n_inc=n_vacuum,
            theta_inc=thetas,
            epsilon=ec[ilayer],
            mu=mu_coat,
            polarisation_mode="TE",
        )

        smatrix_TM = thin_film_characteristic_matrix(
            thickness=d[ilayer],
            k_0=current_k_0,
            n_inc=n_vacuum,
            theta_inc=thetas,
            epsilon=ec[ilayer],
            mu=mu_coat,
            polarisation_mode="TM",
        )

        matrix_TE = np.einsum("ij...,jk...->ik...", smatrix_TE, matrix_TE)
        matrix_TM = np.einsum("ij...,jk...->ik...", smatrix_TM, matrix_TM)

    # Effective admittances seen from vacuum
    Q_final_TE = effective_admittance(matrix=matrix_TE, Q_0=Q_substrate_TE)
    Q_final_TM = effective_admittance(matrix=matrix_TM, Q_0=Q_substrate_TM)

    # Reflection coefficients
    te_coefs = reflection_coefficient(Q_vacuum=Q_vacuum_TE, Q_medium=Q_final_TE)
    tm_coefs = reflection_coefficient(Q_vacuum=Q_vacuum_TM, Q_medium=Q_final_TM)

    # Convert to numpy arrays
    if isinstance(thetas, np.ndarray | list):
        te_coefs = np.array(te_coefs).reshape(np.shape(thetas))
        tm_coefs = np.array(tm_coefs).reshape(np.shape(thetas))
    else:
        te_coefs = np.array(te_coefs).ravel()[0]
        tm_coefs = np.array(tm_coefs).ravel()[0]

    return te_coefs, tm_coefs


def reflect_RB_model(thetas, wavelength):
    """
    Mirror reflection coefficient model.

    This is packaged so you can use it in ``romantrace.py``.

    Right now this is an idealized 1-layer coating. It gets the right behavior for the
    reflectivities, and the correct linear retardance behavior with zero-crossing at 600 nm.

    The reflectivities at 45 degrees (compared to the data we got from L3 Harris) are:

    +------------+-------------+------------+-------------+------------+
    | Wavelength | S-pol model | S-pol data | P-pol model | P-pol data |
    +------------+-------------+------------+-------------+------------+
    | 500 nm     | 98.8%       | 99.0%      | 97.5%       | 97.8%      |
    +------------+-------------+------------+-------------+------------+
    | 1100 nm    | 97.9%       | 97.8%      | 98.2%       | 97.8%      |
    +------------+-------------+------------+-------------+------------+
    | 2400 nm    | 99.2%       | 99.5%      | 98.7%       | 98.5%      |
    +------------+-------------+------------+-------------+------------+

    So for now I recommend using this only to assess how significant the polarization effects
    are likely to be --- this isn't the "truth"!

    There is a phase shift inserted here so that the reflectivity is measured relative to an
    ideal surface at 160 nm below (toward substrate) relative to the top surface of the protective
    coating.

    Parameters
    ----------
    thetas : np.array
        Angles in radians
    wavelength : float
        wavelength in mm

    Returns
    -------
    te_ceofs, tm_coefs: complex
        Complex-valued reflection coefficients for the TE & TM modes

    """

    return np.exp(-4j * np.pi / wavelength * 1.6e-4 * np.cos(thetas)) * reflect_RB_off_mirror(
        thetas, wavelength, epsilon_coat=2.25, thickness=166.0, reduce=0.0
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reflectance for TE and TM modes")
    parser.add_argument("thetas", help="np.array of angles theta, in radians")
    parser.add_argument("wavelength", help="wavelength of interest, in mm")

    args = parser.parse_args()

    reflectance = reflect_RB_off_mirror(args.thetas, args.wavelength)
