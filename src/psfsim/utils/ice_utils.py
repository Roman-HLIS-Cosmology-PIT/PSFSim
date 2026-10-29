import numpy as np
from psfsim.filter_detector_properties import FilterDetector, n_ice, n_mercadtel
from psfsim.polarisation_decomposition import polarisation_mode_decomposition
from psfsim.psfobject import PSFObject
from psfsim.index_cdte import n_cdte
import galsim.roman



def inBandpass(wav, filter_string, bandpasses):
    """
    Compute whether a wavelength is in a filter. 
    This function is an exact copy of that in polychrom.py and is added here to avoid
    circular import errors.

    Parameters
    ----------
    wav : float
        Wavelength in microns
    filter_string: str
        String to identify the filter. Can either be just a letter or letter + wl (e.g. 'H' or 'H158').
    bandpasses : dict
        Dictionary of GalSim Roman bandpasses, typically from ``galsim.roman.getBandpasses()``.

    Returns
    -------
    bool, str
        Whether the wavelength is in the filter, and the galsim key of the filter if it is in the bandpass.

    """

    wav *= 1e3  # convert to nm for galsim
    bp = bandpasses

    # First, check for an exact match of the filter string to a bandpass key.
    if filter_string in bp:
        band = bp[filter_string]
        if band.blue_limit <= wav <= band.red_limit:
            return True, filter_string
        else:
            return False, None

    # Otherwise, look for all keys that contain the filter_string as a substring.
    matching_keys = [key for key in bp if filter_string in key]
    if not matching_keys:
        raise ValueError(
            f"Filter {filter_string} not found in bandpasses. Available filters are: {bp.keys()}"
        )

    # Check each matching key and return True on the first one whose bandpass contains the wavelength.
    for key in matching_keys:
        band = bp[key]
        if band.blue_limit <= wav <= band.red_limit:
            return True, key

    # If none of the candidate bandpasses contain the wavelength, report that it is out of band.
    return False, None

def _matmul_2x2(A, B):
    """
    Explicit multiplication of two two 2x2, similar to that in _calc_transmission_scalar()
    in filter_detector_properties.
    """
    A00, A01, A10, A11 = A
    B00, B01, B10, B11 = B
    C00 = A00 * B00 + A01 * B10
    C01 = A00 * B01 + A01 * B11
    C10 = A10 * B00 + A11 * B10
    C11 = A10 * B01 + A11 * B11
    return C00, C01, C10, C11


def _layer_matrices_from_q(ll, q, n, t, mu=1.0):
    """
    TE and TM characteristic matrix of a single layer, written in terms of q = ux^2 + uy^2.
    This is similar to _calc_transmission_scalar() in filter_detector_properties, and formulas follow 
    from FilterDetector.characteristic_matrix. The matrices only depend on the incident
    direction, so we can evaluate them on a 1D q grid instead of the full pupil.
    """
    k0 = 2.0 * np.pi / ll
    n = complex(n)
    e = n**2 
    mu = complex(mu)

    # z-component of the wave vector inside the layer and the phase accumulated through it
    kzj = k0 * np.sqrt(n**2 - q + 0j)
    phase = kzj * t
    cos_p = np.cos(phase)
    sin_p = np.sin(phase)

    # Off diagonal terms use mu for TE and the dielectric constant for TM.
    TE_j_01 = -(k0 * mu / kzj) * 1j * sin_p
    TE_j_10 = -(kzj / k0 / mu) * 1j * sin_p
    TM_j_01 = -(k0 * e / kzj) * 1j * sin_p
    TM_j_10 = -(kzj / k0 / e) * 1j * sin_p

    return {"TE": (cos_p, TE_j_01, TE_j_10, cos_p), "TM": (cos_p, TM_j_01, TM_j_10, cos_p)}


def _characteristic_matrix_from_q(interference_filter, ll, q):
    """
    Characteristic matrices of all the filter/detector layers without the ice, as in
    FilterDetector.characteristic_matrix but on a 1D q grid.
    These layers don't change with ice thickness, so we compute this once and then only multiply
    by the ice matrix for each thickness in fast_ice_flux_radial().

    """
    ns = interference_filter.n
    ts = interference_filter.t
    mus = interference_filter.mu

    # Start from the identity matrix at every q
    one = np.ones(q.shape, dtype=np.complex128)
    zero = np.zeros(q.shape, dtype=np.complex128)
    M_TE_net = (one, zero, zero, one)
    M_TM_net = (one, zero, zero, one)

    # Multiply the layers in the correct order (see FilterDetector.characteristic_matrix)
    for j in range(interference_filter.nlayer):
        M_j = _layer_matrices_from_q(ll, q, ns[j], ts[j], mus[j])
        M_TE_net = _matmul_2x2(M_TE_net, M_j["TE"])
        M_TM_net = _matmul_2x2(M_TM_net, M_j["TM"])

    return {"TE": M_TE_net, "TM": M_TM_net}


def _deposit_linear_to_q_grid(q_values, weights, q_grid):
    """
    Compress the 2D pupil weights onto the 1D q grid.
    The filter transmission only depends on q, but the Roman pupil is not radially
    symmetric, so we can't just use a radial profile.
    Instead each pupil point deposits its weight onto the two nearest q grid points
    with linear weights, which conserves the total weight.
    """
    nq = len(q_grid)
    dq = (q_grid[-1] - q_grid[0]) / (nq - 1)

    # Fractional grid index of each point, e.g. x = 15.3 is 30% of the way from bin 15 to 16
    x = (q_values - q_grid[0]) / dq
    i0 = np.clip(np.floor(x).astype(np.int64), 0, nq - 2) # lower neighbour, i0+1 must stay in range
    frac = np.clip(x - i0, 0.0, 1.0)

    out = np.bincount(i0, weights=weights * (1.0 - frac), minlength=nq)
    out += np.bincount(i0 + 1, weights=weights * frac, minlength=nq)
    return out


def _prepare_fast_ice_flux_radial(psf_obj, n_q=2048):
    """
    Precompute everything in the flux calculation that does not depend on ice thickness.
    This is the slow part since it starts from the full Roman pupil, but it only needs
    to run once per wavelength and detector position.
    The code here flollows the flux calculation in PSFObject.get_Intensity_in_detector
    and get_Intensity_from_E, and FilterDetector.transmission and transmitted_E.

    Parameters
    ----------
    psf_obj : PSFObject
        Monochromatic PSFObject at the wavelength and detector position of interest.
    n_q : int
        Number of points in the 1D q grid. Default: 2048.

    Returns
    -------
    cache : dict
        Everything needed by fast_ice_flux_radial().
    """
    ll = float(psf_obj.wavelength)
    k0 = 2.0 * np.pi / ll
    ux = np.asarray(psf_obj.ux)
    uy = np.asarray(psf_obj.uy)
    mask = psf_obj.mask 

    # The filter only depends on the incident direction through q = ux^2 + uy^2 = sin(theta)^2
    q = ux**2 + uy**2

    # The decomposition is stored in the PSFObject once get_Intensity_in_detector() has run,
    # otherwise we compute it from the ray-traced field
    TE_TM_h_polarized = psf_obj.TE_TM_h_polarized
    TE_TM_v_polarized = psf_obj.TE_TM_v_polarized
    if TE_TM_h_polarized is None or TE_TM_v_polarized is None:
        E_FPA_h_polarized = psf_obj.optics.rb.E[:, :, 0, 1:4] # horizontal polarization
        E_FPA_v_polarized = psf_obj.optics.rb.E[:, :, 1, 1:4] # vertical polarization
        TE_TM_h_polarized = polarisation_mode_decomposition(ux, uy, E_FPA_h_polarized, sgn=1)
        TE_TM_v_polarized = polarisation_mode_decomposition(ux, uy, E_FPA_v_polarized, sgn=1)

    A_TE_h = TE_TM_h_polarized["TE"]
    A_TM_h = TE_TM_h_polarized["TM"]
    A_TE_v = TE_TM_v_polarized["TE"]
    A_TM_v = TE_TM_v_polarized["TM"]

    A_TE2 = np.abs(A_TE_h)**2 + np.abs(A_TE_v)**2
    A_TM2 = np.abs(A_TM_h)**2 + np.abs(A_TM_v)**2

    # The total flux in the image is the pupil sum of |E|^2 up to a constant, so only |prefactor|^2
    # matters and the phases drop out.
    prefactor2 = np.abs(psf_obj.prefactor)**2

    # Fields inside the HgCdTe detector, see FilterDetector.transmitted_E
    nHgCdTe = complex(n_mercadtel(ll))
    eHgCdTe = nHgCdTe**2

    # kz in the detector, choosing the root with positive imaginary part
    kz = np.zeros_like(q, dtype=np.complex128)
    kz[mask] = k0 * np.sqrt(nHgCdTe**2 - q[mask] + 0j)
    kz[mask & (kz.imag < 0.0)] *= -1.0

    # The TM mode contribution
    tm_E_factor = np.zeros_like(q, dtype=float)
    tm_E_factor[mask] = (np.abs(kz[mask] / (k0 * eHgCdTe))**2 + np.abs(np.sqrt(q[mask]) / eHgCdTe)**2)

    # get_Intensity_from_E integrates over the detector depth L with Gaussian quadrature,
    # here we do it analytically
    if psf_obj.interference_filter.sgn != 1:
        raise ValueError("The depth integral assumes FilterDetector.sgn = 1")
    L = float(psf_obj.detector_thickness)
    alpha = 2.0 * np.imag(kz)
    depth_factor = np.zeros_like(q, dtype=float)
    decaying = mask & (alpha > 1e-12)
    depth_factor[decaying] = (1.0 - np.exp(-alpha[decaying] * L)) / alpha[decaying]
    depth_factor[mask & ~decaying] = L

    ## Pupil weights multiplying |T_TE|^2 and |T_TM|^2
    W_TE_2d = A_TE2 * prefactor2 * depth_factor
    W_TM_2d = A_TM2 * prefactor2 * tm_E_factor * depth_factor

    # Keep only physical and finite pupil points, and compress them onto a q grid
    # that extends up to the edge of the pupil
    valid = mask & np.isfinite(W_TE_2d) & np.isfinite(W_TM_2d)
    q_valid = q[valid]
    q_grid = np.linspace(0.0, np.max(q_valid), int(n_q))
    W_TE = _deposit_linear_to_q_grid(q_valid, W_TE_2d[valid], q_grid)
    W_TM = _deposit_linear_to_q_grid(q_valid, W_TM_2d[valid], q_grid)

    ## Filter stack without ice and quantities for the transmission coefficients
    interference_filter = psf_obj.interference_filter
    M_net = _characteristic_matrix_from_q(interference_filter, ll, q_grid)

    cos_theta = np.sqrt(1.0 - q_grid + 0j)

    # kz in the HgCdTe substrate on the q grid
    kz_sub = k0 * np.sqrt(nHgCdTe**2 - q_grid + 0j)
    kz_sub[kz_sub.imag < 0.0] *= -1.0

    return {
        "wavelength": ll,
        "q_grid": q_grid,
        "W_TE": W_TE,
        "W_TM": W_TM,
        "M_TE_net": M_net["TE"],
        "M_TM_net": M_net["TM"],
        "cos_theta": cos_theta,
        "kz_sub": kz_sub,
        "nHgCdTe": nHgCdTe,
        "eHgCdTe": eHgCdTe,
        "muHgCdTe": complex(interference_filter.muHgCdTe),
    }


def _fast_ice_flux_radial(cache,thickness_um):
    """
    Calculate the total detector flux for one or list of ice thicknesses.
    This function only deals with the ice-layer matrix changes with thickness,
    all other elements of the matrix are dealt in the cache.
    This function basically combines the code in characteristic_matrix and transmission
    from filter_detector_properties. But since relevant variables for ice only
    are not saved, we need to compute them here again.

    Parameters
    ----------
    cache : dict
        Output from _prepare_fast_ice_flux_radial().

    thickness_um : float or ndarray
        Ice thickness or list/array of ice thicknesses in microns.

    Returns
    -------
    flux : float or ndarray
        Relative unnormalized detector flux.
    """

    t = np.atleast_1d(thickness_um).astype(float)
    lam = cache["wavelength"]
    q = cache["q_grid"]
    k0 = 2.0 * np.pi / lam

    # Refractive index
    nice = complex(n_ice(lam))
    # Ice dielectric constant
    eice = nice**2

    # z-component of the wave vector inside the ice. This depends on q and lambda, but NOT thickness.
    kz_ice = k0* np.sqrt(nice**2- q+ 0j)

    # This gives the ice phase at every thickness and q value.
    phase = t[:, None]* kz_ice[None, :]
    c = np.cos(phase)
    s = np.sin(phase)

    # Now we build the electric field modes (TM and TE) characteristic matrix.
    # For ice magnetic constant is essentially 1, so not included in TE calculation
    M00, M11 = c, c # M00 and M11 are the same
    M01 = -(k0 / kz_ice[None, :])* 1j * s
    M10 = -(kz_ice[None, :] / k0)* 1j * s
    I_TE = (M00, M01, M10, M11)

    # Now for the TM matrix, M01 and M10 are identical but we need to include the dielectric constant
    I_TM = (M00, M01*eice, M10/eice, M11)

    # Now we multiply the ice matrices with the characteristic matrix without ice.
    # characteristic_matrix puts the ice as the first layer, so the ice matrix goes on the left.
    M_TE = _matmul_2x2(I_TE,cache["M_TE_net"])
    M_TM = _matmul_2x2(I_TM,cache["M_TM_net"])

    M00s, M01s, M10s, M11s = M_TE
    M00p, M01p, M10p, M11p = M_TM

    cos_theta = cache["cos_theta"][None, :]
    kz = cache["kz_sub"][None, :]

    # Now we calculate the TE and TM transmission coefficients. This is basically identical
    # to what filter_detector_properties.transmission does.

    # TE
    eta_TE = kz/ k0/ cache["muHgCdTe"]
    denom_TE = cos_theta*(M00s+ eta_TE * M01s)+(M10s+ eta_TE * M11s)
    T_TE = 2.0* cos_theta/ denom_TE
    eta_TM = kz/ k0/ cache["eHgCdTe"]

    # TM
    denom_TM = cos_theta* (M00p + eta_TM * M01p)+(M10p + eta_TM * M11p)
    T_TM = 2.0*cos_theta/denom_TM

    # All that is left is to integrate over the Roman pupil to get the flux.
    flux = np.sum(cache["W_TE"][None, :]* np.abs(T_TE)**2+
                  cache["W_TM"][None, :]* np.abs(T_TM)**2,
                  axis=1,
                 )

    if np.ndim(thickness_um) == 0:
        return flux[0]

    return flux


def get_ice_transmission_wave(base_psf, ice_thickness):
    """
    Ice transmission F(lambda,t) / F(lambda,0) for a single monochromatic PSF.
    Builds the cache for the PSF wavelength and detector position and evaluates
    the flux ratio at the given ice thickness.

    Parameters
    ----------
    base_psf : PSFObject
        Monochromatic PSFObject at the wavelength and detector position of interest.
    ice_thickness : float or ndarray
        Ice thickness or array of ice thicknesses in microns.

    Returns
    -------
    float or ndarray
        Flux ratio relative to no ice, one value per thickness.
    """
    cache = _prepare_fast_ice_flux_radial(base_psf)
     #Flux at the some ice thickness.
    flux_t = _fast_ice_flux_radial(cache,thickness_um)
    # Flux with zero ice thickness.
    flux_0 = _fast_ice_flux_radial(cache, 0.0)
    ice_transmission = flux_t / flux_0
    return ice_transmission


def _get_PSF(wave,scanum, scax, scay, use_filter, **kwargs):
    """
    Build a monochromatic PSFObject at a given SCA position.
    """
    # convert positions to analysis frame
    scax_ana = 0.01 * (scax - 2043.5)
    scay_ana = -0.01 * (scay - 2043.5)
    base_psf = PSFObject(
        scanum=scanum,
        scax=scax_ana,
        scay=scay_ana,
        wavelength=wave,
        use_filter=use_filter,
        **kwargs
        )
    return base_psf

def _parse_filter_name(filter_name):
    """
    Convert a Roman filter name to its full bandpass name
    """
    bandpass_names = { 'R': 'R062',  'Z': 'Z087',  'Y': 'Y106',  'J': 'J129',
                       'H': 'H158',  'F': 'F184', 'K':'K213', 'W': 'W146'}
    if filter_name in bandpass_names.keys():
        return bandpass_names[filter_name]
    if filter_name in bandpass_names.values():
        return filter_name
    else:
        raise ValueError(f"Filter name must be a Roman filter: {list(bandpass_names.values())}")

def get_ice_transmission(scanum = 1, scax = 2044, scay = 2044, filter_name = None, waves = None,
                               num_waves = 20, ice_thickness = 0.0, ice_response = None, return_throughput = False,  **psf_kwargs):
    """
    Ice transmission F(lambda,t) / F(lambda,0) as a function of wavelength at a given SCA position.
    If only filter_name is passed, the transmission is evaluated at num_waves wavelengths across
    the bandpass. If waves and filter_name are both passed, only the waves inside the bandpass
    are kept. If only waves is passed, all of them are used.

    Parameters
    ----------
    scanum : int
        SCA number (1-18).
    scax, scay : float
        Position on the SCA in science pixel coordinates.
    filter_name : str
        Roman filter, single letter ('H') or full name ('H158').
    waves : array_like
        Wavelengths in microns.
    num_waves : int
        Number of wavelengths across the bandpass when waves is not passed.
    ice_thickness : float or ndarray
        Ice thickness or array of ice thicknesses in microns.
    ice_response : callable
        Optional precomputed ice response, called as ice_response(scanum, scax, scay, wave, ice_thickness).
        If None, the transmission is calculated from the PSF with get_ice_transmission_wave().
    return_throughput : bool
        Also return the filter throughput multiplied by the ice transmission. Needs filter_name.
    **psf_kwargs
        Any other PSFObject arguments.

    Returns
    -------
    wave_list : ndarray
        Wavelengths in microns where the transmission was evaluated.
    transmission_ratio : ndarray
        Ice transmission at each wavelength, shape (n_waves,) or (n_waves, n_thickness).
    throughput : ndarray
        Only if return_throughput = True. Bandpass throughput times the ice transmission.
    """
    if waves is None and filter_name is None:
        raise ValueError("One of waves or filter_name must be defined")
    if return_throughput and (filter_name is None):
        # in order to return the new throuhput a filter name must be passed
        raise ValueError("filter_name must be defined if return_throughput = True ")


    wave_list = waves
    if filter_name is not None:
        use_filter = _parse_filter_name(filter_name)
        bandpass = galsim.roman.getBandpasses()[use_filter]
        if waves is None:
            wave_list = np.linspace(bandpass.blue_limit, bandpass.red_limit, num_waves)
            wave_list /= 1000 # convert to microns
        else:
            wavelengths = np.asarray(waves, dtype=float)
            # multiply by 1e3 because galsim bandpass takes nm, and this function takes microns
            is_in_bandpass = [inBandpass(wav, use_filter, self.bandpass)[0] for wav in wavelengths*1e3]
            wave_list = waves[is_in_bandpass]
            if len(wave_list) == 0:
                raise ValueError(
                f"No in-band wavelength nodes found for filter '{use_filter}'. "
                f"Provided range: [{wavelengths.min():.6g}, {wavelengths.max():.6g}] microns."
                )

    transmission_ratio = []
    for wave in wave_list:
        base_psf = _get_PSF(wave,scanum, scax, scay, use_filter, **psf_kwargs)
        if ice_response is None:
           ice_transmission = get_ice_transmission_wave(base_psf, ice_thickness)
        else:
            ice_transmission = ice_response(scanum, scax, scay, wave, ice_thickness )
        transmission_ratio.append(ice_transmission)

    transmission_ratio = np.array(transmission_ratio)
    if return_throughput:
        return wave_list, transmission_ratio, transmission_ratio*bandpass(wave_list*1000)
    else:
        return wave_list, transmission_ratio
