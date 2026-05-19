"""Gaussian quadrature integration optimized for exponential decay in detector."""

import numpy as np
from scipy import special


class QuadratureIntegrator:
    """
    Adaptive Gaussian quadrature integrator for decaying functions.

    Optimizes quadrature nodes and weights based on the characteristic decay length
    of the intensity function in the detector. The implementation uses adaptive
    Gauss-Legendre quadrature on the finite detector-thickness interval, selecting
    the quadrature order to better resolve rapidly decaying intensity profiles in
    HgCdTe.

    Attributes
    ----------
    wavelength : float
        Vacuum wavelength in microns.
    detector_thickness : float
        Detector thickness in microns.
    ux, uy : np.ndarray
        Orthographic projections of ray directions (normalized pupil coordinates).
    filter : FilterDetector
        Interference filter object for accessing transmission properties.
    k0 : float
        Wave number in vacuum (2π/wavelength).
    """

    def __init__(self, wavelength, detector_thickness, ux, uy, filter_obj):
        """
        Initialize quadrature integrator.

        Parameters
        ----------
        wavelength : float
            Vacuum wavelength in microns.
        detector_thickness : float
            Detector thickness in microns.
        ux, uy : np.ndarray
            Orthographic projections of propagation directions (same shape).
        filter_obj : FilterDetector
            Interference filter object.
        """
        self.wavelength = wavelength
        self.detector_thickness = detector_thickness
        self.ux = ux
        self.uy = uy
        self.filter = filter_obj

        self.k0 = 2 * np.pi / wavelength

        # Cache for quadrature data
        self._cached_order = None
        self._cached_nodes = None
        self._cached_weights = None
        self._cached_decay_length = None

    def _compute_kz_imag(self):
        """
        Compute the imaginary part of kz (decay constant) across spatial grid.

        Returns
        -------
        kz_imag : np.ndarray (same shape as ux, uy)
            Imaginary part of wave vector in detector material.
            Positive values indicate exponential decay: I(z) ~ exp(-2*kz_imag*z).
        """
        from .filter_detector_properties import n_mercadtel

        u = np.sqrt(self.ux**2 + self.uy**2)
        mask = u <= 1.0

        # Complex refractive index of HgCdTe substrate
        n_hgcdte = n_mercadtel(self.wavelength)

        # Wave vector in substrate: kz = sqrt((k0*n)^2 - (k0*u)^2)
        kz = np.zeros_like(self.ux, dtype=np.complex128)
        kz[mask] = np.sqrt((self.k0 * n_hgcdte) ** 2 - (self.k0 * u[mask]) ** 2)

        # Choose branch with positive imaginary part (decay)
        kz[mask & (kz.imag < 0.0)] = -kz[mask & (kz.imag < 0.0)]

        return 2.0 * kz.imag  # Factor of 2 comes from |E|^2 -> I

    def analyze_decay(self):
        """
        Analyze decay behavior of intensity across spatial grid.

        Returns
        -------
        decay_length_characteristic : float
            Characteristic decay length (1/alpha) in microns, computed as the
            minimum decay length across the pupil. This is conservative and ensures
            adequate sampling even at high-decay regions.
        """
        alpha_field = self._compute_kz_imag()

        # Avoid division by zero at the pupil edge (low u)
        # Set a minimum decay constant to avoid unrealistic long decay lengths
        alpha_field[alpha_field < 1e-8] = 1e-8

        # Characteristic decay length (minimum, conservative estimate)
        decay_length = 1.0 / np.max(alpha_field)

        return decay_length

    def _adaptive_order(self, decay_length):
        """
        Select quadrature order based on decay length relative to detector thickness.

        Parameters
        ----------
        decay_length : float
            Characteristic decay length in microns.

        Returns
        -------
        order : int
            Number of quadrature points (3 to 15).
        """
        # Ratio of decay length to detector thickness
        ratio = decay_length / self.detector_thickness

        if ratio < 0.15:
            # Very sharp decay: need many points near surface
            order = 12
        elif ratio < 0.3:
            # Sharp decay
            order = 10
        elif ratio < 0.6:
            # Moderate decay
            order = 7
        elif ratio < 1.5:
            # Gradual decay
            order = 5
        else:
            # Slow/nearly linear decay
            order = 3

        return order

    def get_nodes_and_weights(self):
        """
        Get optimized quadrature nodes and weights.

        For exponential decay on finite interval [0, d], we use an adaptive strategy:
        1. Compute characteristic decay length from physics
        2. Use Gauss-Legendre as base (optimal for smooth functions on finite intervals)
        3. Adaptive order based on decay length vs detector thickness ratio

        Gauss-Legendre is chosen over Gauss-Laguerre because:
        - We have a *finite* interval [0, detector_thickness], not [0, ∞)
        - Intensity decay is smooth and well-behaved on finite interval
        - Higher order Gauss-Legendre efficiently handles exponential decay

        Returns
        -------
        z_nodes : np.ndarray
            Depth coordinates (in microns) for quadrature points.
        weights : np.ndarray
            Quadrature weights.
        order : int
            Number of quadrature points used.
        """
        # Analyze decay to get adaptive order
        decay_length = self.analyze_decay()
        order = self._adaptive_order(decay_length)

        # Get Gauss-Legendre nodes and weights on [-1, 1]
        legendre_nodes, legendre_weights = special.roots_legendre(order)

        # Transform from [-1, 1] to [0, detector_thickness]
        # Change of variables: z = (detector_thickness/2) * (x + 1) where x in [-1, 1]
        # Then dz = (detector_thickness/2) dx
        z_nodes = (self.detector_thickness / 2.0) * (legendre_nodes + 1.0)
        z_weights = (self.detector_thickness / 2.0) * legendre_weights

        # Store cache
        self._cached_order = order
        self._cached_nodes = z_nodes
        self._cached_weights = z_weights
        self._cached_decay_length = decay_length

        return z_nodes, z_weights, order

    def integrate(self, intensity_array, axis=2):
        """
        Integrate intensity along specified axis using adaptive Gaussian quadrature.

        Uses Gauss-Legendre quadrature with adaptive order selection based on the
        characteristic decay length of the intensity in the detector.

        Parameters
        ----------
        intensity_array : np.ndarray
            Intensity values with shape (..., nz) where nz is the number of z-points.
        axis : int, optional
            Axis along which to integrate (default: 2, for shape (ux, uy, z)).

        Returns
        -------
        integrated : np.ndarray
            Integrated intensity with shape matching intensity_array without the integration axis.
        """
        z_nodes, z_weights, _ = self.get_nodes_and_weights()

        # Perform weighted sum along the specified axis
        # intensity_array has shape (..., nz), weights has shape (nz_quad,)
        # We sum: sum_i weight[i] * intensity[..., i]

        # Use tensordot for efficient broadcasting
        integrated = np.tensordot(intensity_array, z_weights, axes=(axis, 0))

        return integrated
