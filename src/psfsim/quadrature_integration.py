"""Gaussian quadrature integration optimized for exponential decay in detector.

Uses custom orthogonal polynomials built for the weight function w(z) = exp(-alpha*z)
on the finite interval [0, detector_thickness], following the standard Gaussian
quadrature construction method with three-term recurrence relations.
"""

import numpy as np
from scipy import linalg


class ExponentialDecayPolynomials:
    """
    Build orthogonal polynomials optimized for exponential decay weight function.

    Uses Gram-Schmidt orthogonalization with the weight function w(z) = exp(-alpha*z)
    on the finite interval [z_min, z_max]. The resulting polynomials are used to construct
    Gaussian quadrature nodes and weights via the Golub-Welsch algorithm.

    Computes the three-term recurrence relation coefficients and builds the Jacobi matrix
    for eigenvalue decomposition.

    Parameters
    ----------
    alpha : float
        Decay constant in weight function w(z) = exp(-alpha*z).
    z_min, z_max : float
        Integration interval bounds.
    n_order : int
        Maximum polynomial order to compute.

    """

    def __init__(self, alpha, z_min, z_max, n_order):
        """Initialize polynomial builder."""
        self.alpha = alpha
        self.z_min = z_min
        self.z_max = z_max
        self.n_order = n_order

        # Precompute moments for efficient inner products
        self._moments = self._compute_moments()

        # Compute recurrence coefficients
        self.alpha_coeffs = np.zeros(n_order)
        self.beta_coeffs = np.zeros(n_order - 1)
        self._build_recurrence_coefficients()

    def _compute_moments(self):
        """
        Compute moments m_k = integral of z^k * exp(-alpha*z) on [z_min, z_max].

        Uses analytical formulas via integration by parts.

        Returns
        -------
        moments : np.ndarray of shape (2*n_order + 3,)
            moments[k] = integral of z^k * exp(-alpha*z) on [z_min, z_max]
        """
        moments = np.zeros(2 * self.n_order + 3)

        alpha = self.alpha
        a = self.z_min
        b = self.z_max

        # m_0 = integral exp(-alpha*z) dz from a to b
        if alpha > 1e-14:
            moments[0] = (np.exp(-alpha * a) - np.exp(-alpha * b)) / alpha
        else:
            # Limit as alpha -> 0
            moments[0] = b - a

        # Recurrence: integrate by parts
        # m_{k+1} = -(b^(k+1)*exp(-alpha*b) - a^(k+1)*exp(-alpha*a))/alpha + (k+1)*m_k/alpha
        for k in range(2 * self.n_order + 1):
            if alpha > 1e-14:
                boundary = b ** (k + 1) * np.exp(-alpha * b) - a ** (k + 1) * np.exp(-alpha * a)
                moments[k + 1] = -boundary / alpha + (k + 1) * moments[k] / alpha
            else:
                # Limit: m_{k+1} = (b^{k+1} - a^{k+1}) / (k+1)
                moments[k + 1] = (b ** (k + 1) - a ** (k + 1)) / (k + 1)

        return moments

    def _inner_product(self, p_coeffs, q_coeffs):
        """
        Compute inner product of two polynomials with exponential weight.

        <p, q> = integral of p(z) * q(z) * exp(-alpha*z) dz

        Uses moment-based formula for efficiency:
        <p, q> = sum_i sum_j p_i * q_j * m_{i+j}

        Uses einsum for vectorized computation.

        Parameters
        ----------
        p_coeffs : np.ndarray
            Coefficients of polynomial p in ascending order of powers.
        q_coeffs : np.ndarray
            Coefficients of polynomial q in ascending order of powers.

        Returns
        -------
        ip : float
            Inner product.
        """
        # Create indices for moment lookup: i + j for all pairs (i, j)
        i_indices = np.arange(len(p_coeffs))
        j_indices = np.arange(len(q_coeffs))
        moment_indices = i_indices[:, None] + j_indices[None, :]  # shape (len(p), len(q))

        # Get moment values for all index pairs
        moment_vals = self._moments[moment_indices]

        # Vectorized inner product: sum_i sum_j p_i * q_j * m_{i+j}
        ip = np.einsum("i,j,ij->", p_coeffs, q_coeffs, moment_vals)

        return ip

    def _monic_poly_multiply_z(self, p_asc):
        """
        Multiply polynomial by z: if p = sum_i p_i z^i, return sum_i p_i z^{i+1}.

        Parameters
        ----------
        p_asc : np.ndarray
            Coefficients in ascending order of powers.

        Returns
        -------
        result : np.ndarray
            Coefficients of z*p in ascending order.
        """
        return np.concatenate([[0], p_asc])

    def _poly_add(self, p, q):
        """Add two polynomials, handling different lengths."""
        if len(p) < len(q):
            p, q = q, p
        result = p.copy()
        result[: len(q)] += q
        return result

    def _build_recurrence_coefficients(self):
        """
        Build three-term recurrence relation coefficients.

        Standard form: p_{n+1}(z) = (z - alpha_n) * p_n(z) - beta_n * p_{n-1}(z)

        where:
            alpha_n = <z*p_n, p_n> / <p_n, p_n>
            beta_n = <p_n, p_n> / <p_{n-1}, p_{n-1}>

        This builds the Jacobi matrix diagonals for the Golub-Welsch algorithm.
        """
        # Start with p_0 = 1
        p_prev = np.array([1.0])
        norm_prev = self._moments[0]

        # p_1 = z - alpha_0, where alpha_0 = m_1 / m_0
        alpha_0 = self._moments[1] / self._moments[0] if self._moments[0] > 0 else 0.0
        p_curr = np.array([-alpha_0, 1.0])  # ascending order: -alpha_0 + z
        norm_curr = self._inner_product(p_curr, p_curr)

        self.alpha_coeffs[0] = alpha_0

        # Iterate to build higher-order polynomials
        for n in range(1, self.n_order):
            # Compute alpha_n = <z*p_n, p_n> / <p_n, p_n>
            z_p_curr = self._monic_poly_multiply_z(p_curr)
            ip_z_p = self._inner_product(z_p_curr, p_curr)
            alpha_n = ip_z_p / norm_curr if norm_curr > 1e-14 else 0.0

            # Compute beta_n = <p_n, p_n> / <p_{n-1}, p_{n-1}>
            beta_n = norm_curr / norm_prev if norm_prev > 1e-14 else 0.0

            self.alpha_coeffs[n] = alpha_n
            self.beta_coeffs[n - 1] = beta_n

            # Build p_{n+1} = (z - alpha_n) * p_n - beta_n * p_{n-1}
            # (z - alpha_n) * p_n
            z_p = self._monic_poly_multiply_z(p_curr)
            z_minus_alpha_pn = self._poly_add(z_p, -alpha_n * p_curr)

            # Subtract beta_n * p_{n-1}, using _poly_add with negated coefficients to handle length mismatch
            p_next = self._poly_add(z_minus_alpha_pn, -beta_n * p_prev)
            if len(p_next) > 0 and abs(p_next[-1]) > 1e-14:
                p_next = p_next / p_next[-1]

            # Update for next iteration
            p_prev = p_curr
            p_curr = p_next
            norm_prev = norm_curr
            norm_curr = self._inner_product(p_curr, p_curr)

    def compute_quadrature_nodes_and_weights(self, n):
        """
        Compute Gaussian quadrature nodes and weights using Golub-Welsch algorithm.

        The Golub-Welsch algorithm:

        1. Build the symmetric tridiagonal Jacobi matrix from recurrence coefficients
        2. Compute eigenvalues (which are the quadrature nodes)
        3. Compute weights from the first row of the eigenvector matrix

        Parameters
        ----------
        n : int
            Number of quadrature points.

        Returns
        -------
        nodes : np.ndarray of shape (n,)
            Quadrature nodes (sorted, in [z_min, z_max]).
        weights : np.ndarray of shape (n,)
            Quadrature weights (corresponding to nodes). These are in the form of
            int_a^b f(z) dz = sum_j v_j f(z_j).

        """

        # Build symmetric tridiagonal Jacobi matrix
        # Diagonal: alpha_0, alpha_1, ..., alpha_{n-1}
        # Off-diagonal: sqrt(beta_1), sqrt(beta_2), ..., sqrt(beta_{n-1})

        diag = self.alpha_coeffs[:n]
        off_diag = np.sqrt(np.abs(self.beta_coeffs[: n - 1]))

        # Construct tridiagonal matrix
        jacobi = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = linalg.eigh(jacobi)

        # Nodes are eigenvalues
        nodes = eigenvalues

        # Weights from first row of eigenvector matrix and total weight
        # weight_i = mu_0 * v_{0,i}^2, where mu_0 = integral w(z) dz
        mu_0 = self._moments[0]
        weights = mu_0 * (eigenvectors[0, :] ** 2)

        # convert weights from "w" to "v" (in Numerical Recipes notation)
        weights *= np.exp(self.alpha * nodes)

        return nodes, weights


def build_exponential_decay_quadrature(alpha, z_min, z_max, n_order):
    """
    Build Gaussian quadrature nodes and weights for exponential decay weight function.

    Constructs orthogonal polynomials with respect to the weight function
    w(z) = exp(-alpha*z) on the finite interval [z_min, z_max], then extracts
    quadrature nodes and weights using the Golub-Welsch algorithm.

    Parameters
    ----------
    alpha : float
        Decay constant in weight function w(z) = exp(-alpha*z).
    z_min, z_max : float
        Integration interval.
    n_order : int
        Number of quadrature points.

    Returns
    -------
    nodes : np.ndarray of shape (n_order,)
        Quadrature nodes in [z_min, z_max].
    weights : np.ndarray of shape (n_order,)
        Quadrature weights.

    """

    builder = ExponentialDecayPolynomials(alpha, z_min, z_max, n_order)
    nodes, weights = builder.compute_quadrature_nodes_and_weights(n_order)

    return nodes, weights


class QuadratureIntegrator:
    """
    Adaptive Gaussian quadrature integrator for decaying functions.

    Optimizes quadrature nodes and weights based on the characteristic decay length
    of the intensity function in the detector. Uses custom orthogonal polynomials
    built for exponential decay weight function.

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
        self._cached_decay_constant = None

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
            Characteristic decay length (1/alpha) in microns.
        decay_constant : float
            Characteristic decay constant alpha in 1/microns.
        """
        alpha_field = self._compute_kz_imag()

        # Avoid division by zero at the pupil edge (low u)
        alpha_field[alpha_field < 1e-8] = 1e-8

        # Filter out invalid values (NaN, Inf)
        alpha_valid = alpha_field[np.isfinite(alpha_field) & (alpha_field > 0)]

        if len(alpha_valid) == 0:
            # Fallback if all values are invalid
            alpha_char = 1.0 / self.detector_thickness
            return self.detector_thickness, alpha_char

        # Characteristic decay constant (maximum, conservative estimate)
        alpha_char = np.max(alpha_valid)
        decay_length = 1.0 / alpha_char

        return decay_length, alpha_char

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
            Number of quadrature points (3 to 20).
        """
        # Ratio of decay length to detector thickness
        ratio = decay_length / self.detector_thickness
        if ratio < 0.15:
            return 9  # very sharp decay, need many points

        return 7  # will explore later

    def get_nodes_and_weights(self):
        """
        Get optimized quadrature nodes and weights.

        Uses custom orthogonal polynomials built for the exponential decay
        weight function w(z) = exp(-alpha*z) on [0, detector_thickness].

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
        decay_length, alpha_char = self.analyze_decay()
        order = self._adaptive_order(decay_length)

        # Build quadrature for exponential decay weight function
        z_nodes, z_weights = build_exponential_decay_quadrature(
            alpha_char, 0.0, self.detector_thickness, order
        )

        # Store cache
        self._cached_order = order
        self._cached_nodes = z_nodes
        self._cached_weights = z_weights
        self._cached_decay_length = decay_length
        self._cached_decay_constant = alpha_char

        return z_nodes, z_weights, order

    def integrate(self, intensity_array, axis=2):
        """
        Integrate intensity along specified axis using custom quadrature.

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
        integrated = np.tensordot(intensity_array, z_weights, axes=(axis, 0))

        return integrated
