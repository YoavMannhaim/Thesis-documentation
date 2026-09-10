"""
Critical peel force models for the adhesive gripper.

Provides two independent analytical models for the maximum force a single
bonded edge can carry before peel failure, matching the two stress
formulations developed in the thesis:

  1. max_peel_force_kaelble(...)
       Kaelble (1960) model  -- thesis sections 3.1.4.1 and 5.2.3.2
  2. max_peel_force_goland_reissner(...)
       Goland-Reissner (1944) model -- thesis sections 3.1.4.2 and 5.2.3.3

Both share the same call signature and the same failure criterion so their
predictions are directly comparable.

max_peel_force(...) is retained as a dispatching wrapper for backwards
compatibility with existing Max_movement_2D.py call sites.
"""

import math


# The thesis states explicitly (Eq. 5.65-5.68) that in both the Kaelble and
# GR formulations the stress is measured as adhesive work, so its units are
#       [sigma] = J/m^2 = N/m
# i.e. sigma is a line quantity (force per unit width), not a Pa stress, and
# the concentrated force is recovered as F(x) = b * sigma(x)  (Eq. 5.67-5.68).
#
# The critical value of that line quantity is the adhesive fracture energy
# itself:
#       sigma_c = gamma          [J/m^2 = N/m]
#
# This is exactly the quantity measured in thesis section 5.3.4 (Surface
# Energy Test, Tables 5.9/5.10) and is what makes Kendall's Eq. 3.9
# dimensionally consistent. Both models below fail when their peak
# line-stress reaches gamma.
def critical_peel_stress(surface_energy, youngs_modulus=None, adhesive_thickness=None):
    """
    Critical peel line-stress, sigma_c = gamma  [J/m^2 = N/m].

    youngs_modulus / adhesive_thickness are accepted for signature stability
    but are not used: in the thesis formulation sigma is already an energy per
    unit area, so the critical value is the fracture energy directly.
    """
    return surface_energy


def _compressive_limit(length, width, thickness, youngs_modulus,
                       max_compressive_strain=0.10):
    """
    Compressive limit for a bonded adhesive layer.

    FIX: the previous implementation applied the Euler pinned-pinned buckling
    formula, P = pi^2*E*I/L^2, treating the adhesive film as a slender column.
    That is not a physically available failure mode here: the layer is bonded on
    BOTH faces (backing above, substrate below), so it is laterally restrained
    and cannot buckle in that manner. With realistic film thickness
    (t_a = 60 um) the second moment of area I = b*t^3/12 becomes ~1e-15 m^4 and
    the formula returned ~1.3 uN -- four orders of magnitude below the weight of
    the gripped object, so compression became the binding constraint on every
    motion and nothing was feasible.

    A bonded layer loaded in compression is reacted directly by the substrate.
    The relevant limit is therefore compressive yield of the adhesive over the
    BONDED AREA (length x width), not buckling of its cross-section:

        P_comp = sigma_c * A_bond ,    sigma_c = E * eps_max

    Note this is deliberately generous. Confined compression of a near
    incompressible elastomer (nu ~ 0.5) is very stiff, so in practice
    compression is not expected to be the limiting mode -- which is the
    physically correct behaviour for an adhesive pad being pressed onto a
    surface.
    """
    A_bond = length * width
    sigma_c = youngs_modulus * max_compressive_strain
    return sigma_c * A_bond


# --- Model 1: Kaelble ---
def max_peel_force_kaelble(length, width, thickness, peel_angle,
                           youngs_modulus, surface_energy,
                           force_type='tensile', return_params=False):
    """
    Maximum force before peel failure using the Kaelble (1960) stress model.

    Thesis reference: sections 3.1.4.1 (Eq. 3.29) and 5.2.3.2 (Eq. 5.69-5.71).

        sigma(x) = sigma_0 * ( cos(alpha*x) + K*sin(alpha*x) ) * exp(-alpha*x)
        sigma_0  = 2*P*sin(theta) / ( b * (1 - K) )
        K        = Mc / ( Mc + P*sin(theta) )
        alpha    = ( E_a * b / (4 * E * I * a) ) ** 0.25

    The profile peaks at the peel front (x = 0) where sigma(0) = sigma_0, so
    failure is taken as sigma_0 = sigma_c:

        P_cr = sigma_c * b * (1 - K) / ( 2 * sin(theta) )

    With no externally applied moment the bending term vanishes and K -> 0.
    """
    theta = math.radians(peel_angle) if peel_angle > math.pi else peel_angle

    if force_type == 'compressive':
        val = _compressive_limit(length, width, thickness, youngs_modulus)
        return (val, {}) if return_params else val
    if force_type != 'tensile':
        raise ValueError("force_type must be 'tensile' or 'compressive'")

    # Kendall's theta is measured from the surface normal; the peel_angle
    # argument is measured from the surface. Convert.
    theta_from_normal = (math.pi / 2.0) - theta

    sin_t = math.sin(theta)
    if sin_t <= 0:
        return (None, {}) if return_params else None

    sigma_c = critical_peel_stress(surface_energy)
    K = 0.0   # no externally applied bending moment

    # The thesis closes the Kaelble branch with Kendall's energy balance
    # (Eq. 3.9): P_cr = gamma*b / (1 - sin(theta)), where theta is measured
    # from the SURFACE NORMAL. A 90 deg peel (film pulled perpendicular to the
    # surface) corresponds to theta = 0 -> P_cr = gamma*b.
    denom = 1.0 - math.sin(theta_from_normal)
    if denom <= 1e-9:
        return (None, {}) if return_params else None
    P_cr = sigma_c * width / denom

    if not math.isfinite(P_cr) or P_cr <= 0:
        return (None, {}) if return_params else None
    if not return_params:
        return P_cr

    I = width * (thickness ** 3) / 12.0
    denom = 4.0 * youngs_modulus * I * length
    alpha = ((youngs_modulus * width) / denom) ** 0.25 if denom > 0 else float('inf')
    sigma_0 = 2.0 * P_cr * sin_t / (width * (1.0 - K))
    return P_cr, {'sigma_0': sigma_0, 'alpha': alpha, 'K': K, 'sigma_c': sigma_c}


# --- Model 2: Goland-Reissner ---
def max_peel_force_goland_reissner(length, width, thickness, peel_angle,
                                   youngs_modulus, surface_energy,
                                   force_type='tensile',
                                   adherend_thickness=None,
                                   adherend_modulus=None,
                                   return_params=False):
    """
    Maximum force before peel failure using the Goland-Reissner (1944) model.

    Thesis reference: sections 3.1.4.2 (Eq. 3.63) and 5.2.3.3 (Eq. 5.79-5.82).

        sigma(x) = sigma_0 * [ R4*cosh(lam*x/c)*cos(lam*x/c)
                             + R5*sinh(lam*x/c)*sin(lam*x/c) ]
        sigma_0  = P*t / ( R1 * b * c^2 )
        R1  = ( sinh(2*lam) + sin(2*lam) ) / 2
        R2  = sinh(lam)*cos(lam) - cosh(lam)*sin(lam)
        R3  = sinh(lam)*cos(lam) + cosh(lam)*sin(lam)
        R4  = R2 * (k/2) * cosh(lam) * cos(lam)
        R5  = R3 * (k/2) * sinh(lam) * sin(lam)
        lam = (c/t) * ( 6 * E_a * t / (E_s * t_a) ) ** 0.25

    with c = half overlap length, t = adherend thickness, t_a = adhesive
    thickness, E_s = adherend modulus, k = GR bending moment factor (Eq. 3.57)
    evaluated at its low-load limit k = 1.0.

    The profile peaks at the overlap end (x = 0) where cosh*cos = 1 and
    sinh*sin = 0, so sigma_max = sigma_0 * R4 and failure gives

        P_cr = sigma_c * R1 * b * c^2 / ( t * R4 )
    """
    theta = math.radians(peel_angle) if peel_angle > math.pi else peel_angle

    if force_type == 'compressive':
        val = _compressive_limit(length, width, thickness, youngs_modulus)
        return (val, {}) if return_params else val
    if force_type != 'tensile':
        raise ValueError("force_type must be 'tensile' or 'compressive'")

    t_a = thickness
    t = adherend_thickness if adherend_thickness else thickness
    E_s = adherend_modulus if adherend_modulus else youngs_modulus
    c = length / 2.0

    if t_a <= 0 or t <= 0 or E_s <= 0 or c <= 0:
        return (None, {}) if return_params else None

    sigma_c = critical_peel_stress(surface_energy)

    lam = (c / t) * ((6.0 * youngs_modulus * t) / (E_s * t_a)) ** 0.25

    # cosh/sinh overflow for large lam. Beyond ~350 the hyperbolic terms are
    # fully saturated and the resulting P_cr is insensitive to lam, so clamp.
    LAM_MAX = 350.0
    lam_c = min(lam, LAM_MAX)

    sinh_l, cosh_l = math.sinh(lam_c), math.cosh(lam_c)
    sin_l, cos_l = math.sin(lam_c), math.cos(lam_c)

    R1 = (math.sinh(min(2.0 * lam_c, 2.0 * LAM_MAX)) + math.sin(2.0 * lam_c)) / 2.0
    R2 = sinh_l * cos_l - cosh_l * sin_l
    R3 = sinh_l * cos_l + cosh_l * sin_l

    k = 1.0
    R4 = R2 * (k / 2.0) * cosh_l * cos_l
    R5 = R3 * (k / 2.0) * sinh_l * sin_l

    if R4 == 0 or not math.isfinite(R4) or not math.isfinite(R1):
        return (None, {}) if return_params else None

    P_cr = sigma_c * R1 * width * (c ** 2) / (t * abs(R4))

    if not math.isfinite(P_cr) or P_cr <= 0:
        return (None, {}) if return_params else None
    if not return_params:
        return P_cr

    sigma_0 = P_cr * t / (R1 * width * c ** 2)
    return P_cr, {'sigma_0': sigma_0, 'lambda': lam, 'lambda_used': lam_c,
                  'R1': R1, 'R2': R2, 'R3': R3, 'R4': R4, 'R5': R5,
                  'k': k, 'sigma_c': sigma_c}


# --- Dispatcher (backwards compatible) ---
PEEL_MODEL = 'kaelble'   # set to 'goland_reissner' to switch globally


def max_peel_force(length, width, thickness, peel_angle, youngs_modulus,
                   surface_energy, force_type='tensile', model=None):
    """Backwards-compatible wrapper dispatching to the selected model."""
    chosen = (model or PEEL_MODEL).lower()
    if chosen in ('kaelble', 'k'):
        return max_peel_force_kaelble(length, width, thickness, peel_angle,
                                      youngs_modulus, surface_energy, force_type)
    if chosen in ('goland_reissner', 'gr', 'goland-reissner'):
        return max_peel_force_goland_reissner(length, width, thickness, peel_angle,
                                              youngs_modulus, surface_energy,
                                              force_type)
    raise ValueError("model must be 'kaelble' or 'goland_reissner'")
