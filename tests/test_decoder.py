import torch

from gradcell.design import DesignSpace
from gradcell.design.capacity_balance import chen2020_scaled_capacity_ah


def test_decoder_is_hard_feasible():
    torch.manual_seed(0)
    decoder = DesignSpace()
    latent = torch.randn(1000, decoder.latent_dim, dtype=torch.float64)
    design = decoder(latent)
    inactive_p = 1.0 - design.eps_p - design.phi_p
    inactive_n = 1.0 - design.eps_n - design.phi_n
    assert torch.all(inactive_p >= decoder.inactive_p_min - 1e-12)
    assert torch.all(inactive_n >= decoder.inactive_n_min - 1e-12)
    assert torch.all(design.np_ratio >= decoder.np_bounds[0])
    assert torch.all(design.np_ratio <= decoder.np_bounds[1])


def test_np_ratio_is_exact():
    decoder = DesignSpace()
    design = decoder(torch.randn(64, decoder.latent_dim, dtype=torch.float64))
    c = decoder.capacity_constants
    q_p = c.positive_thickness_m * design.phi_p * c.positive_cmax_mol_m3 * c.positive_stoich_window
    q_n = c.negative_thickness_m * design.phi_n * c.negative_cmax_mol_m3 * c.negative_stoich_window
    torch.testing.assert_close(q_n / q_p, design.np_ratio)


def test_chen2020_scaled_capacity_matches_nominal_calibration():
    phi_p = torch.tensor([0.665], dtype=torch.float64)
    torch.testing.assert_close(
        chen2020_scaled_capacity_ah(phi_p),
        torch.tensor([5.0], dtype=torch.float64),
    )


def test_chen2020_scaled_capacity_is_more_conservative_in_current_design_range():
    latent = torch.zeros(8, DesignSpace.latent_dim, dtype=torch.float64)
    theoretical = DesignSpace(capacity_formula="electrode_theoretical")(latent)
    scaled = DesignSpace(capacity_formula="chen2020_scaled")(latent)
    assert torch.all(scaled.nominal_capacity_ah < theoretical.nominal_capacity_ah)


def test_chen2020_diffusivities_are_fixed_material_properties():
    decoder = DesignSpace()
    design = decoder(torch.randn(64, decoder.latent_dim, dtype=torch.float64))
    torch.testing.assert_close(
        design.diffusivity_p_multiplier,
        torch.ones_like(design.diffusivity_p_multiplier),
    )
    torch.testing.assert_close(
        design.diffusivity_n_multiplier,
        torch.ones_like(design.diffusivity_n_multiplier),
    )
