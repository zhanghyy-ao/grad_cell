import torch

from gradcell.design import DesignSpace


def test_decoder_is_hard_feasible():
    torch.manual_seed(0)
    decoder = DesignSpace()
    latent = torch.randn(1000, 7, dtype=torch.float64)
    design = decoder(latent)
    inactive_p = 1.0 - design.eps_p - design.phi_p
    inactive_n = 1.0 - design.eps_n - design.phi_n
    assert torch.all(inactive_p >= decoder.inactive_p_min - 1e-12)
    assert torch.all(inactive_n >= decoder.inactive_n_min - 1e-12)
    assert torch.all(design.np_ratio >= decoder.np_bounds[0])
    assert torch.all(design.np_ratio <= decoder.np_bounds[1])


def test_np_ratio_is_exact():
    decoder = DesignSpace()
    design = decoder(torch.randn(64, 7, dtype=torch.float64))
    c = decoder.capacity_constants
    q_p = c.positive_thickness_m * design.phi_p * c.positive_cmax_mol_m3 * c.positive_stoich_window
    q_n = c.negative_thickness_m * design.phi_n * c.negative_cmax_mol_m3 * c.negative_stoich_window
    torch.testing.assert_close(q_n / q_p, design.np_ratio)

