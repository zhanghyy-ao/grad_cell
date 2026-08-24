from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .capacity_balance import CapacityConstants, negative_active_fraction, nominal_capacity_ah
from .mass_model import MassConstants, stack_mass_kg


@dataclass(frozen=True)
class CellDesign:
    eps_p: torch.Tensor
    eps_n: torch.Tensor
    eps_s: torch.Tensor
    phi_p: torch.Tensor
    phi_n: torch.Tensor
    np_ratio: torch.Tensor
    diffusivity_p_multiplier: torch.Tensor
    diffusivity_n_multiplier: torch.Tensor
    nominal_capacity_ah: torch.Tensor
    stack_mass_kg: torch.Tensor

    def physics_tensor(self, c_rate: float) -> torch.Tensor:
        current_a = self.nominal_capacity_ah * c_rate
        return torch.stack(
            [
                self.eps_p,
                self.eps_n,
                self.eps_s,
                self.phi_p,
                self.phi_n,
                self.diffusivity_p_multiplier,
                self.diffusivity_n_multiplier,
                current_a,
            ],
            dim=-1,
        )


class DesignSpace(nn.Module):
    """Map seven unconstrained variables to a hard-feasible Chen2020 design."""

    latent_dim = 7

    def __init__(
        self,
        eps_p_bounds=(0.20, 0.42),
        eps_n_bounds=(0.20, 0.42),
        eps_s_bounds=(0.35, 0.60),
        np_bounds=(1.02, 1.25),
        phi_p_min=0.20,
        inactive_p_min=0.03,
        inactive_n_min=0.03,
        diffusivity_multiplier_bounds=(0.5, 2.0),
    ) -> None:
        super().__init__()
        self.eps_p_bounds = eps_p_bounds
        self.eps_n_bounds = eps_n_bounds
        self.eps_s_bounds = eps_s_bounds
        self.np_bounds = np_bounds
        self.phi_p_min = phi_p_min
        self.inactive_p_min = inactive_p_min
        self.inactive_n_min = inactive_n_min
        self.diffusivity_multiplier_bounds = diffusivity_multiplier_bounds
        self.capacity_constants = CapacityConstants()
        self.mass_constants = MassConstants()

    @staticmethod
    def _bounded(value: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
        low, high = bounds
        return low + (high - low) * torch.sigmoid(value)

    @staticmethod
    def _log_bounded(value: torch.Tensor, bounds: tuple[float, float]):
        low, high = bounds
        log_multiplier = torch.log(
            torch.as_tensor(low, dtype=value.dtype, device=value.device)
        ) + torch.sigmoid(value) * torch.log(
            torch.as_tensor(high / low, dtype=value.dtype, device=value.device)
        )
        return torch.exp(log_multiplier)

    def forward(self, latent: torch.Tensor) -> CellDesign:
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected latent dimension {self.latent_dim}, got {latent.shape[-1]}")
        eps_p = self._bounded(latent[..., 0], self.eps_p_bounds)
        eps_n = self._bounded(latent[..., 1], self.eps_n_bounds)
        eps_s = self._bounded(latent[..., 2], self.eps_s_bounds)
        np_ratio = self._bounded(latent[..., 4], self.np_bounds)

        numerator = (
            self.capacity_constants.positive_thickness_m
            * self.capacity_constants.positive_cmax_mol_m3
            * self.capacity_constants.positive_stoich_window
        )
        denominator = (
            self.capacity_constants.negative_thickness_m
            * self.capacity_constants.negative_cmax_mol_m3
            * self.capacity_constants.negative_stoich_window
        )
        kappa = np_ratio * numerator / denominator
        max_from_positive = 1.0 - eps_p - self.inactive_p_min
        max_from_negative = (1.0 - eps_n - self.inactive_n_min) / kappa
        phi_p_max = torch.minimum(max_from_positive, max_from_negative)
        if torch.any(phi_p_max <= self.phi_p_min):
            raise RuntimeError("Configured design bounds contain no feasible active-fraction interval")
        phi_p = self.phi_p_min + torch.sigmoid(latent[..., 3]) * (
            phi_p_max - self.phi_p_min
        )
        phi_n = negative_active_fraction(phi_p, np_ratio, self.capacity_constants)
        diffusivity_p_multiplier = self._log_bounded(
            latent[..., 5], self.diffusivity_multiplier_bounds
        )
        diffusivity_n_multiplier = self._log_bounded(
            latent[..., 6], self.diffusivity_multiplier_bounds
        )
        capacity = nominal_capacity_ah(phi_p, self.capacity_constants)
        mass = stack_mass_kg(
            eps_p, eps_n, eps_s, phi_p, phi_n, self.mass_constants
        )
        return CellDesign(
            eps_p=eps_p,
            eps_n=eps_n,
            eps_s=eps_s,
            phi_p=phi_p,
            phi_n=phi_n,
            np_ratio=np_ratio,
            diffusivity_p_multiplier=diffusivity_p_multiplier,
            diffusivity_n_multiplier=diffusivity_n_multiplier,
            nominal_capacity_ah=capacity,
            stack_mass_kg=mass,
        )

    decode = forward

    def nominal_latent(self, batch_shape=(), *, dtype=torch.float64, device=None) -> torch.Tensor:
        return torch.zeros(*batch_shape, self.latent_dim, dtype=dtype, device=device)
