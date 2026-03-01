"""DMRG and TRG runner logic for the MCP server."""

from __future__ import annotations

import jax.numpy as jnp

from tenax import (
    DMRGConfig,
    TRGConfig,
    build_random_mps,
    compute_ising_tensor,
    dmrg,
    ising_free_energy_exact,
    trg,
)

from tenax_mcp.hamiltonian import build_mpo_from_terms


def run_dmrg(
    L: int,
    terms: list[dict],
    d: int = 2,
    max_bond_dim: int = 100,
    num_sweeps: int = 10,
    convergence_tol: float = 1e-10,
    two_site: bool = True,
    noise: float = 0.0,
    initial_bond_dim: int = 4,
    verbose: bool = False,
) -> dict:
    """Run DMRG ground state search and return results."""
    mpo = build_mpo_from_terms(L, terms, d=d)
    mps = build_random_mps(L, physical_dim=d, bond_dim=initial_bond_dim)

    config = DMRGConfig(
        max_bond_dim=max_bond_dim,
        num_sweeps=num_sweeps,
        convergence_tol=convergence_tol,
        two_site=two_site,
        noise=noise,
        verbose=verbose,
    )

    result = dmrg(mpo, mps, config)

    return {
        "energy": float(result.energy),
        "energies_per_sweep": [float(e) for e in result.energies_per_sweep],
        "converged": bool(result.converged),
        "num_sweeps_run": len(result.energies_per_sweep),
        "truncation_error_final": float(result.truncation_errors[-1])
        if result.truncation_errors
        else None,
        "L": L,
        "max_bond_dim": max_bond_dim,
    }


def run_trg(
    model: str = "ising",
    beta: float | None = None,
    temperature: float | None = None,
    J: float = 1.0,
    max_bond_dim: int = 16,
    num_steps: int = 20,
) -> dict:
    """Run TRG on a 2D classical model and return results."""
    if model != "ising":
        return {"error": f"Unknown model '{model}'. Currently only 'ising' is supported."}

    if beta is None and temperature is None:
        return {"error": "Provide either 'beta' (inverse temperature) or 'temperature'."}
    if beta is None:
        beta = 1.0 / temperature

    tensor = compute_ising_tensor(beta=beta, J=J)
    config = TRGConfig(max_bond_dim=max_bond_dim, num_steps=num_steps)

    log_Z_per_site = trg(tensor, config)
    free_energy = -float(log_Z_per_site) / beta

    exact = ising_free_energy_exact(beta=beta, J=J)
    rel_error = abs(free_energy - exact) / abs(exact) if exact != 0 else 0.0

    return {
        "free_energy_per_site": free_energy,
        "exact_free_energy_per_site": exact,
        "relative_error": rel_error,
        "log_Z_per_site": float(log_Z_per_site),
        "beta": beta,
        "temperature": 1.0 / beta,
        "max_bond_dim": max_bond_dim,
        "num_steps": num_steps,
    }
