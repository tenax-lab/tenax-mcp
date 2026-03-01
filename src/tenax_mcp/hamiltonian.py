"""AutoMPO wrapper logic for the MCP server."""

from __future__ import annotations

import numpy as np

from tenax import AutoMPO, spin_half_ops, spin_one_ops


def list_operators(spin: str = "half") -> dict:
    """Return available built-in operators for the given spin type."""
    if spin == "half":
        ops = spin_half_ops()
    elif spin == "one":
        ops = spin_one_ops()
    else:
        return {"error": f"Unknown spin type '{spin}'. Use 'half' or 'one'."}

    result = {}
    for name, mat in ops.items():
        result[name] = {
            "shape": list(mat.shape),
            "matrix": mat.tolist(),
        }
    return {"spin": spin, "operators": result}


def _parse_terms(terms: list[dict]) -> list[tuple]:
    """Convert JSON term dicts to AutoMPO tuple format.

    Each term dict has:
      - "coeff": float
      - "ops": list of {"op": str, "site": int}
    """
    parsed = []
    for t in terms:
        coeff = t["coeff"]
        parts: list = [coeff]
        for op_spec in t["ops"]:
            parts.append(op_spec["op"])
            parts.append(op_spec["site"])
        parsed.append(tuple(parts))
    return parsed


def build_hamiltonian(
    L: int,
    terms: list[dict],
    d: int = 2,
) -> dict:
    """Build MPO from operator terms and return diagnostic info."""
    auto = AutoMPO(L=L, d=d)
    for t in terms:
        coeff = t["coeff"]
        args = []
        for op_spec in t["ops"]:
            args.append(op_spec["op"])
            args.append(op_spec["site"])
        auto.add_term(coeff, *args)

    bond_dims = auto.bond_dims()
    return {
        "L": L,
        "d": d,
        "n_terms": auto.n_terms(),
        "bond_dimensions": bond_dims,
        "total_hilbert_dim": d**L,
    }


def build_mpo_from_terms(
    L: int,
    terms: list[dict],
    d: int = 2,
    compress: bool = False,
    dtype=None,
):
    """Build and return the actual MPO TensorNetwork (for use by run_dmrg)."""
    import jax.numpy as jnp

    if dtype is None:
        dtype = jnp.float64
    auto = AutoMPO(L=L, d=d)
    for t in terms:
        coeff = t["coeff"]
        args = []
        for op_spec in t["ops"]:
            args.append(op_spec["op"])
            args.append(op_spec["site"])
        auto.add_term(coeff, *args)
    return auto.to_mpo(compress=compress, dtype=dtype)
