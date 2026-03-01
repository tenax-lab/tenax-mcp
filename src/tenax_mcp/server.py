"""Tenax MCP Server — exposes tensor network tools via FastMCP."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Tenax", dependencies=["tenax-tn"])


# --- Core tools ---


@mcp.tool()
def list_operators(spin: str = "half") -> dict:
    """List available built-in spin operators and their matrix representations.

    Args:
        spin: Spin type — "half" (d=2, returns Sz/Sp/Sm/Id) or
              "one" (d=3, returns Sz/Sp/Sm/Id with basis |+1⟩,|0⟩,|-1⟩).
    """
    from tenax_mcp.hamiltonian import list_operators as _list_operators

    return _list_operators(spin)


@mcp.tool()
def build_hamiltonian(L: int, terms: list[dict], d: int = 2) -> dict:
    """Build an MPO Hamiltonian from operator terms and return diagnostic info.

    Does NOT return the tensor data — use run_dmrg to actually run a calculation.

    Args:
        L: Number of sites.
        terms: List of term dicts, each with:
            - "coeff": float coefficient
            - "ops": list of {"op": str, "site": int} pairs
            Example: [{"coeff": 1.0, "ops": [{"op": "Sz", "site": 0}, {"op": "Sz", "site": 1}]}]
        d: Local Hilbert space dimension (default 2 for spin-1/2).

    Returns:
        Bond dimensions, number of terms, total Hilbert space dimension.
    """
    from tenax_mcp.hamiltonian import build_hamiltonian as _build_hamiltonian

    return _build_hamiltonian(L, terms, d)


@mcp.tool()
def optimize_contraction(
    tensors: list[dict],
    output_labels: list[str] | None = None,
) -> dict:
    """Find optimal contraction path and FLOP cost for a tensor network.

    Args:
        tensors: List of tensor dicts, each with:
            - "labels": list of string leg labels
            - "dimensions": list of int dimensions per leg
            Labels shared between tensors are contracted.
        output_labels: Explicit output leg ordering (default: all free legs).

    Returns:
        Optimal contraction order, FLOP count, subscript string, speedup over naive.
    """
    from tenax_mcp.contraction import optimize_contraction as _optimize

    return _optimize(tensors, output_labels)


@mcp.tool()
def validate_network(tensors: list[dict]) -> dict:
    """Check tensor network validity — dimension matching, charge consistency, flow directions.

    Args:
        tensors: List of tensor dicts, each with:
            - "name": tensor name
            - "labels": list of string leg labels
            - "dimensions": list of int dimensions
            - "charges" (optional): list of charge arrays per leg
            - "flow" (optional): list of "in"/"out" per leg

    Returns:
        Validation result with valid/invalid status and list of issues.
    """
    from tenax_mcp.contraction import validate_network as _validate

    return _validate(tensors)


# --- Algorithm tools ---


@mcp.tool()
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
) -> dict:
    """Run DMRG ground state search for a 1D quantum Hamiltonian.

    Args:
        L: Number of sites.
        terms: Hamiltonian terms (same format as build_hamiltonian).
        d: Local Hilbert space dimension (default 2).
        max_bond_dim: Maximum MPS bond dimension (default 100).
        num_sweeps: Number of DMRG sweeps (default 10).
        convergence_tol: Energy convergence tolerance (default 1e-10).
        two_site: Use 2-site DMRG (default True).
        noise: Perturbative noise strength (default 0.0).
        initial_bond_dim: Initial random MPS bond dimension (default 4).

    Returns:
        Ground state energy, energy per sweep, convergence status, truncation errors.
    """
    from tenax_mcp.solvers import run_dmrg as _run_dmrg

    return _run_dmrg(
        L=L,
        terms=terms,
        d=d,
        max_bond_dim=max_bond_dim,
        num_sweeps=num_sweeps,
        convergence_tol=convergence_tol,
        two_site=two_site,
        noise=noise,
        initial_bond_dim=initial_bond_dim,
    )


@mcp.tool()
def run_trg(
    model: str = "ising",
    beta: float | None = None,
    temperature: float | None = None,
    J: float = 1.0,
    max_bond_dim: int = 16,
    num_steps: int = 20,
) -> dict:
    """Run TRG on a 2D classical model (currently: 2D Ising).

    Provide either beta (inverse temperature) or temperature.

    Args:
        model: Model type (currently only "ising").
        beta: Inverse temperature (1/T). Provide this OR temperature.
        temperature: Temperature T. Provide this OR beta.
        J: Coupling constant (default 1.0).
        max_bond_dim: Maximum TRG bond dimension (default 16).
        num_steps: Number of coarse-graining steps (default 20).

    Returns:
        Free energy per site, exact solution, relative error, log(Z)/N.
    """
    from tenax_mcp.solvers import run_trg as _run_trg

    return _run_trg(
        model=model,
        beta=beta,
        temperature=temperature,
        J=J,
        max_bond_dim=max_bond_dim,
        num_steps=num_steps,
    )


# --- Code generation tools ---


@mcp.tool()
def generate_code(
    description: str,
    algorithm: str = "dmrg",
    L: int = 20,
    d: int = 2,
    max_bond_dim: int = 100,
    num_sweeps: int = 10,
    beta: float = 0.4,
    num_steps: int = 20,
    Jz: float = 1.0,
    Jxy: float = 1.0,
    hz: float = 0.0,
    D: int = 2,
    chi: int = 20,
) -> dict:
    """Generate complete, runnable Tenax Python code from a high-level description.

    Args:
        description: What the code should do (e.g., "Heisenberg chain DMRG").
        algorithm: One of "dmrg", "trg", "idmrg", "ipeps".
        L: Number of sites (for DMRG).
        d: Local Hilbert space dimension.
        max_bond_dim: Bond dimension.
        num_sweeps: DMRG sweeps.
        beta: Inverse temperature (for TRG).
        num_steps: Coarse-graining steps (TRG) or optimization steps (iPEPS).
        Jz: Ising coupling.
        Jxy: XY coupling.
        hz: External field.
        D: iPEPS bond dimension.
        chi: CTM environment bond dimension.

    Returns:
        Complete Python code string ready to run.
    """
    from tenax_mcp.codegen import generate_code as _generate_code

    return _generate_code(
        description=description,
        algorithm=algorithm,
        L=L,
        d=d,
        max_bond_dim=max_bond_dim,
        num_sweeps=num_sweeps,
        beta=beta,
        num_steps=num_steps,
        Jz=Jz,
        Jxy=Jxy,
        hz=hz,
        D=D,
        chi=chi,
    )


@mcp.tool()
def export_netfile(
    tensors: list[dict],
    output_labels: list[str] | None = None,
    contraction_order: str | None = None,
) -> dict:
    """Convert a tensor network description to .net file format (Cytnx-compatible).

    Args:
        tensors: List of tensor dicts with "name" and "labels" keys.
            Example: [{"name": "A", "labels": ["i", "j"]}, {"name": "B", "labels": ["j", "k"]}]
        output_labels: Output leg labels for TOUT line (optional).
        contraction_order: Explicit ORDER string (optional).

    Returns:
        .net file content string.
    """
    from tenax_mcp.codegen import export_netfile as _export_netfile

    return _export_netfile(tensors, output_labels, contraction_order)


def main():
    """Run the Tenax MCP server."""
    mcp.run(transport="stdio")
