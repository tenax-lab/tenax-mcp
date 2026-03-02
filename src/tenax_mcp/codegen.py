"""Code generation and .net export logic."""

from __future__ import annotations

TEMPLATES = {
    "dmrg": '''\
"""DMRG ground state calculation for {model_name}."""

import tenax

# Build Hamiltonian
{hamiltonian_code}

# Build initial MPS
mps = tenax.build_random_mps(L={L}, physical_dim={d}, bond_dim={initial_bond_dim})

# DMRG configuration
config = tenax.DMRGConfig(
    max_bond_dim={max_bond_dim},
    num_sweeps={num_sweeps},
    two_site=True,
)

# Run DMRG
result = tenax.dmrg(mpo, mps, config)
print(f"Ground state energy: {{result.energy:.10f}}")
print(f"Converged: {{result.converged}}")
print(f"Energies per sweep: {{result.energies_per_sweep}}")
''',
    "trg": '''\
"""TRG calculation for the 2D Ising model."""

import tenax

# Build initial tensor
tensor = tenax.compute_ising_tensor(beta={beta})

# TRG configuration
config = tenax.TRGConfig(
    max_bond_dim={max_bond_dim},
    num_steps={num_steps},
)

# Run TRG
log_Z_per_site = tenax.trg(tensor, config)
free_energy = -log_Z_per_site / {beta}

# Compare with exact solution
exact = tenax.ising_free_energy_exact(beta={beta})
print(f"TRG free energy per site:   {{free_energy:.10f}}")
print(f"Exact free energy per site: {{exact:.10f}}")
print(f"Relative error: {{abs(free_energy - exact) / abs(exact):.2e}}")
''',
    "idmrg": '''\
"""iDMRG calculation for the infinite {model_name}."""

import tenax

# Build bulk MPO
{bulk_mpo_code}

# iDMRG configuration
config = tenax.iDMRGConfig(
    max_bond_dim={max_bond_dim},
    max_iterations={max_iterations},
    convergence_tol={convergence_tol},
)

# Run iDMRG
result = tenax.idmrg(bulk_mpo, config)
print(f"Energy per site: {{result.energy_per_site:.10f}}")
print(f"Converged: {{result.converged}}")
''',
    "ipeps": '''\
"""iPEPS ground state optimization for the 2D {model_name}."""

import jax.numpy as jnp
import tenax

# Build 2-site Hamiltonian gate
{gate_code}

# iPEPS configuration
config = tenax.iPEPSConfig(
    max_bond_dim={D},
    ctm=tenax.CTMConfig(chi={chi}{ctm_extra}),
    gs_optimizer="adam",
    gs_learning_rate={lr},
    gs_num_steps={num_steps},{ipeps_extra}
)

# Run iPEPS optimization
A_opt, env, E_gs = tenax.optimize_gs_ad(gate, A_init=None, config=config)
print(f"Ground state energy per site: {{E_gs:.10f}}")
''',
    "ipeps_2site": '''\
"""iPEPS 2-site AD optimization for the 2D {model_name} (antiferromagnets)."""

import jax.numpy as jnp
import tenax

# Build 2-site Hamiltonian gate
{gate_code}

# 2-site iPEPS configuration (checkerboard unit cell)
config = tenax.iPEPSConfig(
    max_bond_dim={D},
    ctm=tenax.CTMConfig(chi={chi}{ctm_extra}),
    gs_optimizer="adam",
    gs_learning_rate={lr},
    gs_num_steps={num_steps},
    unit_cell="2site",
    su_init=True,
    num_imaginary_steps=200,
    dt=0.01,
)

# Run 2-site AD optimization
(A_opt, B_opt), (env_A, env_B), E_gs = tenax.optimize_gs_ad(gate, None, config)
print(f"Ground state energy per site: {{E_gs:.10f}}")
''',
    "ipeps_split": '''\
"""iPEPS with Split-CTMRG for the 2D {model_name}."""

import jax.numpy as jnp
import tenax

# Build 2-site Hamiltonian gate
{gate_code}

# AD optimization
config = tenax.iPEPSConfig(
    max_bond_dim={D},
    ctm=tenax.CTMConfig(chi={chi}),
    gs_optimizer="adam",
    gs_learning_rate={lr},
    gs_num_steps={num_steps},
    su_init=True,
    num_imaginary_steps=200,
    dt=0.01,
)
A_opt, env, E_gs = tenax.optimize_gs_ad(gate, A_init=None, config=config)
print(f"AD energy: {{E_gs:.10f}}")

# Evaluate with Split-CTMRG (reduced cost: O(chi^3 D^3) vs O(chi^3 D^6))
split_config = tenax.CTMConfig(chi={chi}, max_iter=100, chi_I={chi_I})
split_env = tenax.ctm_split(A_opt, split_config)
E_split = tenax.compute_energy_split_ctm(A_opt, split_env, gate, d={d})
print(f"Split-CTM energy: {{E_split:.10f}}")
''',
    "contraction": '''\
"""Custom tensor network contraction."""

import jax.numpy as jnp
import tenax

{tensor_code}

# Contract
result = tenax.contract({contract_args})
print(f"Result shape: {{result.shape}}")
print(f"Result labels: {{result.labels}}")
''',
}


def _heisenberg_autompo_code(L: int, Jz: float = 1.0, Jxy: float = 1.0, hz: float = 0.0) -> str:
    lines = ["auto = tenax.AutoMPO(L={}, d=2)".format(L)]
    lines.append("for i in range({} - 1):".format(L))
    if Jz != 0:
        lines.append(f"    auto.add_term({Jz}, 'Sz', i, 'Sz', i + 1)")
    if Jxy != 0:
        lines.append(f"    auto.add_term({Jxy / 2}, 'Sp', i, 'Sm', i + 1)")
        lines.append(f"    auto.add_term({Jxy / 2}, 'Sm', i, 'Sp', i + 1)")
    if hz != 0:
        lines.append(f"for i in range({L}):")
        lines.append(f"    auto.add_term({hz}, 'Sz', i)")
    lines.append("mpo = auto.to_mpo()")
    return "\n".join(lines)


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
    lr: float = 1e-3,
    max_iterations: int = 200,
    convergence_tol: float = 1e-8,
    unit_cell: str = "1x1",
    projector_method: str = "eigh",
    qr_warmup_steps: int = 3,
    chi_I: int | None = None,
    su_init: bool = False,
) -> dict:
    """Generate runnable Tenax code from a high-level description."""
    algorithm = algorithm.lower()

    if algorithm == "dmrg":
        code = TEMPLATES["dmrg"].format(
            model_name="Heisenberg chain",
            hamiltonian_code=_heisenberg_autompo_code(L, Jz, Jxy, hz),
            L=L,
            d=d,
            initial_bond_dim=min(4, max_bond_dim),
            max_bond_dim=max_bond_dim,
            num_sweeps=num_sweeps,
        )
    elif algorithm == "trg":
        code = TEMPLATES["trg"].format(
            beta=beta,
            max_bond_dim=max_bond_dim,
            num_steps=num_steps,
        )
    elif algorithm == "idmrg":
        bulk_code = "bulk_mpo = tenax.build_bulk_mpo_heisenberg(Jz={}, Jxy={}, hz={})".format(
            Jz, Jxy, hz
        )
        code = TEMPLATES["idmrg"].format(
            model_name="Heisenberg chain",
            bulk_mpo_code=bulk_code,
            max_bond_dim=max_bond_dim,
            max_iterations=max_iterations,
            convergence_tol=convergence_tol,
        )
    elif algorithm in ("ipeps", "ipeps_2site", "ipeps_split"):
        gate_code = (
            "# Heisenberg gate: H = Jz*Sz⊗Sz + Jxy/2*(Sp⊗Sm + Sm⊗Sp)\n"
            "ops = tenax.spin_half_ops()\n"
            "Sz, Sp, Sm = ops['Sz'], ops['Sp'], ops['Sm']\n"
            "Id = ops['Id']\n"
            f"gate = ({Jz} * jnp.kron(Sz, Sz)\n"
            f"        + {Jxy / 2} * (jnp.kron(Sp, Sm) + jnp.kron(Sm, Sp)))\n"
            "gate = gate.reshape(2, 2, 2, 2)"
        )

        # Build CTM extra config (QR projectors)
        ctm_extra_parts = []
        if projector_method == "qr":
            ctm_extra_parts.append(f', projector_method="qr"')
            ctm_extra_parts.append(f", qr_warmup_steps={qr_warmup_steps}")
        ctm_extra = "".join(ctm_extra_parts)

        # Build iPEPS extra config (su_init)
        ipeps_extra = ""
        if su_init and algorithm == "ipeps":
            ipeps_extra = (
                "\n    su_init=True,"
                "\n    num_imaginary_steps=200,"
                "\n    dt=0.01,"
            )

        # Select template based on algorithm variant
        if algorithm == "ipeps_2site" or (algorithm == "ipeps" and unit_cell == "2site"):
            template_key = "ipeps_2site"
        elif algorithm == "ipeps_split":
            template_key = "ipeps_split"
        else:
            template_key = "ipeps"

        fmt_kwargs = dict(
            model_name="Heisenberg model",
            gate_code=gate_code,
            D=D,
            chi=chi,
            lr=lr,
            num_steps=num_steps,
            ctm_extra=ctm_extra,
            ipeps_extra=ipeps_extra,
            d=d,
            chi_I=chi_I if chi_I is not None else chi * D,
        )
        code = TEMPLATES[template_key].format(**fmt_kwargs)
    else:
        return {
            "error": f"Unknown algorithm '{algorithm}'. "
            "Supported: dmrg, trg, idmrg, ipeps, ipeps_2site, ipeps_split."
        }

    return {
        "code": code,
        "algorithm": algorithm,
        "description": description,
    }


def export_netfile(
    tensors: list[dict],
    output_labels: list[str] | None = None,
    contraction_order: str | None = None,
) -> dict:
    """Generate .net file content from tensor descriptions.

    Each tensor dict has:
      - "name": tensor name
      - "labels": list of leg labels
    """
    lines = []
    for t in tensors:
        name = t["name"]
        labels = ", ".join(t["labels"])
        lines.append(f"{name}: {labels}")

    if output_labels:
        lines.append(f"TOUT: {', '.join(output_labels)}")

    if contraction_order:
        lines.append(f"ORDER: {contraction_order}")

    content = "\n".join(lines) + "\n"
    return {
        "netfile_content": content,
        "num_tensors": len(tensors),
        "output_labels": output_labels,
    }
