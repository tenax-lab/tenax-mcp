"""Unit tests for Tenax MCP tool logic."""

import pytest

from tenax_mcp.hamiltonian import build_hamiltonian, list_operators
from tenax_mcp.solvers import run_dmrg, run_hotrg, run_trg
from tenax_mcp.contraction import optimize_contraction, validate_network
from tenax_mcp.codegen import generate_code, export_netfile


# --- list_operators ---


def test_list_operators_half():
    result = list_operators("half")
    assert "operators" in result
    ops = result["operators"]
    assert set(ops.keys()) == {"Sz", "Sp", "Sm", "Id"}
    for name, info in ops.items():
        assert info["shape"] == [2, 2]


def test_list_operators_one():
    result = list_operators("one")
    assert "operators" in result
    ops = result["operators"]
    assert "Sz" in ops
    assert ops["Sz"]["shape"] == [3, 3]


def test_list_operators_invalid():
    result = list_operators("two")
    assert "error" in result


# --- build_hamiltonian ---


def test_build_hamiltonian_heisenberg():
    terms = []
    L = 4
    for i in range(L - 1):
        terms.append({"coeff": 1.0, "ops": [{"op": "Sz", "site": i}, {"op": "Sz", "site": i + 1}]})
        terms.append({"coeff": 0.5, "ops": [{"op": "Sp", "site": i}, {"op": "Sm", "site": i + 1}]})
        terms.append({"coeff": 0.5, "ops": [{"op": "Sm", "site": i}, {"op": "Sp", "site": i + 1}]})
    result = build_hamiltonian(L=L, terms=terms)
    assert result["L"] == L
    assert result["d"] == 2
    assert result["n_terms"] == 3 * (L - 1)
    assert result["total_hilbert_dim"] == 2**L
    assert len(result["bond_dimensions"]) == L - 1


# --- optimize_contraction ---


def test_optimize_contraction_matrix_chain():
    tensors = [
        {"labels": ["i", "j"], "dimensions": [10, 20]},
        {"labels": ["j", "k"], "dimensions": [20, 30]},
        {"labels": ["k", "l"], "dimensions": [30, 5]},
    ]
    result = optimize_contraction(tensors)
    assert "flops" in result
    assert "contraction_path" in result
    assert "subscripts" in result
    assert result["flops"] > 0


def test_optimize_contraction_with_output():
    tensors = [
        {"labels": ["i", "j"], "dimensions": [10, 20]},
        {"labels": ["j", "k"], "dimensions": [20, 30]},
    ]
    result = optimize_contraction(tensors, output_labels=["k", "i"])
    assert "subscripts" in result


# --- validate_network ---


def test_validate_network_valid():
    tensors = [
        {"name": "A", "labels": ["i", "bond"], "dimensions": [2, 10]},
        {"name": "B", "labels": ["bond", "j"], "dimensions": [10, 2]},
    ]
    result = validate_network(tensors)
    assert result["valid"] is True
    assert len(result["issues"]) == 0
    assert "bond" in result["shared_labels"]


def test_validate_network_dim_mismatch():
    tensors = [
        {"name": "A", "labels": ["i", "bond"], "dimensions": [2, 10]},
        {"name": "B", "labels": ["bond", "j"], "dimensions": [5, 2]},
    ]
    result = validate_network(tensors)
    assert result["valid"] is False
    assert any("mismatch" in issue.lower() for issue in result["issues"])


def test_validate_network_triple_label():
    tensors = [
        {"name": "A", "labels": ["x"], "dimensions": [5]},
        {"name": "B", "labels": ["x"], "dimensions": [5]},
        {"name": "C", "labels": ["x"], "dimensions": [5]},
    ]
    result = validate_network(tensors)
    assert result["valid"] is False


# --- generate_code ---


def test_generate_code_dmrg():
    result = generate_code("Heisenberg chain", algorithm="dmrg", L=10, max_bond_dim=50)
    assert "code" in result
    assert "import tenax" in result["code"]
    assert "dmrg" in result["code"].lower()


def test_generate_code_trg():
    result = generate_code("2D Ising", algorithm="trg", beta=0.44)
    assert "code" in result
    assert "trg" in result["code"].lower()


def test_generate_code_idmrg():
    result = generate_code("infinite Heisenberg", algorithm="idmrg")
    assert "code" in result
    assert "idmrg" in result["code"].lower()


def test_generate_code_ipeps():
    result = generate_code("2D Heisenberg", algorithm="ipeps")
    assert "code" in result
    assert "optimize_gs_ad" in result["code"]


def test_generate_code_ipeps_qr():
    result = generate_code("2D Heisenberg", algorithm="ipeps", projector_method="qr")
    assert "code" in result
    assert 'projector_method="qr"' in result["code"]
    assert "qr_warmup_steps" in result["code"]


def test_generate_code_ipeps_2site():
    result = generate_code("2D Heisenberg AFM", algorithm="ipeps_2site")
    assert "code" in result
    assert 'unit_cell="2site"' in result["code"]
    assert "A_opt, B_opt" in result["code"]


def test_generate_code_ipeps_unit_cell_2site():
    result = generate_code("2D Heisenberg AFM", algorithm="ipeps", unit_cell="2site")
    assert "code" in result
    assert 'unit_cell="2site"' in result["code"]


def test_generate_code_ipeps_split():
    result = generate_code("2D Heisenberg", algorithm="ipeps_split")
    assert "code" in result
    assert "ctm_split" in result["code"]
    assert "compute_energy_split_ctm" in result["code"]
    assert "chi_I" in result["code"]


def test_generate_code_ipeps_su_init():
    result = generate_code("2D Heisenberg", algorithm="ipeps", su_init=True)
    assert "code" in result
    assert "su_init=True" in result["code"]


def test_generate_code_split_ctm():
    result = generate_code("2D Heisenberg split CTM", algorithm="split_ctm", D=2, chi=10)
    assert "code" in result
    assert "ctm_split_tensor" in result["code"]
    assert "compute_energy_split_ctm_tensor" in result["code"]
    assert "DenseTensor" in result["code"]


def test_generate_code_hotrg():
    result = generate_code("2D Ising HOTRG", algorithm="hotrg", beta=0.44)
    assert "code" in result
    assert "hotrg" in result["code"].lower()
    assert "HOTRGConfig" in result["code"]


def test_generate_code_fpeps():
    result = generate_code("Fermionic iPEPS", algorithm="fpeps", D=2, chi=8)
    assert "code" in result
    assert "FPEPSConfig" in result["code"]
    assert "spinless_fermion_gate" in result["code"]
    assert "fpeps" in result["code"]


def test_generate_code_ctm_tensor():
    result = generate_code("Standard CTM", algorithm="ctm_tensor", D=2, chi=10)
    assert "code" in result
    assert "ctm_tensor" in result["code"]
    assert "compute_energy_ctm_tensor" in result["code"]
    assert "DenseTensor" in result["code"]


def test_generate_code_excitations():
    result = generate_code("Excitation spectra", algorithm="excitations", D=2, chi=10)
    assert "code" in result
    assert "compute_excitations" in result["code"]
    assert "ExcitationConfig" in result["code"]
    assert "make_momentum_path" in result["code"]


def test_generate_code_invalid():
    result = generate_code("something", algorithm="unknown")
    assert "error" in result


# --- export_netfile ---


def test_export_netfile():
    tensors = [
        {"name": "A", "labels": ["i", "j"]},
        {"name": "B", "labels": ["j", "k"]},
    ]
    result = export_netfile(tensors, output_labels=["i", "k"])
    content = result["netfile_content"]
    assert "A: i, j" in content
    assert "B: j, k" in content
    assert "TOUT: i, k" in content


def test_export_netfile_with_order():
    tensors = [
        {"name": "T1", "labels": ["a", "b"]},
        {"name": "T2", "labels": ["b", "c"]},
    ]
    result = export_netfile(tensors, contraction_order="(T1, T2)")
    assert "ORDER: (T1, T2)" in result["netfile_content"]


# --- Integration tests (require JAX/Tenax) ---


@pytest.mark.slow
def test_run_dmrg_heisenberg():
    L = 6
    terms = []
    for i in range(L - 1):
        terms.append({"coeff": 1.0, "ops": [{"op": "Sz", "site": i}, {"op": "Sz", "site": i + 1}]})
        terms.append({"coeff": 0.5, "ops": [{"op": "Sp", "site": i}, {"op": "Sm", "site": i + 1}]})
        terms.append({"coeff": 0.5, "ops": [{"op": "Sm", "site": i}, {"op": "Sp", "site": i + 1}]})
    result = run_dmrg(L=L, terms=terms, max_bond_dim=20, num_sweeps=6)
    assert "energy" in result
    assert result["energy"] < 0  # Ground state should be negative
    assert result["converged"] or result["num_sweeps_run"] == 6


@pytest.mark.slow
def test_run_trg_ising():
    result = run_trg(model="ising", beta=0.3, max_bond_dim=16, num_steps=20)
    assert "free_energy_per_site" in result
    assert "exact_free_energy_per_site" in result
    assert result["relative_error"] < 1e-2


def test_run_trg_invalid_model():
    result = run_trg(model="potts")
    assert "error" in result


def test_run_trg_no_temperature():
    result = run_trg(model="ising")
    assert "error" in result


@pytest.mark.slow
def test_run_hotrg_ising():
    result = run_hotrg(model="ising", beta=0.3, max_bond_dim=16, num_steps=10)
    assert "free_energy_per_site" in result
    assert "exact_free_energy_per_site" in result
    assert result["relative_error"] < 1e-2
    assert result["direction_order"] == "alternating"


def test_run_hotrg_invalid_model():
    result = run_hotrg(model="potts")
    assert "error" in result
