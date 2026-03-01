# Tenax MCP Server

An [MCP](https://modelcontextprotocol.io/) server that exposes [Tenax](https://github.com/tenax-lab/tenax) tensor network tools for use with Claude Code and Claude Desktop.

Build Hamiltonians, run DMRG/TRG, optimize contraction orders, generate code, and validate tensor networks — all through natural language.

## Tools

| Tool | Description |
|------|-------------|
| `list_operators` | List built-in spin operators (spin-1/2, spin-1) |
| `build_hamiltonian` | Build MPO from operator terms, return bond dimensions |
| `run_dmrg` | Run DMRG ground state search for 1D quantum Hamiltonians |
| `run_trg` | Run TRG on 2D classical models (Ising) |
| `optimize_contraction` | Find optimal contraction path and FLOP cost |
| `validate_network` | Check tensor network validity (dimensions, charges) |
| `generate_code` | Generate runnable Tenax Python code from descriptions |
| `export_netfile` | Convert network description to `.net` file format |

## Setup

### Claude Code

```bash
claude mcp add tenax -- uv run --directory /path/to/tenax-mcp python -m tenax_mcp
```

### Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "tenax": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/tenax-mcp", "python", "-m", "tenax_mcp"]
    }
  }
}
```

### Project-level `.mcp.json`

Add to any project directory:

```json
{
  "mcpServers": {
    "tenax": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/tenax-mcp", "python", "-m", "tenax_mcp"]
    }
  }
}
```

## Usage Examples

Once configured, ask Claude:

- "Run DMRG on a 20-site Heisenberg chain with bond dimension 100"
- "What's the free energy of the 2D Ising model at T = 2.27?"
- "Find the optimal contraction order for this tensor network"
- "Generate code for an iPEPS calculation on the 2D Heisenberg model"
- "Validate this tensor network for dimension consistency"
- "Show me the spin-1/2 operators"

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Test with MCP inspector
uv run mcp dev src/tenax_mcp/server.py

# Run the server directly
uv run python -m tenax_mcp
```

## Hamiltonian Term Format

Terms are specified as a list of dicts:

```json
[
  {
    "coeff": 1.0,
    "ops": [
      {"op": "Sz", "site": 0},
      {"op": "Sz", "site": 1}
    ]
  },
  {
    "coeff": 0.5,
    "ops": [
      {"op": "Sp", "site": 0},
      {"op": "Sm", "site": 1}
    ]
  }
]
```

This encodes H = Sz₀·Sz₁ + 0.5·S⁺₀·S⁻₁.

## License

MIT
