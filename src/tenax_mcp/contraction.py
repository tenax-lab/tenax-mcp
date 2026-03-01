"""Contraction optimization and network validation logic."""

from __future__ import annotations

import opt_einsum


def optimize_contraction(
    tensors: list[dict],
    output_labels: list[str] | None = None,
) -> dict:
    """Find optimal contraction path and FLOP cost.

    Each tensor dict has:
      - "labels": list of string leg labels
      - "dimensions": list of int dimensions for each leg
    """
    label_to_char: dict[str, str] = {}
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    char_idx = 0

    size_dict: dict[str, int] = {}

    subscript_parts = []
    for t in tensors:
        part = ""
        for label, dim in zip(t["labels"], t["dimensions"]):
            if label not in label_to_char:
                if char_idx >= len(chars):
                    return {"error": "Too many unique labels (max 52)."}
                label_to_char[label] = chars[char_idx]
                char_idx += 1
            c = label_to_char[label]
            size_dict[c] = dim
            part += c
        subscript_parts.append(part)

    # Determine output labels
    if output_labels is not None:
        out_str = "".join(label_to_char[l] for l in output_labels)
    else:
        # Free indices: appear exactly once
        all_chars = "".join(subscript_parts)
        out_str = ""
        for c in dict.fromkeys(all_chars):  # preserve order, unique
            if all_chars.count(c) == 1:
                out_str += c

    subscripts = ",".join(subscript_parts) + "->" + out_str

    # Build dummy shapes for opt_einsum
    shapes = []
    for t in tensors:
        shape = tuple(t["dimensions"])
        shapes.append(shape)

    path, info = opt_einsum.contract_path(subscripts, *[__import__("numpy").empty(s) for s in shapes])

    # Map back to label names
    char_to_label = {v: k for k, v in label_to_char.items()}
    label_subscripts = ",".join(
        "".join(char_to_label[c] for c in part) for part in subscript_parts
    ) + "->" + "".join(char_to_label.get(c, c) for c in out_str)

    return {
        "subscripts": label_subscripts,
        "einsum_subscripts": subscripts,
        "contraction_path": [list(p) for p in path],
        "flops": int(info.opt_cost),
        "naive_flops": int(info.naive_cost) if hasattr(info, "naive_cost") else None,
        "speedup": float(info.naive_cost / info.opt_cost)
        if hasattr(info, "naive_cost") and info.opt_cost > 0
        else None,
        "largest_intermediate": int(info.largest_intermediate),
    }


def validate_network(
    tensors: list[dict],
) -> dict:
    """Check tensor network validity.

    Each tensor dict has:
      - "name": string tensor name
      - "labels": list of string leg labels
      - "dimensions": list of int dimensions
      - "charges" (optional): list of charge arrays per leg
      - "flow" (optional): list of "in"/"out" per leg
    """
    issues = []

    # Check dimension matching on shared labels
    label_info: dict[str, list[tuple[str, int]]] = {}
    for t in tensors:
        name = t.get("name", "unnamed")
        for label, dim in zip(t["labels"], t["dimensions"]):
            if label not in label_info:
                label_info[label] = []
            label_info[label].append((name, dim))

    for label, entries in label_info.items():
        if len(entries) > 2:
            issues.append(
                f"Label '{label}' appears on {len(entries)} tensors "
                f"({', '.join(e[0] for e in entries)}); expected at most 2."
            )
        if len(entries) == 2:
            (name_a, dim_a), (name_b, dim_b) = entries
            if dim_a != dim_b:
                issues.append(
                    f"Dimension mismatch on label '{label}': "
                    f"{name_a} has dim {dim_a}, {name_b} has dim {dim_b}."
                )

    # Check flow direction consistency
    for t in tensors:
        flows = t.get("flow")
        if flows:
            for f in flows:
                if f not in ("in", "out"):
                    issues.append(
                        f"Tensor '{t.get('name', 'unnamed')}': invalid flow '{f}', "
                        f"expected 'in' or 'out'."
                    )

    # Check charge matching
    for label, entries in label_info.items():
        if len(entries) == 2:
            t1 = next(t for t in tensors if label in t["labels"])
            t2 = next(
                t for t in tensors if label in t["labels"] and t is not t1
            )
            charges_1 = t1.get("charges")
            charges_2 = t2.get("charges")
            if charges_1 and charges_2:
                idx1 = t1["labels"].index(label)
                idx2 = t2["labels"].index(label)
                if charges_1[idx1] != charges_2[idx2]:
                    issues.append(
                        f"Charge mismatch on label '{label}' between "
                        f"'{t1.get('name')}' and '{t2.get('name')}'."
                    )

    return {
        "valid": len(issues) == 0,
        "num_tensors": len(tensors),
        "num_labels": len(label_info),
        "shared_labels": [l for l, e in label_info.items() if len(e) == 2],
        "free_labels": [l for l, e in label_info.items() if len(e) == 1],
        "issues": issues,
    }
