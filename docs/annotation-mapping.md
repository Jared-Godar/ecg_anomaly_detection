# Versioned annotation mapping

## Scope

The supported package maps original WFDB annotation symbols into a project-specific binary target
and explicitly reports exclusions. This mapping preserves the 2022 project's selected symbol policy
for historical continuity; it is not a diagnosis, a complete clinical taxonomy, or evidence that
the selected classes are appropriate for another use.

The versioned contract is stored in `configs/annotation-map-v1.toml`.

## Target policy

| Target | Value | Source symbols | Meaning in this project |
|---|---:|---|---|
| `reference_normal` | 0 | `N` | Upstream `N` reference annotation |
| `selected_other` | 1 | `L R V / A f F j a E J e S` | Selected beat symbols grouped by the historical project |

The second target is deliberately named `selected_other`, not “disease” or a diagnostic label. It
combines heterogeneous annotation types solely to reproduce the historical binary project target.

The mapping explicitly excludes 24 symbols retained from the archived workflow:

```text
[ ! ] x ( ) p t u ` ' ^ | ~ + s T * D = " @ Q ?
```

Excluded annotations are counted in the audit report and are not returned in the mapped annotation
set. Any symbol absent from both target rules and the exclusion list causes an error. This
closed-world behavior prevents a new upstream annotation from being silently discarded or assigned
to a target.

## Audit one record

The command first runs structural record validation, then applies the versioned mapping:

```fish
uv run ecg-data map-annotations \
  --config configs/mitdb-v1.0.0.toml \
  --mapping-config configs/annotation-map-v1.toml \
  --data-dir data/raw/mitdb/1.0.0 \
  --record-id 100 \
  --output artifacts/record-100-annotation-map.json
```

The generated JSON report records:

- mapping name and version;
- record ID and total input count;
- counts for every observed source symbol;
- included counts by project target; and
- excluded counts by source symbol.

Reports under `artifacts/` remain ignored. The mapped arrays retain record ID, source sample index,
original source symbol, and integer target value for downstream window generation.

## Change control

Changing a symbol assignment changes the derived target definition. Such a change must use a new
mapping version, regenerate downstream data, and be evaluated separately. It must not silently
overwrite results produced with an earlier mapping.
