# Boundary-safe beat-window extraction

## Scope

The supported package extracts fixed-width, single-channel windows centered on mapped annotation
locations. It preserves row-level source identity, excludes incomplete boundary windows, records
overlap counts, and writes an ignored NPZ artifact plus a JSON audit report.

This stage does not itself split records, train a model, assess signal quality, or establish that the
configured channel and window geometry are appropriate for another purpose.

## Versioned historical configuration

`configs/windowing-v1.toml` makes the archived workflow's choices explicit:

| Setting | Value | Effect |
|---|---:|---|
| Pre-window duration | 3 seconds | 1,080 samples at 360 Hz |
| Post-window duration | 3 seconds | 1,080 samples at 360 Hz |
| Total width | 6 seconds | 2,160 samples |
| Channel index | 0 | First loaded WFDB channel |
| Boundary policy | `exclude` | Do not pad or truncate incomplete windows |

For a center sample `c`, the slice is `[c - 1080:c + 1080]`. Python's exclusive right bound gives
2,160 samples. The annotation is at offset 1,080. The implementation rejects duration/sample-rate
combinations that do not produce whole sample counts.

Channel index 0 preserves the historical project behavior but is not a channel-selection analysis.
The selected channel name is recorded in every extraction report and artifact.

## Extract one record

```fish
uv run ecg-data extract-windows \
  --config configs/mitdb-v1.0.0.toml \
  --mapping-config configs/annotation-map-v1.toml \
  --window-config configs/windowing-v1.toml \
  --data-dir data/raw/mitdb/1.0.0 \
  --record-id 100 \
  --output data/interim/record-100-windows.npz \
  --report artifacts/record-100-window-report.json
```

Both output locations are ignored by Git.

## NPZ artifact contract

The artifact uses NumPy arrays that can be opened with `allow_pickle=False`. It contains:

- `windows`: rows by window samples;
- `record_ids`: source record ID for every row;
- `center_sample_indices`: original annotation location;
- `source_symbols`: original WFDB symbol;
- `target_values`: versioned project target;
- sample rate, channel index, and channel name; and
- mapping and window-configuration names and versions.

## Audit report

The JSON report records input mapped annotations, emitted windows, left- and right-boundary
exclusions, output counts by target, configured geometry, selected channel, and the number of
adjacent emitted windows that overlap.

Overlap is expected when annotations are closer than six seconds. It is reported because overlapping
windows must never be allowed to cross train, validation, or test boundaries. The supported
[split stage](record-grouped-splitting.md) assigns complete records before any future model work.

## Change control

Changing duration, channel, boundary behavior, annotation mapping, or sample rate changes the
derived dataset definition. Such changes require new configuration versions and regenerated
artifacts rather than overwriting earlier results.

## Channel identity contract

Window extraction uses an explicit channel identity contract. The public window configuration selects the MIT-BIH channel by name:

```toml
channel_name = "MLII"
```

Name-based selection is resolved per record. For each record, extraction inspects the available signal names, finds the configured channel name, and uses the resolved record-local channel index for slicing. This avoids assuming that positional channel index `0` has the same signal identity in every record.

Positional selection remains supported for compatibility:

```toml
channel_index = 0
```

However, positional selection is a lower-level selector. It records the configured index and the resolved channel identity. It is unsafe for datasets where signal ordering varies by record unless downstream shard identity validation confirms that all produced shards resolve to one channel identity.

A valid window configuration must provide exactly one channel selector:

- `channel_name`; or
- `channel_index`.

Providing both selectors, or neither selector, is rejected during configuration loading.

Generated window artifacts preserve channel lineage fields sufficient to audit selection:

- selector type;
- configured channel index, when positional selection is used;
- configured channel name, when name-based selection is used;
- resolved per-record channel index; and
- resolved per-record channel name.

Missing named channels fail as configuration/data-contract errors. The error reports the affected record and the available channel names. This is not treated as a local artifact cleanup issue.

Strict channel identity validation is intentional. Mixed channel identities such as `MLII` and `V5` must not be treated as equivalent, even when the window geometry is otherwise identical. This repository treats that mismatch as a reproducibility and lineage defect, not as a modeling or clinical claim.
