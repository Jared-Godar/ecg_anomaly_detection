# Local dataset inventory and integrity checks

## Scope

The supported package can retrieve the expected MIT-BIH v1.0.0 files, validate their inventory, and
create separate acquisition and local-integrity evidence. Acquisition is documented in
[dataset acquisition](dataset-acquisition.md). These controls do not commit data or independently
establish authenticity beyond the configured versioned source, reviewed hashes, and HTTPS transport.

The versioned configuration at `configs/mitdb-v1.0.0.toml` defines 48 record IDs and three required
file types per record:

- `.hea`: WFDB header;
- `.dat`: signal data; and
- `.atr`: reference annotations.

This produces an expected inventory of 144 required files. The same committed config pins each
relative path, exact byte size, and SHA-256 digest. Files outside that contract are rejected; keep
additional upstream documentation outside `data/raw/mitdb/1.0.0/`.

## Create a local baseline

After acquisition, create a separate current-state integrity manifest:

```fish
uv run ecg-data inventory \
  --config configs/mitdb-v1.0.0.toml \
  --data-dir data/raw/mitdb/1.0.0 \
  --output artifacts/mitdb-v1.0.0-inventory.json
```

The command fails if any required file is missing, unexpected, not a regular file, or differs from
the committed expected size or digest. The output records the observed
dataset identity, UTC creation time, relative file name, byte size, and SHA-256 digest. The manifest
is generated evidence and remains ignored by Git under `artifacts/`.

## Verify files later

```fish
uv run ecg-data verify \
  --config configs/mitdb-v1.0.0.toml \
  --data-dir data/raw/mitdb/1.0.0 \
  --manifest artifacts/mitdb-v1.0.0-inventory.json
```

Verification fails when the configured dataset identity or required file set differs, a required
file is missing, or a file's size or SHA-256 digest changed.

## Trust boundary

The committed values are repository-reviewed expectations. All 144 SHA-256 values match the
`SHA256SUMS.txt` that PhysioNet distributes inside its versioned v1.0.0 ZIP; exact byte sizes were
measured from that clean extraction. The ZIP integrity test passed, and `100.hea` was also compared
with a separate direct download. Review confirmed exact coverage of the configured file set.
PhysioNet's checksum file is not signed, so this pins expected content and detects later change but
is not cryptographic proof of publisher identity.

No ECG files or generated inventory manifests are required by CI. Tests build small synthetic files
inside temporary directories.
