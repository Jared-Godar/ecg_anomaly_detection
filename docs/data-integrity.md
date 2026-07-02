# Local dataset inventory and integrity checks

## Scope

The supported package can validate the expected MIT-BIH v1.0.0 file inventory and create a local
SHA-256 baseline. It does not download the dataset, commit data, or establish that local files are
authentic copies of the PhysioNet source.

The versioned configuration at `configs/mitdb-v1.0.0.toml` defines 48 record IDs and three required
file types per record:

- `.hea`: WFDB header;
- `.dat`: signal data; and
- `.atr`: reference annotations.

This produces an expected inventory of 144 required files. Additional upstream documentation files
may be present locally; they are outside this pipeline contract and are not hashed by this command.

## Create a local baseline

After retrieving the dataset from the authoritative source into an ignored local directory, create
an integrity manifest:

```fish
uv run ecg-data inventory \
  --config configs/mitdb-v1.0.0.toml \
  --data-dir data/raw/mitdb/1.0.0 \
  --output artifacts/mitdb-v1.0.0-inventory.json
```

The command fails if any required file is missing or is not a regular file. The output records the
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

This is a local integrity control: it detects change relative to the first recorded baseline. It is
not a substitute for trusted checksums or signatures published by the dataset owner. Acquisition
work should validate upstream authenticity when authoritative digest material is available and
record how the source was retrieved.

No ECG files or generated inventory manifests are required by CI. Tests build small synthetic files
inside temporary directories.
