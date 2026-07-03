# Repeatable dataset acquisition

## Scope

The supported acquisition stage retrieves the configured MIT-BIH v1.0.0 record files directly from
PhysioNet's versioned HTTPS file directory. It writes source files only to the ignored canonical raw
zone and creates an ignored acquisition manifest containing source URLs, sizes, and SHA-256 digests.

The authoritative [PhysioNet dataset page](https://physionet.org/content/mitdb/1.0.0/) identifies
`https://physionet.org/files/mitdb/1.0.0/` as the versioned download directory. The project config
keeps the attribution landing page and file transport base as separate fields.

## Acquire the configured files

From the repository root:

```fish
uv run ecg-data acquire \
  --repository-root . \
  --config configs/mitdb-v1.0.0.toml \
  --data-dir data/raw/mitdb/1.0.0 \
  --output artifacts/mitdb-v1.0.0-acquisition.json
```

The command retrieves only the 144 configured `.atr`, `.dat`, and `.hea` files. It does not fetch
extra website content, historical alternates, or documentation files.

## Safety and recovery behavior

Acquisition applies the following controls:

- HTTPS is required and credentials, query strings, and all redirects are rejected;
- each response must return HTTP 200 and respect a bounded per-file size;
- downloads are streamed into a temporary staging directory rather than partial destination files;
- the baseline manifest is written atomically before staged files are committed to the raw zone;
- source files and the baseline manifest are never overwritten;
- required files already present without a baseline cause a failure rather than being trusted;
- reruns verify existing files by size and SHA-256 without redownloading them; and
- files missing after a completed or interrupted run are downloaded again only when they match the
  recorded baseline digest.

This makes normal reruns idempotent and makes interrupted finalization recoverable. A changed local
file or a changed upstream response fails closed and requires investigation; the command provides no
`--force` option.

## Trust boundary

The first acquisition relies on the configured versioned PhysioNet URL, HTTPS transport, and the
repository's reviewed configuration. Its SHA-256 values become a local baseline. They detect later
change but are not independent proof of authenticity because this workflow does not consume an
independently published signed checksum manifest for the selected files.

The files remain subject to PhysioNet's access terms and the Open Data Commons Attribution License.
Acquisition does not transfer ownership, authorize republishing, or make the data suitable for
clinical use. Raw files and generated manifests remain excluded from Git and are never required by
CI; tests use synthetic byte streams only.
