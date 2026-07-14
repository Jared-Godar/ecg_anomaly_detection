# Repeatable dataset acquisition

## Scope

The supported acquisition stage retrieves the configured MIT-BIH v1.0.0 record files directly from
PhysioNet's versioned HTTPS file directory. It writes source files only to the ignored canonical raw
zone and creates an ignored acquisition manifest containing source URLs, sizes, and SHA-256 digests.
Before installation, every observed size and digest must match the repository-reviewed expectation
committed in `configs/mitdb-v1.0.0.toml`.

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

`acquire` prints a `[1/1] acquire: starting` banner, its existing completion message, and a
`[1/1] acquire: complete in MM:SS` banner (or `failed after MM:SS` if it raises) to stdout, matching
[the orchestrated `run-pipeline` command's progress output](pipeline-orchestration.md#progress-output).
Between those banners it prints one completed line per configured record—not one line per file—so
a first 48-record download stays visibly active. Each line says whether that record's companion
files were downloaded and verified, verified from existing local files, or partially restored,
followed by a qualified timing suffix: the record's own measured duration, the acquisition phase's
measured elapsed time, and an `approx. remaining` projection. The projection multiplies the mean
completed-record duration observed in the current run by the records still outstanding; until three
records have completed it reports an explicit `estimating...` warm-up state instead of an unstable
number, and the final record reports `00:00`. Measured durations are observations; the remaining
value is always labeled `approx.` and is never a deadline or guarantee:

```text
    record 5/48 (104): downloaded and verified 3 files | record 00:14 | elapsed 01:09 | approx. remaining 09:54
```

This is purely observational and never changes the command's exit code or the acquired files.

## Safety and recovery behavior

Acquisition applies the following controls:

- HTTPS is required and credentials, query strings, and all redirects are rejected;
- each response must return HTTP 200 and respect a bounded per-file size;
- downloads are streamed into a temporary staging directory rather than partial destination files;
- transient connectivity failures (timeouts, dropped/reset connections, name-resolution failures,
  and HTTP 429/500/502/503/504) are retried per file up to three total attempts with exponential
  backoff (2s, then 4s), with the partially staged file removed between attempts; permanent
  failures (HTTP 404/403, size-cap violations, digest or size mismatches, rejected redirects) fail
  fast on the first attempt, and every retry outcome still passes the same staged integrity checks;
- when retries are exhausted, the failure message names the affected URL and attempt count, states
  that the cause is an external connectivity or service condition rather than a repository or setup
  defect, and gives re-run remediation — atomicity means a re-run restarts cleanly;
- missing expected metadata, unexpected directory entries, size mismatches, and SHA-256 mismatches
  fail with the affected path and expected/observed evidence;
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
repository-reviewed expected values. Those values were calculated from a clean download of the
versioned v1.0.0 ZIP and reviewed for exact coverage of the configured 144-file set. Every digest
matches PhysioNet's `SHA256SUMS.txt` distributed in that ZIP; the ZIP integrity test passed and one
file was cross-checked through a separate direct download. The upstream checksum file is unsigned,
so it does not independently prove publisher identity. Network availability and continued external
hosting remain outside repository control.

The files remain subject to PhysioNet's access terms and the Open Data Commons Attribution License.
Acquisition does not transfer ownership, authorize republishing, or make the data suitable for
clinical use. Raw files and generated manifests remain excluded from Git and are never required by
CI; tests use synthetic byte streams only.
