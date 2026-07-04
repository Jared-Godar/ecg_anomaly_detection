# Split quality reporting

The split stage writes `split_quality_summary.json` beside the split manifest by default. Pipeline
runs place it at `artifacts/runs/<run-id>/split_quality_summary.json` and hash it as run evidence.
Both paths are generated, ignored artifacts and must not be committed.

## Diagnostics

The stable JSON document records the complete acceptance configuration, configured and actual
partition ratios, explicit subject and
record disjointness results, and per-partition subject, record, shard, window, and represented-class
counts. It also reports class counts and prevalence. When the complete observed target set is
binary `0`/`1`, `binary_counts` identifies negative and positive counts; otherwise that field is
`null`. Lists and mappings use deterministic ordering.

## Acceptance policy

`[split.quality]` in `configs/splitting-v2.toml` configures minimum subjects, records, windows, and
positive examples per partition; required classes and the partitions requiring their coverage; and
the maximum deviation between configured and actual subject ratios. `default_severity` is either
`warning` or `failure`. Check names in `warning_checks` override the default to warnings.

The supported check names are `minimum_subjects`, `minimum_records`, `minimum_windows`,
`minimum_positive_examples`, `required_class_coverage`, `partition_ratio_deviation`,
`subject_disjointness`, and `record_disjointness`. Failure-level violations are written to the
summary and then stop the command before downstream indexing, training, or evaluation. Warnings are
reported but do not stop processing.

## Limitations

Ratios are assigned by subject count, not by class or window stratification. Quality checks expose
sparse or incomplete partitions; they do not prove population generalization or model quality.
This workflow does not score the test partition, publish held-out benchmark metrics, or make
clinical or diagnostic claims. Evaluation remains validation-only.
