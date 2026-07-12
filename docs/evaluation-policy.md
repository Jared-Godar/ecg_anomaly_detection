# Evaluation policy

![Technical banner for evaluation policy, showing validation boundaries and limitation motifs.](assets/ecg-evaluation-policy-banner.png)

## Supported development evaluation

The implemented evaluator scores only the `validation` partition. Validation evidence supports
pipeline verification and bounded model development. It is not a final benchmark, does not establish
generalization, and must not be presented as clinical evidence.

Training and selection decisions may use training and validation evidence. They must not inspect the
protected `test` partition, its shards, labels, predictions, aggregates, or metrics.

## Protected benchmark evaluation

The `test` partition remains disabled by default and may be accessed only by the separately
reviewed, explicitly enabled command under the eligibility, immutable lineage, approval, rerun,
disclosure, and archival controls in [Benchmark governance](benchmark-governance.md). Issue #73
completed one governed execution; its aggregate evidence is recorded in
[Held-out evaluation](held-out-evaluation.md).

Protected results may not be used for model or configuration iteration. Any rerun remains limited
to the policy's documented pre-result infrastructure-failure or invalidating-defect conditions.

## Interpretation limits

Evaluation must disclose dataset, annotation, class-imbalance, binary-mapping, grouped-split, and
historical-dataset limitations. Neither validation nor a future benchmark can by itself establish
model quality beyond its stated measures, generalization, clinical validity, or medical utility.
Claims of diagnostic usefulness, healthcare-AI readiness, medical-device suitability, or production
healthcare readiness are prohibited.
