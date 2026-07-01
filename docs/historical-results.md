# Historical results

## Scope

This page records the outputs and known evaluation limitations of the original 2022 experiment. It is an audit trail, not a claim that the model is suitable for medical or clinical use.

## Saved test output

The final random forest in [`report.ipynb`](../archive/original_2022/report.ipynb) uses a maximum depth of 10 and a minimum leaf size of 3. Its saved test confusion matrix is:

| | Predicted normal | Predicted abnormal |
|---|---:|---:|
| Reference normal | 7,473 | 17 |
| Reference abnormal | 497 | 2,923 |

The notebook derives these historical values:

| Metric | Value |
|---|---:|
| Accuracy | 0.953 |
| Abnormal-class recall | 0.855 |
| Abnormal-class precision | 0.994 |
| Abnormal-class F1 | 0.919 |
| False-positive rate | 0.002 |
| False-negative rate | 0.145 |

The training baseline predicts the majority class and has saved accuracy of 0.684. The change to 0.953 is 26.9 percentage points, or approximately 39.3% relative to the baseline. Future reporting should state which comparison is used.

## Known metric defect

One cell reports `Accuracy of random forest classifier 1 on validation set: 1.00` using this expression:

```python
rf1.score(X_validate, y_pred_validate_rf1)
```

The second argument should be the reference labels, `Y_validate`, not the model's own predictions. The same notebook's confusion matrix gives a validation accuracy of 0.845 for this depth-three model.

The historical notebook also reverses the human-readable names for positive and negative support in two hand-calculated metric sections. The confusion matrices and scikit-learn classification reports should be treated as the more reliable record.

## Evaluation design limitation

The dataset is divided after beat windows are created, using ordinary random train/test splits. Record identity is not retained as a grouping constraint. This creates two related risks:

1. windows from the same subject can occur in training, validation, and test sets; and
2. neighboring six-second windows can overlap while being assigned to different sets.

The saved metrics therefore measure performance on a randomly held-out set of windows from a mixed pool. They do not establish performance on unseen subjects or real-world data.

## Requirements for a modern benchmark

A replacement benchmark should:

- retain the record identifier with every generated window;
- allocate complete records to only one split;
- document the annotation-to-target mapping;
- report record and class distributions by split;
- fit all learned transformations only on training data;
- calculate metrics through tested library functions;
- include per-class results and confusion matrices;
- record configuration, dependency, dataset, and source-code versions; and
- distinguish exploratory validation from one-time final evaluation.

Until that work is complete, historical values should always be presented with the leakage caveat.
