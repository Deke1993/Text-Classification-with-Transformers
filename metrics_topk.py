#note this a copy from metrics.py, adjusted by myself to be able to handle top k classification.


# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Measures for models interacting with an oracle.

Oracle Collaborative AUC measures the usefulness of model uncertainty scores in
facilitating human-computer collaboration (e.g., between a neural model and an
"oracle", e.g. a human moderator in moderating online toxic comments).

The idea is that given a large amount of testing examples, the model will first
generate predictions for all examples, and then send a certain percentage of
examples that it is not confident about to the oracle, which returns perfect
predictions for those examples.

The goal of this metric is to understand, under capacity constraints (e.g. if
the model is only allowed to send 0.1% of total examples to the oracle), how
well the model can collaborate with it to achieve the best overall performance.
In this way, these metrics attempt to quantify the behavior of the full
model-oracle system rather than of the model alone.

A model that collaborates with an oracle well should not be accurate, but also
capable of quantifying its uncertainty well (i.e., its uncertainty should be
calibrated such that uncertainty ≅ model accuracy).
"""
from typing import Any, Dict, Mapping, Optional, Sequence
import tensorflow as tf


def _replace_first_and_last_elements(original_tensor: Sequence[float],
                                     new_first_elem: float,
                                     new_last_elem: float):
  """Return a copy of original_tensor replacing its first and last elements."""
  return tf.concat([[new_first_elem], original_tensor[1:-1], [new_last_elem]],
                   axis=0)


def _compute_correct_predictions(y_true: Sequence[float],
                                 y_pred: Sequence[float],
                                 k,
                                 dtype: tf.DType = tf.float32) -> tf.Tensor:
  """Computes binary 'labels' of prediction correctness.

  Args:
    y_true: The ground truth labels. Shape (batch_size, ).
    y_pred: The predicted labels. Must be integer valued predictions for label
      index rather than the predictive probability. For multi-label
      classification problems, y_pred is typically obtained as
      `tf.math.reduce_max(logits)`. Shape (batch_size, ).
    dtype: (Optional) data type of the metric result.

  Returns:
    A Tensor of dtype and shape (batch_size, ).
  """
  y_true = tf.cast(tf.convert_to_tensor(y_true), dtype=tf.int32)
  y_pred = tf.cast(tf.convert_to_tensor(y_pred), dtype=tf.float32)

  # Ranks of both y_pred and y_true should be 1.
  if len(y_true.shape) != 1 or len(y_pred.shape) != 2:
    raise ValueError("Ranks of y_true and y_pred must both be 1. "
                     f"Got {len(y_true.shape)} and {len(y_pred.shape)}")

  # Creates binary 'label' of correct prediction, shape (batch_size, ).
  correct_preds = tf.math.in_top_k(y_true, y_pred,k=k)
  return tf.cast(correct_preds, dtype=tf.float32)


class OracleCollaborativeAUC(tf.keras.metrics.AUC):
  """Computes the approximate oracle-collaborative equivalent of the AUC.

  This metric computes four local variables: binned_true_positives,
  binned_true_negatives, binned_false_positives, and binned_false_negatives, as
  a function of a linearly spaced set of thresholds and score bins. These are
  then sent to the oracle in increasing bin order, and used to compute the
  Oracle-Collaborative ROC-AUC or Oracle-Collaborative PR-AUC.

  Note because the AUC must be computed online that the results are not exact,
  but rather are expected values, similar to the regular AUC computation.
  """

  def __init__(self,
               oracle_fraction: float = 0.01,
               max_oracle_count: Optional[int] = None,
               oracle_threshold: Optional[float] = None,
               num_bins: int = 1000,
               num_thresholds: int = 200,
               curve: str = "ROC",
               summation_method: str = "interpolation",
               name: Optional[str] = None,
               dtype: Optional[tf.DType] = None):
    """Constructs an expected oracle-collaborative AUC metric.

    Args:
      oracle_fraction: the fraction of total examples to send to the oracle.
      max_oracle_count: if set, the maximum number of total examples to send to
        the oracle.
      oracle_threshold: (Optional) Threshold below which to send all predictions
        to the oracle (less than or equal to), irrespective of oracle_fraction
        and max_oracle_count (i.e. these arguments are unused).
      num_bins: Number of bins for the uncertainty score to maintain over the
        interval [0, 1].
      num_thresholds: (Optional) Number of thresholds to use in linearly
        interpolating the AUC curve.
      curve: Name of the curve to be computed, either ROC (default) or PR
        (Precision-Recall).
      summation_method: Specifies the Riemann summation method. 'interpolation'
        applies the mid-point summation scheme for ROC. For PR-AUC, interpolates
        (true/false) positives but not the ratio that is precision (see Davis &
        Goadrich 2006 for details); 'minoring' applies left summation for
        increasing intervals and right summation for decreasing intervals;
        'majoring' does the opposite.
      name: (Optional) Name of this metric.
      dtype: (Optional) Data type. Must be floating-point.  Currently only
        binary data is supported.

    oracle_fraction and max_oracle_count place different limits on how many
    examples can be sent to the oracle (scaling with the number of total
    examples, and a constant limit independent of it, respectively). Both limits
    are applied, i.e. the stricter of the two rules determines the total number.
    """
    # Validate inputs.
    if not 0 <= oracle_fraction <= 1:
      raise ValueError("oracle_fraction must be between 0 and 1.")
    if max_oracle_count and max_oracle_count < 0:
      raise ValueError("max_oracle_count must be a non-negative integer.")
    if oracle_threshold and not 0 <= oracle_fraction <= 1:
      raise ValueError("oracle_threshold must be between 0 and 1.")
    if num_bins <= 1:
      raise ValueError("num_bins must be > 1.")
    if dtype and not dtype.is_floating:
      raise ValueError("dtype must be a float type.")

    self.oracle_fraction = oracle_fraction
    self.max_oracle_count = max_oracle_count
    self.num_bins = num_bins
    self.oracle_threshold = oracle_threshold

    # If oracle_threshold is set, the examples sent to the oracle are computed
    # differently; we only need two bins in this case.
    if self.oracle_threshold is not None:
      self.num_bins = 2

    super().__init__(
        num_thresholds=num_thresholds,
        curve=curve,
        summation_method=summation_method,
        name=name,
        dtype=dtype)

    self.binned_true_positives = self.add_weight(
        "binned_true_positives",
        shape=(self.num_thresholds, self.num_bins),
        initializer=tf.zeros_initializer)

    self.binned_true_negatives = self.add_weight(
        "binned_true_negatives",
        shape=(self.num_thresholds, self.num_bins),
        initializer=tf.zeros_initializer)

    self.binned_false_positives = self.add_weight(
        "binned_false_positives",
        shape=(self.num_thresholds, self.num_bins),
        initializer=tf.zeros_initializer)

    self.binned_false_negatives = self.add_weight(
        "binned_false_negatives",
        shape=(self.num_thresholds, self.num_bins),
        initializer=tf.zeros_initializer)

  def update_state(self,
                   labels: Sequence[float],
                   probabilities: Sequence[float],
                   custom_binning_score: Optional[Sequence[float]] = None,
                   **kwargs: Mapping[str, Any]) -> None:
    """Updates the confusion matrix for OracleCollaborativeAUC.

    This will flatten the labels, probabilities, and custom binning score, and
    then compute the confusion matrix over all predictions.

    Args:
      labels: Tensor of shape [N,] of class labels in [0, k-1], where N is the
        number of examples. Currently only binary labels (0 or 1) are supported.
      probabilities: Tensor of shape [N,] of normalized probabilities associated
        with the positive class.
      custom_binning_score: (Optional) Tensor of shape [N,] used for assigning
        predictions to uncertainty bins. If not set, the default is to bin by
        predicted probability. All elements of custom_binning_score must be in
        [0, 1].
      **kwargs: Other potential keywords, which will be ignored by this method.
    """
    del kwargs  # Unused
    labels = tf.convert_to_tensor(labels)
    probabilities = tf.cast(probabilities, self.dtype)

    # Reshape labels, probabilities, custom_binning_score to [1, num_examples].
    labels = tf.reshape(labels, [1, -1])
    probabilities = tf.reshape(probabilities, [1, -1])
    if custom_binning_score is not None:
      custom_binning_score = tf.cast(
          tf.reshape(custom_binning_score, [1, -1]), self.dtype)
    # Reshape thresholds to [num_thresholds, 1] for easy tiling.
    thresholds = tf.cast(tf.reshape(self._thresholds, [-1, 1]), self.dtype)

    # pred_labels and true_labels both have shape [num_thresholds, num_examples]
    pred_labels = probabilities > thresholds
    true_labels = tf.tile(tf.cast(labels, tf.bool), [self.num_thresholds, 1])

    # Bin by distance from threshold if a custom_binning_score was not set.
    if custom_binning_score is None:
      custom_binning_score = tf.abs(probabilities - thresholds)
    else:
      # Tile the provided custom_binning_score for each threshold.
      custom_binning_score = tf.tile(custom_binning_score,
                                     [self.num_thresholds, 1])

    # Bin thresholded predictions using custom_binning_score.
    batch_binned_confusion_matrix = self._bin_confusion_matrix_by_score(
        pred_labels, true_labels, custom_binning_score)

    self.binned_true_positives.assign_add(
        batch_binned_confusion_matrix["true_positives"])
    self.binned_true_negatives.assign_add(
        batch_binned_confusion_matrix["true_negatives"])
    self.binned_false_positives.assign_add(
        batch_binned_confusion_matrix["false_positives"])
    self.binned_false_negatives.assign_add(
        batch_binned_confusion_matrix["false_negatives"])

  def _bin_confusion_matrix_by_score(
      self, pred_labels: Sequence[Sequence[bool]],
      true_labels: Sequence[Sequence[bool]],
      binning_score: Sequence[Sequence[float]]) -> Dict[str, tf.Tensor]:
    """Compute the confusion matrix, binning predictions by a specified score.

    Computes the confusion matrix over matrices of predicted and true labels.
    Each element of the resultant confusion matrix is itself a matrix of the
    same shape as the original input labels.

    In the typical use of this function in OracleCollaborativeAUC, the variables
    T and N (in the args and returns sections below) are the number of
    thresholds and the number of examples, respectively.

    Args:
      pred_labels: Boolean tensor of shape [T, N] of predicted labels.
      true_labels: Boolean tensor of shape [T, N] of true labels.
      binning_score: Boolean tensor of shape [T, N] of scores to use in
        assigning labels to bins.

    Returns:
      Dictionary of strings to entries of the confusion matrix
      ('true_positives', 'true_negatives', 'false_positives',
      'false_negatives'). Each entry is a tensor of shape [T, nbins].

      If oracle_threshold was set, nbins=2, storing respectively the number of
      examples below the oracle_threshold (i.e. sent to the oracle) and above it
      (not sent to the oracle).
    """
    correct_preds = tf.math.equal(pred_labels, true_labels)

    # Elements of the confusion matrix have shape [M, N]
    pred_true_positives = tf.math.logical_and(correct_preds, pred_labels)
    pred_true_negatives = tf.math.logical_and(correct_preds,
                                              tf.math.logical_not(pred_labels))
    pred_false_positives = tf.math.logical_and(
        tf.math.logical_not(correct_preds), pred_labels)
    pred_false_negatives = tf.math.logical_and(
        tf.math.logical_not(correct_preds), tf.math.logical_not(pred_labels))

    # Cast confusion matrix elements from bool to self.dtype.
    pred_true_positives = tf.cast(pred_true_positives, self.dtype)
    pred_true_negatives = tf.cast(pred_true_negatives, self.dtype)
    pred_false_positives = tf.cast(pred_false_positives, self.dtype)
    pred_false_negatives = tf.cast(pred_false_negatives, self.dtype)

    histogram_value_range = tf.constant([0.0, 1.0], self.dtype)
    if self.oracle_threshold is not None:
      # All predictions with score <= oracle_threshold are sent to the oracle.
      # With two bins, centering the value range on oracle_threshold yields a
      # histogram with all examples sent to the oracle in the lower (left) bin.
      histogram_value_range += self.oracle_threshold - 0.5
      # Move the histogram center up by epsilon to ensure <= rather than <.
      # By default, tf histogram gives [low, high); we want (low, high].
      histogram_value_range += tf.keras.backend.epsilon()
    bin_indices = tf.histogram_fixed_width_bins(
        binning_score, histogram_value_range, nbins=self.num_bins)

    binned_true_positives = self._map_unsorted_segment_sum(
        pred_true_positives, bin_indices)
    binned_true_negatives = self._map_unsorted_segment_sum(
        pred_true_negatives, bin_indices)
    binned_false_positives = self._map_unsorted_segment_sum(
        pred_false_positives, bin_indices)
    binned_false_negatives = self._map_unsorted_segment_sum(
        pred_false_negatives, bin_indices)

    return {
        "true_positives": binned_true_positives,
        "true_negatives": binned_true_negatives,
        "false_positives": binned_false_positives,
        "false_negatives": binned_false_negatives
    }

  def _map_unsorted_segment_sum(self, tensor, indices):

    def unsorted_segment_sum_row(tensor_and_indices):
      return tf.math.unsorted_segment_sum(
          data=tensor_and_indices[0],
          segment_ids=tensor_and_indices[1],
          num_segments=self.num_bins)

    return tf.map_fn(
        fn=unsorted_segment_sum_row,
        elems=[tensor, indices],
        fn_output_signature=self.dtype)

  def reset_state(self):
    """Resets OracleCollaborativeAUC's state variables."""
    threshold_bin_zeros = tf.zeros((self.num_thresholds, self.num_bins),
                                   dtype=self.dtype)
    binned_confusion_matrix = (self.binned_true_positives,
                               self.binned_true_negatives,
                               self.binned_false_positives,
                               self.binned_false_negatives)

    tf.keras.backend.batch_set_value([
        (v, threshold_bin_zeros) for v in binned_confusion_matrix
    ])

    # Reset AUC confusion matrix variables from parent class.
    super().reset_state()

  def result(self):
    """Returns the approximate Oracle-Collaborative AUC.

    true_positives, true_negatives, false_positives, and false_negatives contain
    the binned confusion matrix for each threshold. We thus compute the
    confusion matrix (after collaborating with the oracle) as a function of the
    threshold and then integrate over threshold to approximate the final AUC.
    """
    cum_examples = tf.cumsum(
        self.binned_true_positives + self.binned_true_negatives +
        self.binned_false_positives + self.binned_false_negatives,
        axis=1)
    # The number of examples in each row is the same; choose the first.
    num_total_examples = cum_examples[0, -1]

    num_relative_oracle_examples = tf.cast(
        tf.floor(num_total_examples * self.oracle_fraction), self.dtype)
    num_absolute_oracle_examples = (
        tf.cast(self.max_oracle_count, self.dtype)
        if self.max_oracle_count else num_total_examples)
    num_oracle_examples = tf.minimum(num_relative_oracle_examples,
                                     num_absolute_oracle_examples)

    # Send all examples below the threshold, i.e. all examples in the first bin.
    if self.oracle_threshold is not None:
      num_oracle_examples = cum_examples[0, 0]

    expected_true_positives = tf.zeros_like(self.true_positives)
    expected_true_negatives = tf.zeros_like(self.true_negatives)
    expected_false_positives = tf.zeros_like(self.false_positives)
    expected_false_negatives = tf.zeros_like(self.false_negatives)

    # Add true positives and true negatives predicted by the oracle. All
    # incorrect predictions are corrected.
    expected_true_positives += tf.reduce_sum(
        tf.where(cum_examples <= num_oracle_examples,
                 self.binned_true_positives + self.binned_false_negatives, 0.0),
        axis=1)
    expected_true_negatives += tf.reduce_sum(
        tf.where(cum_examples <= num_oracle_examples,
                 self.binned_true_negatives + self.binned_false_positives, 0.0),
        axis=1)

    # Identify the final bin the oracle sees examples from, and the remaining
    # number of predictions it can make on that bin.
    last_oracle_bin = tf.argmax(cum_examples > num_oracle_examples, axis=1)
    last_oracle_bin_indices = tf.stack(
        [tf.range(self.num_thresholds, dtype=tf.int64), last_oracle_bin],
        axis=1)
    last_complete_bin = last_oracle_bin - 1
    # The indices for tf.gather_nd must be positive; use this list for selection
    error_guarded_last_complete_bin = tf.abs(last_complete_bin)
    last_complete_bin_indices = (
        tf.stack([
            tf.range(self.num_thresholds, dtype=tf.int64),
            error_guarded_last_complete_bin
        ],
                 axis=1))

    last_complete_bin_cum_examples = tf.gather_nd(cum_examples,
                                                  last_complete_bin_indices)
    last_oracle_bin_cum_examples = tf.gather_nd(cum_examples,
                                                last_oracle_bin_indices)
    oracle_predictions_used = tf.where(last_complete_bin >= 0,
                                       last_complete_bin_cum_examples, 0.0)
    remaining_oracle_predictions = tf.where(
        last_oracle_bin_cum_examples > num_oracle_examples,
        num_oracle_examples - oracle_predictions_used, 0.0)

    # Add the final oracle bin (where the oracle makes some predictions) to the
    # confusion matrix.
    tp_last_oracle_bin = tf.gather_nd(self.binned_true_positives,
                                      last_oracle_bin_indices)
    tn_last_oracle_bin = tf.gather_nd(self.binned_true_negatives,
                                      last_oracle_bin_indices)
    fp_last_oracle_bin = tf.gather_nd(self.binned_false_positives,
                                      last_oracle_bin_indices)
    fn_last_oracle_bin = tf.gather_nd(self.binned_false_negatives,
                                      last_oracle_bin_indices)
    last_bin_count = (
        tp_last_oracle_bin + tn_last_oracle_bin + fp_last_oracle_bin +
        fn_last_oracle_bin)

    corrected_fn_last_bin = tf.math.divide_no_nan(
        fn_last_oracle_bin * remaining_oracle_predictions, last_bin_count)
    corrected_fp_last_bin = tf.math.divide_no_nan(
        fp_last_oracle_bin * remaining_oracle_predictions, last_bin_count)

    expected_true_positives += corrected_fn_last_bin
    expected_true_negatives += corrected_fp_last_bin
    expected_false_positives -= corrected_fp_last_bin
    expected_false_negatives -= corrected_fn_last_bin

    # Add the section of the confusion matrix untouched by the oracle.
    expected_true_positives += tf.reduce_sum(
        tf.where(cum_examples > num_oracle_examples, self.binned_true_positives,
                 0.0),
        axis=1)
    expected_true_negatives += tf.reduce_sum(
        tf.where(cum_examples > num_oracle_examples, self.binned_true_negatives,
                 0.0),
        axis=1)
    expected_false_positives += tf.reduce_sum(
        tf.where(cum_examples > num_oracle_examples,
                 self.binned_false_positives, 0.0),
        axis=1)
    expected_false_negatives += tf.reduce_sum(
        tf.where(cum_examples > num_oracle_examples,
                 self.binned_false_negatives, 0.0),
        axis=1)

    # Reset the first and last elements of the expected confusion matrix to get
    # the final confusion matrix. Because the thresholds for these entries are
    # outside [0, 1], they should be left untouched and not sent to the oracle.
    expected_true_positives = _replace_first_and_last_elements(
        expected_true_positives, tf.reduce_sum(self.binned_true_positives[0]),
        tf.reduce_sum(self.binned_true_positives[-1]))
    expected_true_negatives = _replace_first_and_last_elements(
        expected_true_negatives, tf.reduce_sum(self.binned_true_negatives[0]),
        tf.reduce_sum(self.binned_true_negatives[-1]))
    expected_false_positives = _replace_first_and_last_elements(
        expected_false_positives, tf.reduce_sum(self.binned_false_positives[0]),
        tf.reduce_sum(self.binned_false_positives[-1]))
    expected_false_negatives = _replace_first_and_last_elements(
        expected_false_negatives, tf.reduce_sum(self.binned_false_negatives[0]),
        tf.reduce_sum(self.binned_false_negatives[-1]))

    self.true_positives.assign(expected_true_positives)
    self.true_negatives.assign(expected_true_negatives)
    self.false_positives.assign(expected_false_positives)
    self.false_negatives.assign(expected_false_negatives)

    return super().result()


class CalibrationAUC_Topk(tf.keras.metrics.AUC):
  """Implements AUC metric for uncertainty calibration.

  [1]: Ranganath Krishnan, Omesh Tickoo. Improving model calibration with
       accuracy versus uncertainty optimization. In _Neural Information Process
       Systems_, 2020.
       https://papers.nips.cc/paper/2020/file/d3d9446802a44259755d38e6d163e820-Paper.pdf

  Given a model that computes uncertainty score, this metric computes the AUC
  metric for a binary prediction task where the binary "label" is the predictive
  correctness (a binary label of 0's and 1's), and the prediction score is the
  confidence score. Both ROC- and PR-type curves are supported. It measures
  a model's uncertainty calibration in the sense that it examines the degree to
  which a model uncertainty is predictive of its generalization error.

  Different from Expected Calibration Error (ECE), calibration AUC is scale
  invariant and focuses on the ranking performance of the uncertainty score
  (i.e., whether high uncertainty predictions are wrong) rather than the exact
  value match between the accuracy and the uncertainty scores.

  As a result, calibration AUC more closely reflects the use case of uncertainty
  in an autonomous system, where the uncertainty score is either used as a
  ranking signal, or is used to make a binary decision based on a
  machine-learned threshold. Another benefit of calibration AUC is that it
  cannot be trivially reduced using post-hoc calibration heuristics such as
  temperature scaling or isotonic regression, since these methods don't improve
  the ranking performance of the uncertainty score.
  """

  def __init__(self,
               curve: str = "ROC",
               multi_label: bool = False,
               correct_pred_as_pos_label: bool = True,
               **kwargs: Mapping[str, Any]):
    """Constructs CalibrationAUC_Topk class.

    Args:
      curve: Specifies the name of the curve to be computed, 'ROC' [default] or
        'PR' for the Precision-Recall-curve.
      multi_label: Whether tf.keras.metrics.AUC should treat input label as
        multi-class. Ignored.
      correct_pred_as_pos_label: Whether to use correct prediction as positive
        label for AUC computation. If False then use it as negative label.
      **kwargs: Other keyword arguments to tf.keras.metrics.AUC.
    """
    # Ignore `multi_label` since accuracy v.s. uncertainty is a binary problem
    # (i.e., the "label" is whether prediction is correct or not).
    if multi_label:
      raise ValueError("`multi_label` must be False for Calibration AUC.")

    super().__init__(curve=curve, multi_label=False, **kwargs)
    self.correct_pred_as_pos_label = correct_pred_as_pos_label

  def update_state(self, y_true: Sequence[float], y_pred: Sequence[float],
                   confidence: Sequence[float],k, **kwargs: Mapping[str,
                                                                  Any]) -> None:
    """Updates confidence versus accuracy AUC statistics.

    Args:
      y_true: The ground truth labels. Shape (batch_size, ).
      y_pred: The predicted label indices. Must be integer valued predictions
        for label indices rather than the predictive probability. For the
        multi-label classification problem, y_pred is typically obtained as
        `tf.math.reduce_max(logits)`. Shape (batch_size, ).
      confidence: The confidence score where higher value indicates lower
        uncertainty. Values should be within [0, 1].
      **kwargs: Additional keyword arguments.
    """
    # Creates binary 'label' of prediction correctness, shape (batch_size, ).
    scores = tf.convert_to_tensor(confidence, dtype=self.dtype)
    labels = _compute_correct_predictions(
        y_true, y_pred, k=k, dtype=self.dtype)

    if not self.correct_pred_as_pos_label:
      # Use incorrect prediction as the positive class.
      # This is important since an accurate model has few incorrect predictions.
      # This results in label imbalance in the calibration AUC computation, and
      # can lead to overly optimistic results.
      scores = 1. - scores
      labels = 1. - labels

    # Updates confidence v.s. accuracy AUC statistic.
    super().update_state(y_true=labels, y_pred=scores, **kwargs)


class AbstainPrecision_Topk(tf.keras.metrics.Metric):
  """Implements the abstention precision metric.

  `AbstainPrecision_Topk` measures a model's uncertainty quantification ability
  by assuming the model has the ability to abstain (i.e., refuse to predict

  for an example due to low confidence). The abstention process can be done
  either per example, or globally over the dataset. In the latter case, the
  rejection decision is made by rejecting a pre-specified percentage of examples
  according to prediction confidence. This metric computes the percentage of
  correctly rejected examples, which is the percentage of incorrect predictions
  among all the abstained examples.

  The abstention decision is made under a budget, i.e., the model is only
  allowed to abstain a small amount of examples. For `AbstainPrecision_Topk`, this
  budget can be specified to be either a fixed number (`max_abstain_count`), or
  the fraction of the total dataset (`abstain_fraction`).

  It can be understood as the uncertainty analogy of 'Precision@TopK', where
  the ranking signal is the uncertainty score and the "label" is the prediction
  correctness. "TopK" is specified as the fraction of the total examples.

  For a AUC-style metric of the abstention policy, see `CalibrationAUC_Topk`.

  Attributes:
    abstain_fraction: The fraction of total examples to abstain.
    num_approx_bins: Number of histogram bins to use to approximate the
      distribution of the uncertainty score.
    max_abstain_count: The maximum number of total examples to abstain. If set,
      then the number of example to abstain is limited to be not larger than
      this value.
    binned_total_counts: The number of total examples in each bins of the
      uncertainty score historgram, shape (num_approx_bins, ).
    binned_correct_counts: The number of correct predictions in each bins of the
      uncertainty score historgram, shape (num_approx_bins, ).
    return_abstain_count: Whether to return the number of abstained examples
      rather than the precision.
  """

  # TODO(jereliu): Implement threshold-based abstention policy.

  def __init__(self,
               abstain_fraction: float = 0.01,
               num_approx_bins: int = 1000,
               max_abstain_count: Optional[int] = None,
               name: Optional[str] = None,
               dtype: Optional[tf.DType] = None,
               return_abstain_count: bool = False):
    """Constructs the abstention precision metric.

    Notice that `abstain_fraction` and `max_abstain_count` interact
    (i.e. the number abstained is the minimum of the two numbers defined
    by `abstain_fraction` and `max_abstain_count`).

    Args:
      abstain_fraction: The fraction of total examples to abstain. A float value
        between [0, 1].
      num_approx_bins: (Optional) Number of histogram bins to use to approximate
        the distribution of the uncertainty score.
      max_abstain_count: The maximum number of total examples to abstain. If
        set, then the number of example to abstain is limited to be not larger
        than this value.
      name: (Optional) Name of this metric.
      dtype: (Optional) Data type. Must be floating-point.
      return_abstain_count: (Optional) Whether to return the number of abstained
        examples rather than the precision. For debugging purpose only, default
        to False.
    """
    super().__init__(name=name, dtype=dtype)

    if max_abstain_count is not None:
      max_abstain_count = tf.cast(max_abstain_count, dtype=self.dtype)

    self.abstain_fraction = tf.cast(abstain_fraction, dtype=self.dtype)
    self.num_approx_bins = num_approx_bins
    self.max_abstain_count = max_abstain_count
    self.return_abstain_count = return_abstain_count

    # Initializes histogram for confidence score distributions.
    self.binned_total_counts = self.add_weight(
        "binned_total_counts",
        shape=(num_approx_bins,),
        initializer=tf.zeros_initializer,
        dtype=self.dtype)
    self.binned_correct_counts = self.add_weight(
        "binned_correct_counts",
        shape=(num_approx_bins,),
        initializer=tf.zeros_initializer,
        dtype=self.dtype)

  def update_state(self,
                   y_true: Sequence[float],
                   y_pred: Sequence[float],
                   confidence: Sequence[float],
                   k,
                   sample_weight: Optional[Sequence[float]] = None) -> None:
    """Updates confidence and accuracy statistics.

    Args:
      y_true: The ground truth labels. Shape (batch_size, ).
      y_pred: The predicted labels. Must be integer valued predictions for label
        index rather than the predictive probability. For multi-label
        classification problems, `y_pred` is typically obtained as
        `tf.math.reduce_max(logits)`. Shape (batch_size, ).
      confidence: The confidence score where higher value indicates lower
        uncertainty. Values should be within [0, 1].
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        Tensor whose rank is either 0, or the same rank as `y_true`, and must be
        broadcastable to `y_true`.
    """
    batch_size = tf.shape(y_true)[0]

    # Preprocess `confidence` and `sample_weight` tensors.
    confidence = tf.cast(tf.convert_to_tensor(confidence), dtype=self.dtype)
    confidence = tf.reshape(confidence, shape=(batch_size,))

    if sample_weight is not None:
      sample_weight = tf.convert_to_tensor(sample_weight)
      sample_weight = tf.reshape(sample_weight, shape=(batch_size,))
      sample_weight = tf.cast(sample_weight, dtype=self.dtype)
    else:
      sample_weight = tf.ones((batch_size,), dtype=self.dtype)

    # Computes correct predictions.
    correct_preds = _compute_correct_predictions(
        y_true, y_pred,k=k, dtype=self.dtype)
    correct_preds_weighted = correct_preds * sample_weight

    # Computes batch-specific histogram statistics for confidence score.
    batch_bin_indices = tf.histogram_fixed_width_bins(
        confidence,
        tf.constant([0., 1.], self.dtype),
        nbins=self.num_approx_bins)
    batch_total_counts = tf.math.unsorted_segment_sum(
        data=sample_weight,
        segment_ids=batch_bin_indices,
        num_segments=self.num_approx_bins)
    batch_correct_counts = tf.math.unsorted_segment_sum(
        data=correct_preds_weighted,
        segment_ids=batch_bin_indices,
        num_segments=self.num_approx_bins)

    self.binned_total_counts.assign_add(batch_total_counts)
    self.binned_correct_counts.assign_add(batch_correct_counts)

  def result(self):
    """Computes the abstention precision."""
    # TODO(jereliu): Incorporate uncertainty threshold into the computation of
    # `total_count_abstained`.

    # Computes the number of examples to abstain.
    total_counts = tf.reduce_sum(self.binned_total_counts)
    total_count_abstained = tf.floor(total_counts * self.abstain_fraction)

    if self.max_abstain_count is not None:
      total_count_abstained = tf.reduce_min(
          [total_count_abstained, self.max_abstain_count])

    if self.return_abstain_count:
      return total_count_abstained

    # Computes the correct predictions among the examples to be abstained.
    correct_predictions_abstained = self._compute_correct_predictions_abstained(
        total_count_abstained)

    return tf.math.divide_no_nan(
        total_count_abstained - correct_predictions_abstained,
        total_count_abstained)

  def _compute_correct_predictions_abstained(
      self, total_count_abstained: int) -> tf.Tensor:
    """Approximates the number of correct predictions in abstained examples.

    Args:
      total_count_abstained: Maximum number of examples to abstain.

    Returns:
      A scalar Tensor of self.dtype.
    """
    # Computes unique cumulative counts for non-empty bins.
    non_empty_bin_mask = self.binned_total_counts > 0.
    binned_total_counts_masked = tf.boolean_mask(self.binned_total_counts,
                                                 non_empty_bin_mask)
    binned_correct_counts_masked = tf.boolean_mask(self.binned_correct_counts,
                                                   non_empty_bin_mask)
    cumulative_total_counts = tf.cumsum(binned_total_counts_masked)

    # Finds the index of the bin whose cumulative count first exceeds the
    # `total_count_abstained`.
    final_bin_index = tf.argmax(
        cumulative_total_counts >= total_count_abstained, output_type=tf.int32)

    # Computes the example counts before the final bin.
    total_count_before_final_bin = tf.cond(
        final_bin_index > 0,
        lambda: cumulative_total_counts[final_bin_index - 1], lambda: 0.)
    correct_count_before_final_bin = tf.cond(
        final_bin_index > 0,
        lambda: tf.reduce_sum(binned_correct_counts_masked[:final_bin_index]),
        lambda: 0.)

    # Approximates the correct count for the final bin.
    total_count_abstained_final_bin = (
        total_count_abstained - total_count_before_final_bin)
    accuracy_final_bin = (
        binned_correct_counts_masked[final_bin_index] /
        binned_total_counts_masked[final_bin_index])
    correct_count_final_bin = (
        accuracy_final_bin * total_count_abstained_final_bin)

    return correct_count_before_final_bin + correct_count_final_bin

  def reset_states(self):
    """Resets all of the metric state variables.

    This function is called between epochs/steps,
    when a metric is evaluated during training.
    """
    vars_to_reset = (self.binned_total_counts, self.binned_correct_counts)
    tf.keras.backend.batch_set_value([(v, [
        0.,
    ] * self.num_approx_bins) for v in vars_to_reset])


class AbstainRecall_Topk(AbstainPrecision_Topk):
  """Implements the abstention recall metric.

  Different from `AbstainPrecision_Topk`, `AbstainRecall_Topk` computes the percentage of
  correctly abstained examples among all the incorrect predictions that **could
  have been abstained**.

  As a result, assume the model abstains according to confidence and under the
  budget. The numerator is the total number of incorrect predictions among the
  abstained examples, and the denominator is the total incorrect predictions
  made by the model.
  """

  def result(self):
    """Computes the abstention recall."""
    # TODO(jereliu): Incorporate uncertainty threshold into the computation of
    # `total_count_abstained`.

    # Computes numerator: the number of successfully abstained examples.
    total_counts = tf.reduce_sum(self.binned_total_counts)
    total_count_abstained = tf.floor(total_counts * self.abstain_fraction)
    correct_predictions_abstained = self._compute_correct_predictions_abstained(
        total_count_abstained)
    incorrect_predictions_abstained = (
        total_count_abstained - correct_predictions_abstained)

    # Computes denominator: the total number of incorrect predictions.
    correct_counts = tf.reduce_sum(self.binned_correct_counts)
    incorrect_counts = total_counts - correct_counts

    if self.return_abstain_count:
      return incorrect_counts

    return incorrect_predictions_abstained / incorrect_counts
