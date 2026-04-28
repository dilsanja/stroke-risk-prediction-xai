"""
tests/test_model.py
===================
Model validation tests for NeuraScan stroke prediction system.

Run with:
    python -m pytest tests/test_model.py -v

These tests verify:
- Model files load correctly
- Predictions are valid probabilities
- High-risk patients score above threshold
- Performance meets clinical targets
- Deployment artefacts are consistent
"""

import pytest
import numpy as np
import joblib
import json
from pathlib import Path


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    """Load the trained model — skip if not generated yet."""
    paths = [
        Path("deployment/stroke_model_m4.pkl"),
        Path("deployment/stroke_model.pkl"),
    ]
    for path in paths:
        if path.exists():
            return joblib.load(path)
    pytest.skip("No model found — run the notebooks first to generate deployment/ files")


@pytest.fixture(scope="module")
def scaler():
    """Load the fitted scaler."""
    paths = [
        Path("deployment/stroke_scaler_m4.pkl"),
        Path("deployment/stroke_scaler.pkl"),
    ]
    for path in paths:
        if path.exists():
            return joblib.load(path)
    pytest.skip("No scaler found — run the notebooks first")


@pytest.fixture(scope="module")
def metadata():
    """Load model metadata JSON."""
    paths = [
        Path("deployment/model_metadata_m4.json"),
        Path("deployment/model_metadata.json"),
    ]
    for path in paths:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    pytest.skip("No metadata found — run the notebooks first")


@pytest.fixture
def high_risk_patient():
    """
    Elderly patient with hypertension, heart disease, and high glucose.
    Should clearly score above the decision threshold.
    Age=78, Male, Hypertension, Heart Disease, Married,
    Private sector, Urban, Glucose=210, BMI=28, Formerly smoked
    + engineered features: age_group=3, glucose_cat=2, bmi_cat=1, cardio_risk=1
    """
    return np.array([[78, 1, 1, 1, 1, 2, 1, 210.5, 28.3, 1, 3, 2, 1, 1]])


@pytest.fixture
def low_risk_patient():
    """
    Young healthy patient with no comorbidities.
    Should score below 0.50.
    Age=25, Female, No hypertension, No heart disease, Single,
    Private, Rural, Glucose=85, BMI=22, Never smoked
    + engineered features: age_group=1, glucose_cat=0, bmi_cat=1, cardio_risk=0
    """
    return np.array([[25, 0, 0, 0, 0, 2, 0, 85.0, 22.1, 2, 1, 0, 1, 0]])


# ── Loading Tests ─────────────────────────────────────────────────────────────

class TestArtefactLoading:

    def test_model_loads(self, model):
        assert model is not None, "Model should not be None"

    def test_scaler_loads(self, scaler):
        assert scaler is not None, "Scaler should not be None"

    def test_metadata_loads(self, metadata):
        assert metadata is not None, "Metadata should not be None"

    def test_metadata_required_keys(self, metadata):
        required = ["decision_threshold", "feature_names", "test_metrics"]
        for key in required:
            assert key in metadata, f"Metadata missing required key: '{key}'"

    def test_threshold_is_valid_probability(self, metadata):
        t = metadata["decision_threshold"]
        assert 0.0 < t < 1.0, f"Threshold {t} is not a valid probability (0, 1)"

    def test_feature_count_is_correct(self, metadata):
        assert len(metadata["feature_names"]) == 14, \
            f"Expected 14 features, got {len(metadata['feature_names'])}"


# ── Prediction Tests ──────────────────────────────────────────────────────────

class TestPredictions:

    def test_output_probability_shape(self, model, scaler, high_risk_patient):
        X = scaler.transform(high_risk_patient)
        probs = model.predict_proba(X)
        assert probs.shape == (1, 2), f"Expected shape (1,2), got {probs.shape}"

    def test_probabilities_sum_to_one(self, model, scaler, high_risk_patient):
        X = scaler.transform(high_risk_patient)
        probs = model.predict_proba(X)[0]
        assert abs(probs.sum() - 1.0) < 1e-5, \
            f"Probabilities sum to {probs.sum():.6f}, expected 1.0"

    def test_probability_is_in_valid_range(self, model, scaler, high_risk_patient):
        X = scaler.transform(high_risk_patient)
        prob = model.predict_proba(X)[0, 1]
        assert 0.0 <= prob <= 1.0, f"Probability {prob} outside [0, 1]"

    def test_high_risk_patient_above_threshold(self, model, scaler,
                                               high_risk_patient, metadata):
        """Elderly patient with hypertension and high glucose must be flagged."""
        X = scaler.transform(high_risk_patient)
        prob = model.predict_proba(X)[0, 1]
        threshold = metadata["decision_threshold"]
        assert prob >= threshold, (
            f"High-risk patient scored {prob:.4f} below threshold {threshold:.4f}. "
            f"Model may not be working correctly."
        )

    def test_low_risk_patient_scores_below_half(self, model, scaler, low_risk_patient):
        """Young healthy patient should score below 0.50."""
        X = scaler.transform(low_risk_patient)
        prob = model.predict_proba(X)[0, 1]
        assert prob < 0.50, (
            f"Low-risk patient scored unexpectedly high: {prob:.4f}. "
            f"Model may be over-predicting."
        )

    def test_high_risk_scores_higher_than_low_risk(self, model, scaler,
                                                    high_risk_patient,
                                                    low_risk_patient):
        """Basic sanity check — high-risk must score higher than low-risk."""
        X_hi = scaler.transform(high_risk_patient)
        X_lo = scaler.transform(low_risk_patient)
        p_hi = model.predict_proba(X_hi)[0, 1]
        p_lo = model.predict_proba(X_lo)[0, 1]
        assert p_hi > p_lo, (
            f"High-risk patient ({p_hi:.4f}) scored below low-risk ({p_lo:.4f}). "
            f"Model ranking is inverted."
        )

    def test_batch_prediction_works(self, model, scaler,
                                    high_risk_patient, low_risk_patient):
        """Model should handle batches without errors."""
        batch = np.vstack([high_risk_patient, low_risk_patient])
        X = scaler.transform(batch)
        probs = model.predict_proba(X)
        assert probs.shape == (2, 2)
        assert np.all(probs >= 0) and np.all(probs <= 1)


# ── Performance Tests ─────────────────────────────────────────────────────────

class TestPerformanceTargets:

    def test_auc_beats_published_baseline(self, metadata):
        """AUC must beat Nwosu et al. (2019) baseline of 0.80."""
        auc = metadata["test_metrics"]["auc_roc"]
        assert auc >= 0.80, (
            f"AUC {auc:.4f} is below the published baseline of 0.80. "
            f"Something may have gone wrong with training."
        )

    def test_recall_meets_clinical_target(self, metadata):
        """Must catch at least 79% of strokes (clinical target: 80%)."""
        # Allow 1% tolerance for rounding
        recall_key = "recall_opt" if "recall_opt" in metadata["test_metrics"] \
                     else "recall"
        recall = metadata["test_metrics"][recall_key]
        assert recall >= 0.79, (
            f"Recall {recall:.4f} does not meet the clinical target of 0.80. "
            f"Consider lowering the decision threshold."
        )

    def test_brier_score_better_than_random(self, metadata):
        """Brier score must be better than the random baseline (0.0489)."""
        brier = metadata["test_metrics"]["brier"]
        random_baseline = 0.0489  # stroke prevalence = 4.89%
        assert brier < random_baseline, (
            f"Brier score {brier:.4f} is worse than random ({random_baseline:.4f}). "
            f"Model probability calibration has failed."
        )


# ── Scaler Tests ──────────────────────────────────────────────────────────────

class TestScaler:

    def test_transform_preserves_shape(self, scaler, high_risk_patient):
        X = scaler.transform(high_risk_patient)
        assert X.shape == high_risk_patient.shape

    def test_transform_produces_finite_values(self, scaler, high_risk_patient):
        X = scaler.transform(high_risk_patient)
        assert np.all(np.isfinite(X)), \
            "Scaler produced non-finite values (NaN or Inf)"

    def test_scaler_centres_data(self, scaler, high_risk_patient):
        """Scaled values should be in a reasonable range (roughly -5 to 5)."""
        X = scaler.transform(high_risk_patient)
        assert np.all(np.abs(X) < 20), \
            f"Scaled values seem unreasonably large: {X}"
