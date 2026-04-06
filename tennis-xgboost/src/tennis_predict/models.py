"""XGBoost model training and report generation.

Single model type: XGBoost. No MLP, no RandomForest, no DecisionTree.

Note: evaluation/scoring logic lives in evaluate.py (immutable) to prevent
auto-research agents from gaming validation scores via post-hoc adjustments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from tennis_predict.config import (
    CATEGORICAL_FEATURES,
    CUTOFF_DATES_BY_TOUR,
    DEFAULT_RANDOM_STATE,
    META_COLUMNS,
    TARGET_COLUMN,
)
from tennis_predict.data import date_slice, event_slice
from tennis_predict.evaluate import (
    evaluate_model,
    extract_feature_importances,
    markdown_report,
    per_tournament_accuracy,
)


DEFAULT_XGBOOST_PARAMS: dict[str, float | int] = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
}

XGBOOST_PARAMS_BY_TOUR: dict[str, dict[str, float | int]] = {
    # ATP benefitted from a slower, deeper model with a small split penalty.
    "atp": {
        "n_estimators": 900,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
        "min_child_weight": 5,
        "gamma": 0.15,
    },
    # WTA responded better to a denser depth-4 model with mild L1 regularization.
    "wta": {
        "n_estimators": 800,
        "max_depth": 4,
        "learning_rate": 0.04,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
    },
}

ENSEMBLE_SPECS_BY_TOUR: dict[str, tuple[tuple[float, dict[str, float | int]], ...]] = {
    "atp": (
        (1.0, XGBOOST_PARAMS_BY_TOUR["atp"]),
    ),
    "wta": (
        (0.65, XGBOOST_PARAMS_BY_TOUR["wta"]),
        (
            0.35,
            {
                "n_estimators": 650,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.95,
                "colsample_bytree": 0.85,
                "reg_lambda": 1.5,
                "min_child_weight": 4,
                "reg_alpha": 0.2,
                "gamma": 0.05,
            },
        ),
    ),
}


RELIABILITY_BASES: tuple[str, ...] = (
    "career_matches",
    "surface_matches",
    "season_matches",
    "season_surface_matches",
    "tourney_matches",
    "matches_last_14_days",
    "matches_last_30_days",
    "days_since_last_match",
    "days_since_last_surface_match",
    "days_since_last_tourney_match",
)


FEATURE_PREFIX_EXCLUSIONS_BY_TOUR: dict[str, tuple[str, ...]] = {
    # Entry-status flags helped the WTA slice but added noise on ATP.
    "atp": ("entry_",),
    # Current-season, streak, and handedness history features lifted ATP but
    # added noise on the smaller WTA slice.
    "wta": (
        "season_",
        "current_streak",
        "surface_streak",
        "hand_",
        "vs_opp_hand_",
        "surface_vs_opp_hand_",
    ),
}

FEATURE_EXCLUSIONS_BY_TOUR: dict[str, tuple[str, ...]] = {
    # Height is sparse and noisy on WTA; the prompt explicitly calls this out.
    "wta": ("height_diff",),
}


@dataclass(frozen=True)
class WeightedEstimatorSpec:
    """Weighted XGBoost estimator definition for soft-voting ensembles."""

    weight: float
    params: dict[str, float | int]


@dataclass(frozen=True)
class SegmentBlendSpec:
    """Ordered segment specialist blended against the global model."""

    column: str
    value: str
    global_weight: float
    params: dict[str, float | int]


@dataclass(frozen=True)
class TemporalBlendSpec:
    """Blend the full-history model with a recent-era global model."""

    recent_start: str
    recent_weight: float
    min_recent_rows: int


SEGMENT_BLEND_SPECS_BY_TOUR: dict[str, tuple[SegmentBlendSpec, ...]] = {
    # Keep the low-tier ATP specialists that still add signal; the clay
    # specialist no longer improves the global stack once reliability features
    # are in place.
    "atp": (
        SegmentBlendSpec(
            column="tourney_level",
            value="250",
            global_weight=0.775,
            params={
                "n_estimators": 1000,
                "max_depth": 3,
                "learning_rate": 0.03,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.5,
                "min_child_weight": 4,
                "gamma": 0.05,
            },
        ),
        SegmentBlendSpec(
            column="tourney_level",
            value="A",
            global_weight=0.0,
            params={
                "n_estimators": 800,
                "max_depth": 3,
                "learning_rate": 0.04,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.5,
                "min_child_weight": 3,
                "gamma": 0.0,
            },
        ),
    ),
    # WTA responds best to a mostly hard-court specialist with a small amount
    # of global context retained, plus only a light blend on the tourney-level
    # I subset.
    "wta": (
        SegmentBlendSpec(
            column="surface",
            value="Hard",
            global_weight=0.1,
            params={
                "n_estimators": 900,
                "max_depth": 4,
                "learning_rate": 0.03,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 1.5,
                "reg_alpha": 0.15,
                "min_child_weight": 5,
                "gamma": 0.05,
            },
        ),
        SegmentBlendSpec(
            column="tourney_level",
            value="I",
            global_weight=0.1,
            params={
                "n_estimators": 900,
                "max_depth": 3,
                "learning_rate": 0.035,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.5,
                "reg_alpha": 0.2,
                "min_child_weight": 4,
                "gamma": 0.0,
            },
        ),
    ),
}


TEMPORAL_BLEND_SPECS_BY_TOUR: dict[str, TemporalBlendSpec] = {
    # ATP gets a small lift from a recent-era correction once the tour's
    # faster post-2019 landscape is separated from the full 1985-2025 history.
    "atp": TemporalBlendSpec(
        recent_start="2019-01-01",
        recent_weight=0.2,
        min_recent_rows=15000,
    ),
    # WTA benefits from a light recent-era correction because the style and
    # field depth have shifted materially across the full 1985-2025 history.
    "wta": TemporalBlendSpec(
        recent_start="2017-01-01",
        recent_weight=0.35,
        min_recent_rows=15000,
    ),
}


class WeightedXGBoostEnsemble(BaseEstimator, ClassifierMixin):
    """Shared-preprocessor weighted ensemble of XGBoost classifiers."""

    def __init__(
        self,
        preprocessor: ColumnTransformer,
        estimators: tuple[WeightedEstimatorSpec, ...],
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.preprocessor = preprocessor
        self.estimators = estimators
        self.random_state = random_state

    def fit(self, x: pd.DataFrame, y: pd.Series) -> WeightedXGBoostEnsemble:
        """Fit the shared preprocessor once, then train each weighted model."""
        self.preprocessor_ = clone(self.preprocessor)
        transformed = self.preprocessor_.fit_transform(x)
        self.estimators_ = []
        weights = []
        for spec in self.estimators:
            estimator = build_xgboost_classifier(
                self.random_state,
                params_override=spec.params,
            )
            estimator.fit(transformed, y)
            self.estimators_.append(estimator)
            weights.append(spec.weight)
        self.weights_ = np.asarray(weights, dtype=float)
        self.weights_ = self.weights_ / self.weights_.sum()
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Average per-class probabilities across ensemble members."""
        transformed = self.preprocessor_.transform(x)
        probabilities = np.stack(
            [estimator.predict_proba(transformed) for estimator in self.estimators_],
            axis=0,
        )
        return np.tensordot(self.weights_, probabilities, axes=(0, 0))

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Predict labels using a 0.5 threshold on the averaged probability."""
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    @property
    def primary_estimator_(self) -> Any:
        """Return the highest-weight model for diagnostics and importances."""
        return self.estimators_[int(np.argmax(self.weights_))]


class SegmentBlendModel(BaseEstimator, ClassifierMixin):
    """Blend a global model with ordered segment-specific specialists."""

    def __init__(
        self,
        tour: str,
        specs: tuple[SegmentBlendSpec, ...],
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.tour = tour
        self.specs = specs
        self.random_state = random_state

    def fit(self, x: pd.DataFrame, y: pd.Series) -> SegmentBlendModel:
        """Fit the shared global model plus any configured segment specialists."""
        self.global_model_ = build_global_model(
            x,
            tour=self.tour,
            random_state=self.random_state,
        )
        self.global_model_.fit(x, y)
        self.segment_models_: list[tuple[SegmentBlendSpec, Pipeline]] = []

        for spec in self.specs:
            segment_mask = x[spec.column] == spec.value
            if not segment_mask.any():
                continue
            segment_x = x.loc[segment_mask]
            segment_y = y.loc[segment_mask]
            specialist = build_single_xgboost_pipeline(
                segment_x,
                self.random_state,
                params_override=spec.params,
                tour=self.tour,
            )
            specialist.fit(segment_x, segment_y)
            self.segment_models_.append((spec, specialist))
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Blend global probabilities with matching segment specialists."""
        probabilities = self.global_model_.predict_proba(x).astype(float, copy=True)
        for spec, model in self.segment_models_:
            segment_mask = (x[spec.column] == spec.value).to_numpy()
            if not segment_mask.any():
                continue
            segment_probs = model.predict_proba(x.loc[segment_mask])[:, 1].astype(float)
            probabilities[segment_mask, 1] = (
                spec.global_weight * probabilities[segment_mask, 1]
                + (1.0 - spec.global_weight) * segment_probs
            )
            probabilities[segment_mask, 0] = 1.0 - probabilities[segment_mask, 1]
        return probabilities

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Predict labels using a 0.5 threshold on blended probabilities."""
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    @property
    def primary_model_(self) -> Pipeline | WeightedXGBoostEnsemble:
        """Use the global model for diagnostics and feature importances."""
        return self.global_model_


class TemporalBlendModel(BaseEstimator, ClassifierMixin):
    """Blend the base full-history model with a recent-era global model."""

    def __init__(
        self,
        tour: str,
        recent_indices: pd.Index,
        recent_weight: float,
        min_recent_rows: int,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.tour = tour
        self.recent_indices = tuple(int(index) for index in recent_indices)
        self.recent_weight = recent_weight
        self.min_recent_rows = min_recent_rows
        self.random_state = random_state

    def fit(self, x: pd.DataFrame, y: pd.Series) -> TemporalBlendModel:
        """Fit the full-history model plus a recent-era correction model."""
        self.full_model_ = build_base_model(
            x,
            tour=self.tour,
            random_state=self.random_state,
        )
        self.full_model_.fit(x, y)

        recent_mask = x.index.isin(self.recent_indices)
        self.recent_model_: Pipeline | WeightedXGBoostEnsemble | None = None
        if int(recent_mask.sum()) < self.min_recent_rows:
            return self

        recent_x = x.loc[recent_mask]
        recent_y = y.loc[recent_mask]
        self.recent_model_ = build_global_model(
            recent_x,
            tour=self.tour,
            random_state=self.random_state + 17,
        )
        self.recent_model_.fit(recent_x, recent_y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Blend the recent-era model into the full-history probabilities."""
        probabilities = self.full_model_.predict_proba(x).astype(float, copy=True)
        if self.recent_model_ is None:
            return probabilities

        recent_probabilities = self.recent_model_.predict_proba(x)[:, 1].astype(float)
        probabilities[:, 1] = (
            (1.0 - self.recent_weight) * probabilities[:, 1]
            + self.recent_weight * recent_probabilities
        )
        probabilities[:, 0] = 1.0 - probabilities[:, 1]
        return probabilities

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Predict labels using a 0.5 threshold on blended probabilities."""
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    @property
    def primary_model_(self) -> Pipeline | WeightedXGBoostEnsemble | SegmentBlendModel:
        """Expose the full-history model for diagnostics and importances."""
        return self.full_model_


def build_xgboost_classifier(
    random_state: int,
    tour: str | None = None,
    params_override: dict[str, float | int] | None = None,
) -> Any:
    """Create an XGBClassifier with sensible defaults."""
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError(
            "XGBoost could not be loaded. On macOS the wheel needs libomp. "
            "Install it with `brew install libomp` or point `DYLD_LIBRARY_PATH` "
            "at a directory containing `libomp.dylib`."
        ) from exc

    params = DEFAULT_XGBOOST_PARAMS.copy()
    if tour is not None:
        params.update(XGBOOST_PARAMS_BY_TOUR.get(tour.lower(), {}))
    if params_override is not None:
        params.update(params_override)

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        **params,
    )


def recover_player_values(
    frame: pd.DataFrame,
    base: str,
) -> tuple[pd.Series | None, pd.Series | None]:
    """Reconstruct player A/B values from diff/sum feature pairs."""
    diff_col = f"{base}_diff"
    sum_col = f"{base}_sum"
    if diff_col not in frame.columns or sum_col not in frame.columns:
        return None, None
    diff = frame[diff_col]
    total = frame[sum_col]
    return (total + diff) / 2.0, (total - diff) / 2.0


def add_reliability_features(
    frame: pd.DataFrame,
    bases: tuple[str, ...],
) -> None:
    """Add min/max/imbalance features from diff/sum reliability bases."""
    for base in bases:
        diff_col = f"{base}_diff"
        sum_col = f"{base}_sum"
        if diff_col not in frame.columns or sum_col not in frame.columns:
            continue
        abs_diff = frame[diff_col].abs()
        sum_abs = frame[sum_col].abs()
        prefix = f"reliability_{base}"
        frame[f"{prefix}_min"] = (frame[sum_col] - abs_diff) / 2.0
        frame[f"{prefix}_max"] = (frame[sum_col] + abs_diff) / 2.0
        frame[f"{prefix}_imbalance"] = abs_diff / (sum_abs + 1.0)


def add_wta_fatigue_features(frame: pd.DataFrame) -> None:
    """Add workload and surface-transition composites for WTA."""
    rest_a, rest_b = recover_player_values(frame, "days_since_last_match")
    if rest_a is None or rest_b is None:
        return

    for window in (14, 30):
        matches_a, matches_b = recover_player_values(frame, f"matches_last_{window}_days")
        minutes_a, minutes_b = recover_player_values(frame, f"minutes_last_{window}_days")
        if matches_a is None or matches_b is None or minutes_a is None or minutes_b is None:
            continue
        frame[f"match_density_{window}_diff"] = (
            matches_a / (rest_a + 3.0) - matches_b / (rest_b + 3.0)
        )
        frame[f"minute_density_{window}_diff"] = (
            minutes_a / (rest_a + 3.0) - minutes_b / (rest_b + 3.0)
        )

    surface_rest_a, surface_rest_b = recover_player_values(
        frame, "days_since_last_surface_match"
    )
    if surface_rest_a is None or surface_rest_b is None:
        return
    frame["surface_switch_gap_diff"] = (
        (surface_rest_a - rest_a) - (surface_rest_b - rest_b)
    )


def augment_derived_features(
    frame: pd.DataFrame,
    tour: str | None = None,
) -> pd.DataFrame:
    """Add lightweight derived features that proved tour-specific on validation."""
    augmented = frame.copy()
    tour_key = (tour or "").lower()

    if tour_key in {"atp", "wta"}:
        add_reliability_features(augmented, RELIABILITY_BASES)
    if tour_key == "wta":
        add_wta_fatigue_features(augmented)
    return augmented


def feature_columns(frame: pd.DataFrame, tour: str | None = None) -> list[str]:
    """Return all columns that are not meta/label columns."""
    meta = set(META_COLUMNS)
    tour_key = (tour or "").lower()
    excluded_prefixes = FEATURE_PREFIX_EXCLUSIONS_BY_TOUR.get(tour_key, ())
    excluded_columns = set(FEATURE_EXCLUSIONS_BY_TOUR.get(tour_key, ()))
    return [
        column for column in frame.columns
        if column not in meta
        and column not in excluded_columns
        and not any(column.startswith(prefix) for prefix in excluded_prefixes)
    ]


def numeric_columns(frame: pd.DataFrame, tour: str | None = None) -> list[str]:
    """Return numeric feature columns (excluding categoricals)."""
    feature_set = set(feature_columns(frame, tour=tour))
    return [
        column for column in frame.columns
        if column in feature_set and column not in CATEGORICAL_FEATURES
    ]


def build_preprocessor(frame: pd.DataFrame, tour: str | None = None) -> ColumnTransformer:
    """Build sklearn preprocessor: median imputation for numeric, one-hot for categorical."""
    numeric_features = numeric_columns(frame, tour=tour)
    feature_set = set(feature_columns(frame, tour=tour))
    categorical_features = [
        column for column in CATEGORICAL_FEATURES if column in feature_set
    ]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_single_xgboost_pipeline(
    frame: pd.DataFrame,
    random_state: int,
    params_override: dict[str, float | int] | None = None,
    tour: str | None = None,
) -> Pipeline:
    """Build a single preprocessor + XGBoost pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(frame, tour=tour)),
            (
                "model",
                build_xgboost_classifier(
                    random_state,
                    params_override=params_override,
                ),
            ),
        ]
    )


def build_global_model(
    frame: pd.DataFrame,
    tour: str | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline | WeightedXGBoostEnsemble:
    """Build the existing global model stack for a tour."""
    preprocessor = build_preprocessor(frame, tour=tour)
    if tour is None:
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", build_xgboost_classifier(random_state, tour=tour)),
            ]
        )

    ensemble_specs = tuple(
        WeightedEstimatorSpec(weight=weight, params=params.copy())
        for weight, params in ENSEMBLE_SPECS_BY_TOUR.get(
            tour.lower(),
            ((1.0, XGBOOST_PARAMS_BY_TOUR.get(tour.lower(), DEFAULT_XGBOOST_PARAMS)),),
        )
    )
    if len(ensemble_specs) == 1:
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    build_xgboost_classifier(
                        random_state,
                        params_override=ensemble_specs[0].params,
                    ),
                ),
            ]
        )
    return WeightedXGBoostEnsemble(
        preprocessor=preprocessor,
        estimators=ensemble_specs,
        random_state=random_state,
    )


def build_base_model(
    frame: pd.DataFrame,
    tour: str | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline | WeightedXGBoostEnsemble | SegmentBlendModel:
    """Build the configured base model stack for a tour."""
    if tour is not None:
        segment_specs = SEGMENT_BLEND_SPECS_BY_TOUR.get(tour.lower(), ())
        if segment_specs:
            return SegmentBlendModel(
                tour=tour,
                specs=segment_specs,
                random_state=random_state,
            )
    return build_global_model(frame, tour=tour, random_state=random_state)


def build_model(
    frame: pd.DataFrame,
    tour: str | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    recent_indices: pd.Index | None = None,
) -> Pipeline | WeightedXGBoostEnsemble | SegmentBlendModel | TemporalBlendModel:
    """Build the configured model stack for a tour."""
    if tour is not None and recent_indices is not None:
        temporal_spec = TEMPORAL_BLEND_SPECS_BY_TOUR.get(tour.lower())
        if temporal_spec is not None:
            return TemporalBlendModel(
                tour=tour,
                recent_indices=recent_indices,
                recent_weight=temporal_spec.recent_weight,
                min_recent_rows=temporal_spec.min_recent_rows,
                random_state=random_state,
            )
    return build_base_model(frame, tour=tour, random_state=random_state)


def split_frame(
    features: pd.DataFrame,
    cutoff_date: str,
    test_event: str,
    test_year: int,
    test_mode: str = "event",
    tour: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split features into train (before cutoff) and test.

    Args:
        features: full feature DataFrame.
        cutoff_date: date string (e.g. '2025-12-31'). Overridden by per-tour
            cutoff from CUTOFF_DATES_BY_TOUR when tour is provided.
        test_event: tournament name for event-based test mode.
        test_year: year for event-based test mode.
        test_mode: 'date' = all matches after cutoff; 'event' = single tournament.
        tour: optional tour name ('atp' or 'wta'). When provided and test_mode
            is 'date', uses the per-tour cutoff date from config for balanced
            test set sizes.

    Returns:
        (train, test) DataFrames.
    """
    # Use per-tour cutoff when available (date mode only)
    effective_cutoff = cutoff_date
    if tour is not None and test_mode == "date":
        effective_cutoff = CUTOFF_DATES_BY_TOUR.get(tour.lower(), cutoff_date)

    cutoff = pd.Timestamp(effective_cutoff)
    train = features.loc[features["match_date"] <= cutoff].copy()

    if test_mode == "date":
        test = date_slice(features, effective_cutoff)
    else:
        test = event_slice(features, test_event, test_year)

    if train.empty:
        raise ValueError("Training split is empty. Check the cutoff date.")
    if test.empty:
        raise ValueError(
            f"Test split is empty. mode={test_mode}, cutoff={effective_cutoff}, "
            f"event={test_event}, year={test_year}"
        )
    return train, test


def train_and_report(
    features: pd.DataFrame,
    models_dir: Path,
    tour: str,
    cutoff_date: str,
    test_event: str,
    test_year: int,
    test_mode: str = "event",
) -> dict[str, Any]:
    """Train XGBoost, evaluate, save model + metrics + report.

    Returns dict with summary, scores, and feature_importances.
    """
    augmented_features = augment_derived_features(features, tour=tour)
    train, test = split_frame(
        augmented_features,
        cutoff_date=cutoff_date,
        test_event=test_event,
        test_year=test_year,
        test_mode=test_mode,
        tour=tour,
    )
    feat_cols = feature_columns(train)
    x_train = train[feat_cols]
    y_train = train[TARGET_COLUMN].astype(int)
    x_test = test[feat_cols]
    y_test = test[TARGET_COLUMN].astype(int)

    models_dir.mkdir(parents=True, exist_ok=True)

    recent_indices = None
    if tour is not None:
        temporal_spec = TEMPORAL_BLEND_SPECS_BY_TOUR.get(tour.lower())
        if temporal_spec is not None:
            recent_start = pd.Timestamp(temporal_spec.recent_start)
            recent_indices = train.index[train["match_date"] >= recent_start]

    model = build_model(x_train, tour=tour, recent_indices=recent_indices)
    scores = evaluate_model(model, x_train, y_train, x_test, y_test)

    # Save model
    joblib.dump(model, models_dir / "model.joblib")

    # Feature importances
    importances = extract_feature_importances(model)
    if importances is not None:
        importances.to_csv(models_dir / "feature_importance.csv", index=False)

    # Predictions
    probabilities = model.predict_proba(x_test)[:, 1].astype(float)
    prediction_frame = test[
        ["match_id", "match_date", "tourney_name", "surface", "round",
         "player_a_name", "player_b_name", "winner_name", "label"]
    ].copy()
    prediction_frame["prob_player_a_wins"] = probabilities
    prediction_frame["predicted_label"] = (probabilities >= 0.5).astype(int)
    prediction_frame["predicted_winner"] = np.where(
        prediction_frame["predicted_label"] == 1,
        prediction_frame["player_a_name"],
        prediction_frame["player_b_name"],
    )
    prediction_frame.to_csv(models_dir / "predictions.csv", index=False)

    # Per-tournament breakdown
    tourney_breakdown = per_tournament_accuracy(prediction_frame)
    if not tourney_breakdown.empty:
        tourney_breakdown.to_csv(models_dir / "per_tournament.csv", index=False)

    # Summary — use the effective cutoff (per-tour override if applicable)
    effective_cutoff = cutoff_date
    if tour is not None and test_mode == "date":
        effective_cutoff = CUTOFF_DATES_BY_TOUR.get(tour.lower(), cutoff_date)
    test_tournaments = int(test["tourney_name"].nunique()) if "tourney_name" in test.columns else 0
    summary = {
        "tour": tour,
        "cutoff_date": effective_cutoff,
        "test_event": test_event,
        "test_year": test_year,
        "test_mode": test_mode,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_tournaments": test_tournaments,
    }
    metrics_output = {"summary": summary, "scores": scores}
    (models_dir / "metrics.json").write_text(
        json.dumps(metrics_output, indent=2), encoding="utf-8"
    )
    (models_dir / "report.md").write_text(
        markdown_report(summary, scores, importances, tourney_breakdown), encoding="utf-8"
    )

    return {
        "summary": summary,
        "scores": scores,
        "feature_importances": importances,
        "per_tournament": tourney_breakdown,
    }
