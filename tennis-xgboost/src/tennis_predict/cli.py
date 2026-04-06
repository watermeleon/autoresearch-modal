"""CLI interface for tennis prediction pipeline.

All commands accept --tour atp|wta to select the tour.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from tennis_predict.config import (
    DEFAULT_CUTOFF_DATE,
    DEFAULT_ELO_K,
    DEFAULT_START_YEAR,
    DEFAULT_TEST_EVENT,
    DEFAULT_TEST_MODE,
    DEFAULT_TEST_YEAR,
    RepoPaths,
)
from tennis_predict.data import (
    load_matches,
    load_matches_with_validation,
    load_rankings,
    sync_repo,
)
from tennis_predict.features import FeatureConfig, build_feature_frame
from tennis_predict.models import train_and_report


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="tennis-predict",
        description="Tennis match prediction pipeline (ATP/WTA)",
    )
    parser.add_argument(
        "--repo-root", type=Path, default=None, help="Override project root."
    )
    parser.add_argument(
        "--tour",
        choices=["atp", "wta"],
        default="wta",
        help="Tour to process (default: wta)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # sync-data
    sync_parser = subparsers.add_parser("sync-data", help="Clone/pull Sackmann data")
    sync_parser.add_argument("--refresh", action="store_true")

    # build-features
    build_parser_ = subparsers.add_parser(
        "build-features", help="Build feature parquet from raw data"
    )
    _add_feature_args(build_parser_)

    # train
    train_parser = subparsers.add_parser("train", help="Train XGBoost model")
    _add_feature_args(train_parser)
    _add_train_args(train_parser)

    # run-pipeline (sync + build + train)
    pipeline_parser = subparsers.add_parser(
        "run-pipeline", help="Full pipeline: sync, features, train"
    )
    _add_feature_args(pipeline_parser)
    _add_train_args(pipeline_parser)
    pipeline_parser.add_argument("--refresh-data", action="store_true")

    return parser


def _add_feature_args(parser: argparse.ArgumentParser) -> None:
    """Add feature-building arguments."""
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--elo-k", type=float, default=DEFAULT_ELO_K)


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    """Add training arguments."""
    parser.add_argument("--cutoff-date", default=DEFAULT_CUTOFF_DATE)
    parser.add_argument("--test-event", default=DEFAULT_TEST_EVENT)
    parser.add_argument("--test-year", type=int, default=DEFAULT_TEST_YEAR)
    parser.add_argument(
        "--test-mode",
        choices=["date", "event"],
        default=DEFAULT_TEST_MODE,
        help="Test split strategy: 'date' = all matches after cutoff; "
             "'event' = single tournament (default: %(default)s)",
    )


def _get_paths(args: argparse.Namespace) -> RepoPaths:
    """Build and ensure paths from CLI args."""
    paths = RepoPaths.from_root(args.repo_root, tour=args.tour)
    paths.ensure()
    return paths


def cmd_sync_data(paths: RepoPaths, refresh: bool) -> None:
    """Sync Sackmann data repo."""
    dest = sync_repo(paths.tour, paths.raw_repo_dir, refresh=refresh)
    print(f"Synced {paths.tour.upper()} data: {dest}")


def cmd_build_features(
    paths: RepoPaths, start_year: int, elo_k: float
) -> Path:
    """Build features and write parquet."""
    sync_repo(paths.tour, paths.raw_repo_dir, refresh=False)
    validation_dir = paths.data_dir / "validation"
    matches = load_matches_with_validation(
        paths.raw_repo_dir, paths.tour,
        validation_dir=validation_dir if validation_dir.exists() else None,
        start_year=start_year,
    )
    rankings = load_rankings(paths.raw_repo_dir, paths.tour, start_year=start_year)
    print(f"Loaded {len(matches)} matches, {len(rankings)} ranking entries")

    features = build_feature_frame(
        matches,
        config=FeatureConfig(k_factor=elo_k),
        rankings=rankings,
    )
    output_path = paths.features_parquet
    features.to_parquet(output_path, index=False)
    print(f"Wrote {output_path} ({len(features)} rows)")
    return output_path


def cmd_train(
    paths: RepoPaths,
    start_year: int,
    elo_k: float,
    cutoff_date: str,
    test_event: str,
    test_year: int,
    test_mode: str = "event",
) -> None:
    """Train model, building features if needed."""
    output_path = paths.features_parquet
    if not output_path.exists():
        output_path = cmd_build_features(paths, start_year=start_year, elo_k=elo_k)
    features = pd.read_parquet(output_path)

    result = train_and_report(
        features=features,
        models_dir=paths.models_dir,
        tour=paths.tour,
        cutoff_date=cutoff_date,
        test_event=test_event,
        test_year=test_year,
        test_mode=test_mode,
    )
    scores = result["scores"]
    summary = result["summary"]
    print(f"\n{paths.tour.upper()} XGBoost Results (mode={test_mode}):")
    print(f"  train: {summary['train_rows']} rows, test: {summary['test_rows']} rows ({summary['test_tournaments']} tournaments)")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")

    # Per-tournament breakdown
    per_tourney = result.get("per_tournament")
    if per_tourney is not None and not per_tourney.empty:
        print(f"\nPer-Tournament Accuracy:")
        for _, row in per_tourney.iterrows():
            print(f"  {row['tourney_name']}: {row['accuracy']:.3f} ({int(row['correct'])}/{int(row['matches'])})")

    # Print gate-compatible output on last line
    print(f"ROC_AUC={scores['roc_auc']:.4f}")


def cmd_pipeline(
    paths: RepoPaths,
    start_year: int,
    elo_k: float,
    cutoff_date: str,
    test_event: str,
    test_year: int,
    test_mode: str,
    refresh_data: bool,
) -> None:
    """Run full pipeline: sync, features, train."""
    if refresh_data:
        cmd_sync_data(paths, refresh=True)
    else:
        sync_repo(paths.tour, paths.raw_repo_dir, refresh=False)

    cmd_build_features(paths, start_year=start_year, elo_k=elo_k)
    cmd_train(
        paths=paths,
        start_year=start_year,
        elo_k=elo_k,
        cutoff_date=cutoff_date,
        test_event=test_event,
        test_year=test_year,
        test_mode=test_mode,
    )


def main() -> None:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()
    paths = _get_paths(args)

    if args.command == "sync-data":
        cmd_sync_data(paths, refresh=args.refresh)
    elif args.command == "build-features":
        cmd_build_features(paths, start_year=args.start_year, elo_k=args.elo_k)
    elif args.command == "train":
        cmd_train(
            paths=paths,
            start_year=args.start_year,
            elo_k=args.elo_k,
            cutoff_date=args.cutoff_date,
            test_event=args.test_event,
            test_year=args.test_year,
            test_mode=args.test_mode,
        )
    elif args.command == "run-pipeline":
        cmd_pipeline(
            paths=paths,
            start_year=args.start_year,
            elo_k=args.elo_k,
            cutoff_date=args.cutoff_date,
            test_event=args.test_event,
            test_year=args.test_year,
            test_mode=args.test_mode,
            refresh_data=args.refresh_data,
        )
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
