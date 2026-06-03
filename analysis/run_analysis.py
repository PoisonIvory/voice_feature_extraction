"""End-to-end orchestrator for the voice-cycle analysis.

Single command: stage inputs, assemble the independent daily table, derive
phases + hormone features, run the discovery and association tiers, save every
result table, render the figures, and write the standalone report.

Usage:
    python -m analysis.run_analysis
"""

from __future__ import annotations

import pandas as pd

from analysis.cycle_voice import (
    associations,
    data_assembly,
    explore,
    figures,
    html_export,
    paths,
    phases,
    report,
    stage_external,
)
from analysis.cycle_voice.features import egemaps_feature_columns


def main() -> None:
    paths.ensure_output_dirs()
    stage_external.ensure()

    df = phases.derive(data_assembly.assemble())
    df.to_parquet(paths.ASSEMBLED_TABLE, index=False)

    coverage = data_assembly.coverage_manifest(df)
    phasedist = phases.phase_distribution(df)
    coverage.to_csv(paths.TABLES_DIR / "coverage_manifest.csv", index=False)
    data_assembly.data_dictionary(df).to_csv(paths.TABLES_DIR / "data_dictionary.csv", index=False)
    phasedist.to_csv(paths.TABLES_DIR / "phase_distribution.csv", index=False)

    features = egemaps_feature_columns(df)
    explore_res = explore.compute(df, features)
    assoc_res = associations.compute(df, features)

    for name, frame in {**explore_res, **assoc_res}.items():
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame.to_csv(paths.TABLES_DIR / f"{name}.csv", index=False)

    figs = figures.generate_all(df, explore_res, assoc_res)
    report.build(df, explore_res, assoc_res, figs, coverage, phasedist)
    html_path = html_export.build()

    print(f"Assembled table: {paths.ASSEMBLED_TABLE}")
    print(f"Result tables:   {paths.TABLES_DIR}")
    print(f"Figures:         {paths.FIGURES_DIR}  ({len(figs)} figures)")
    print(f"Report (md):     {paths.REPORT_FILE}")
    print(f"Report (html):   {html_path}")


if __name__ == "__main__":
    main()
