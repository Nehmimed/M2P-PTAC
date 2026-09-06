# M2P-PTAC

Official reproducibility repository accompanying the manuscript:

**M2P-PTAC: A Predict-then-Correct Framework for Water-Quality Forecasting Under Sensor Degradation**

**Authors:** Nadir Ehmimed, Mimoun Lamrini, Mohamed Yassin Chkouri, Abdellah Touhafi

## Overview

M2P-PTAC is a two-stage predict-then-correct framework for robust water-quality forecasting under sensor degradation.

Stage 1 provides the operational forecasting anchor. Stage 2 estimates bounded degradation-specific corrections. Correction estimation is separated from authorization: when the qualification, local-safety, or gating conditions are not satisfied, the applied correction is exactly zero and the Stage-1 anchor is retained.

This repository corresponds to the frozen experimental configuration used for the revised OJ-IES manuscript.

## Repository structure

* `src/`

  * frozen M2P-PTAC implementation
  * Study A/B protocol implementation
  * data-quality assessment (DQA) implementation
* `experiments/`

  * model selection
  * benchmark evaluation
  * Studies A, B, and C
  * model-side sensitivity analysis
  * reserved-seed confirmation
  * final artifact validation
* `slurm/`

  * SLURM launch scripts used for the reported VUB/VSC HPC runs
* `configs/`

  * selected architecture/hyperparameters
  * frozen operating configuration
* `reproducibility/`

  * canonical data manifest
  * paper-run manifest
  * release gate
  * Study A/B/C manifests and verdicts
  * confirmation manifest and verdict
* `results/`

  * field benchmark summaries
  * controlled-degradation confirmation results
  * Study A/B/C summaries
  * model-sensitivity summary
  * deployment-cost measurements
* `data/README.md`

  * data-availability statement and expected use of frozen inputs

## Frozen model selection

Model selection is performed exclusively on development data before reserved-seed confirmation.

The development procedure includes:

1. a 25-iteration architecture search;
2. validation-only selection of the base regularization parameter from
   `{1e-4, 5e-4, 1e-3, 5e-3, 2e-2}`;
3. controlled sudden-shift policy selection using development seeds
   `42, 43, 44`.

The resulting frozen configuration is provided in:

`configs/selected_configuration.json`

and the selected architecture parameters in:

`configs/best_hyperparameters.json`

## Reserved confirmation

The five reserved confirmation seeds are:

`88191, 26722, 35230, 45158, 73564`

These seeds are not used for architecture selection, regularization selection, policy selection, or model-side sensitivity analysis.

Under controlled multichannel sudden degradation, the frozen confirmation obtained a mean positive residual-error rescue of **34.39%**, with a 95% seed-level confidence interval of **32.55%–36.23%**.

Target-level confirmation summaries and seed-level results are available under:

`results/confirmation/`

## Safety stress test

Study B evaluates prospective correction-induced catastrophic events, defined specifically as an unprotected correction causing absolute error greater than twice the corresponding matched-anchor error.

Across 380 paired evaluations, the unprotected system produced 121 such events, whereas the complete safety/authorization stack produced zero.

The relevant artifacts are available under:

`results/study_b/` and `reproducibility/`.

This definition is specific to the prespecified experiment and should not be interpreted as a universal operational safety guarantee.

## Additional controlled studies

Study A evaluates sudden-shift rescue across target, sign, and intensity conditions and is retained as negative/boundary evidence.

Study C evaluates architectural and supervision ablations. The study does not establish individual indispensability of the architectural branches; its strongest mechanistic evidence concerns the degradation-specific correction target.

Artifacts are provided under:

`results/study_a/`
`results/study_c/`

## Field benchmark interpretation

Field benchmark summaries are provided for the nominal and naturally challenged deployments.

Because independent reference-quality measurements are not available for the naturally challenged field periods, these data are not used to claim physical sensor recalibration or direct correction accuracy. Controlled degradation with a known clean counterfactual provides the direct correction-efficacy evidence.

## Data availability

Raw field telemetry is subject to institutional and deployment-specific restrictions and is therefore not redistributed in this repository.

The released DQA implementation documents the preprocessing and fault-screening procedures. The paper experiment pipeline consumes frozen cleaned datasets together with timestamp-aligned observation masks, fault masks, target exclusions, and available fault-kind masks.

Before model-side experiments are run, the provided validation procedure checks and hashes the canonical frozen inputs.

Additional details are provided in:

`data/README.md`

## Environment

The Python dependencies used for the frozen release are listed in:

`requirements.txt`

The reported experiments were executed in the VUB/VSC HPC environment. The included SLURM files reproduce the original scheduler configuration; users running the code on another computing environment may need to adapt scheduler directives and filesystem paths.

## Running the experiment pipeline

After cloning the repository and providing the required frozen input data:

```bash
python -m pip install -r requirements.txt
```

The input-validation and experiment orchestration scripts are contained in:

`experiments/`
`slurm/`
`submit_all.sh`

For the original SLURM-based workflow, the experiment order is:

```text
canonical-input validation
    -> development-only model selection
    -> full benchmark
    -> Studies A / B / C
    -> model-side sensitivity
    -> reserved five-seed confirmation
    -> final artifact validation
```

The released scripts preserve the chronological split logic, development/confirmation seed separation, and frozen model-selection procedure used for the manuscript.

## Reproducibility and provenance

The release includes machine-readable provenance artifacts under:

`reproducibility/`

These include:

* canonical input manifest;
* paper-run manifest;
* frozen configuration provenance;
* Study A/B/C manifests;
* confirmation manifest and verdict;
* final release gate.

The frozen release gate completed with no reported release failures.

## Version corresponding to the manuscript

The manuscript-matched software release is:

**v1.0.0**

Repository:

`https://github.com/Nehmimed/M2P-PTAC`

## License

No license is currently granted beyond the rights provided by applicable copyright law. Please contact the authors regarding reuse beyond inspection and reproducibility assessment.
