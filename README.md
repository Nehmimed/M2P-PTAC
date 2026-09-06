# M2P-PTAC paper experiment pipeline

This package creates and runs a clean, reproducible paper evaluation.

## Frozen data-quality handoff

The pipeline does **not** run the DQA procedure.

`cleaned_data/` is treated as a frozen input. Before any SLURM job is
submitted, `validate_inputs.py` checks and hashes the canonical cleaned
datasets, observation masks, fault masks, target exclusions, and available
fault-kind masks. Nothing in this workflow writes to `cleaned_data/`.

## Fresh model selection

No previously selected architecture, base C, or operating threshold is reused.

The development-only selection stage performs:

1. a 25-iteration architecture search on the stable development data;
2. validation-only C selection from `{1e-4, 5e-4, 1e-3, 5e-3, 2e-2}`;
3. controlled sudden-shift policy selection over
   `k in {0.5, 1.0, 1.5, 2.0}` and `q in {4, 8, 16, 32}` using development
   seeds `42, 43, 44`.

The method topology is fixed before selection: the two-stage M2P-PTAC model
uses its CNN branch, attention branch, separate encoders, and independent
safety/authorization layers. Those are model-definition choices rather than
values selected using confirmation results.

The selected configuration is written to:

`selection/selected_configuration.json`

Every downstream benchmark, study, sensitivity arm, and confirmation loads
that file.

## Execution order

```text
canonical-input validation (read only)
    -> fresh model selection
    -> full benchmark
    -> Study A / Study B / Study C
    -> model-side sensitivity
    -> fresh five-seed confirmation
    -> final artifact manifest
```

The model-side sensitivity changes only residual scaling or target-variance
normalization and always uses the frozen canonical DQA masks. It does not run
IQR or stuck-window DQA variants.

## Fresh confirmation

The five reserved confirmation seeds are:

`88191, 26722, 35230, 45158, 73564`

They are not used by architecture selection, C selection, policy selection,
or the model-side sensitivity analysis. A project-wide single-use lock is
created before the first confirmation fit.

## Build the public release

From the project directory:

```bash
python m2p_paper_pipeline_fresh/build_release.py \
  --project-dir "$VSC_DATA/m2p_RR_Backup_delta" \
  --output-dir paper_release

bash m2p_paper_pipeline_fresh/install_launch_files.sh \
  "$VSC_DATA/m2p_RR_Backup_delta/paper_release"
```

The release builder verifies the audited `full_exp.py` SHA before producing
`paper_release/src/m2p_ptac.py`.

## Run the complete experiment chain

```bash
cd "$VSC_DATA/m2p_RR_Backup_delta"
./paper_release/submit_all.sh ojies_final
```

Results are written under:

`paper_results/ojies_final/`
