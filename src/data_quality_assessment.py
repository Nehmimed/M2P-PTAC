"""
================================================================================
 DATA QUALITY ASSESSMENT -- OJIES RESUBMISSION (26-OJIES-0226)
================================================================================
 This revision implements the reviewer-facing data provenance safeguards needed
 to distinguish source observations, detected upstream fills, sensor faults,
 and values reconstructed by this script.

 1. AN OBSERVATION MASK IS NOW EMITTED (interp_mask_<station>.csv). It records,
    per point and per parameter, whether the value was measured or produced by
    Stages 1-4 nullification followed by interpolation. After Stage 6 the two are
    indistinguishable in the cleaned CSV, so without this mask the benchmark
    cannot report observed-only metrics at all. full_exp.py consumes it via
    load_observation_mask().

 2. DETECTED UPSTREAM LINEAR FILLS ARE ACTUALLY REMOVED BEFORE RECONSTRUCTION.
    Earlier code demoted them in the observation mask but accidentally left the
    values in the model input.

 3. VALIDATION AND TEST FILLING ARE CAUSAL BY DEFAULT. They use forward-fill
    seeded only by earlier partitions. pandas time interpolation is deliberately
    not used in evaluation partitions because limit_direction='forward' can
    still use a future endpoint.

 4. COLUMN NAMES ARE CANONICALISED BEFORE RULE LOOKUP. This prevents the former
    batteryVoltage/batteryvoltage mismatch from silently bypassing guardrails.

 5. INPUT ORDER, duplicate timestamps, numeric coercion, daily timestamp
    normalisation, exclusion-file isolation, and mask alignment are validated.

 train_split remains the backward-compatible name for the test-start fraction
 (0.85). model_train_split is the true model-training fraction (0.70). Both are
 applied after full_exp.py's 72-row feature warm-up, so IQR thresholds never use
 validation or test observations and the row boundaries match the benchmark.
================================================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

class DataQualityAssessor:
    """
    Handles the validation, cleaning, and reporting of sensor data quality
    using a multi-stage, adaptive approach.
    """
    def __init__(self, input_dir, output_dir, plot_dir, train_split=0.85,
                 model_train_split=0.70,
                 causal_test_fill=True, fault_handling='preserve',
                 reported_stations=None, acknowledged_exclusions=None,
                 iqr_multiplier=2.5, output_suffix='',
                 normalize_daily_timestamps=True, prefilled_min_len=8,
                 feature_warmup=72):
        if not 0 < model_train_split < train_split < 1:
            raise ValueError(
                "Expected 0 < model_train_split < train_split < 1; "
                f"received {model_train_split=}, {train_split=}.")
        if iqr_multiplier <= 0:
            raise ValueError("iqr_multiplier must be positive.")
        if prefilled_min_len < 3:
            raise ValueError("prefilled_min_len must be at least 3.")
        if int(feature_warmup) != feature_warmup or feature_warmup < 0:
            raise ValueError("feature_warmup must be a non-negative integer.")

        # Backward compatibility: train_split was already used by run scripts
        # for the 0.85 test-start boundary. Keep the attribute while making the
        # true 0.70 model-training boundary explicit.
        self.train_split = float(train_split)
        self.test_start_split = float(train_split)
        self.model_train_split = float(model_train_split)
        # C-T4: the IQR multiplier was hard-coded at the call site (2.5) while
        # being a parameter of the function -- so the sensitivity of every
        # downstream number to this one choice was never measured. It is now a
        # constructor argument, and output_suffix lets a sweep write each
        # setting's masks to distinct filenames without overwriting the run
        # the benchmark actually consumes.
        self.iqr_multiplier = iqr_multiplier
        self.output_suffix = output_suffix
        self.normalize_daily_timestamps = bool(normalize_daily_timestamps)
        self.prefilled_min_len = int(prefilled_min_len)
        self.feature_warmup = int(feature_warmup)

        self.failures = []   # (station, kind, detail) accumulated by Gates A and B
        # ==================================================================
        # [FAULT-PRESERVING MODE]
        #
        # 'preserve' (default): Stages 1-4 LABEL degradation instead of
        #   deleting it. The reading stays in the data -- a deployed model
        #   sees stuck probes and drifted values, so the benchmark must too --
        #   and a fault mask records where each fault mode fired.
        #
        # 'nullify' : the original behaviour, retained only so the revision
        #   can quantify how much the old pipeline changed the data.
        #
        # WHY THIS IS THE DEFAULT. The paper is about sensors that degrade
        # through biofouling, calibration loss, and hardware faults. In the
        # first run, 92.5% of every point Stages 1-4 removed WAS one of those
        # signatures -- 1780 points to the stuck-value filter alone, which is
        # the biofouling detector. Interpolation then replaced each fault with
        # a smooth ramp. The 'challenged' stations were being handed to the
        # models already repaired, which is why a tree baseline performs well
        # on them and why the headline gains are so concentrated.
        # ==================================================================
        self.fault_handling = fault_handling
        # Stations whose numbers appear in the manuscript's results tables.
        # A gate failure inside one of these is a decision the author must make
        # and record; a failure outside them is bookkeeping.
        self.REPORTED_STATIONS = set(reported_stations or [])
        self.acknowledged_exclusions = set(acknowledged_exclusions or [])
        if fault_handling not in ('preserve', 'nullify'):
            raise ValueError("fault_handling must be 'preserve' or 'nullify'")

        # A reading can be out of range two different ways, and they are not
        # the same event. A plausible-but-out-of-spec value (temperature 45 C
        # where the guardrail max is 40) is a real measurement from a drifting
        # probe and must be preserved. A sentinel (-999, 6553.5) is not a
        # measurement at all; it cannot enter a scaler and is treated as
        # missing. Anything beyond this many range-widths outside the
        # guardrail is classified as a sentinel.
        self.sentinel_range_multiplier = 3.0
        # [R3] Causal forward-fill of validation/test partitions. See Stage 6.
        self.causal_test_fill = causal_test_fill
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.plot_dir = plot_dir
        
        # ==================================================================
        # THREE-TIER GUARDRAILS.
        #
        # The previous two-tier scheme (spec range + a multiplicative pad) had
        # two failure modes that both let impossible values through as
        # "plausible drift":
        #
        #   1. A parameter with no 'max' had NO upper bound at all. Dissolved
        #      oxygen was declared {'min': 0} only, so the 100000 mg/L sentinel
        #      at station 861513064189509 -- 66.8% of that column -- passed every
        #      stage untouched and was carried into the model as a measurement.
        #
        #   2. The pad scaled with the spec width, so temperature (range 55) got
        #      a pad of 3*55 = 165 and treated anything above -170 C as a
        #      plausible drifting reading. -100 C water is not plausible.
        #
        # So the bounds are now explicit and separate:
        #   hard_min/hard_max -- outside this a value is PHYSICALLY IMPOSSIBLE.
        #                        It is not a measurement; it is removed.
        #   min/max           -- the instrument's specified operating range.
        #                        Outside this but inside hard bounds is a REAL
        #                        reading from a drifting probe: flagged, kept.
        # ==================================================================
        self.GUARDRAIL_RANGES = {
            # hard_min must be strictly below min, or there is no low-side drift
            # band at all and a probe reading -8 C in a river is deleted as
            # "impossible" instead of kept as a cold-drifted reading. Surface
            # water cannot be colder than about -2 C, but a MISCALIBRATED probe
            # in that water legitimately reports lower; -30 is the electronics
            # floor below which the number is not a reading.
            'temperature':           {'min': -5,   'max': 50,
                                      'hard_min': -30, 'hard_max': 70},
            'ph':                    {'min': 0,    'max': 14,
                                      'hard_min': 0,  'hard_max': 14},
            'electric conductivity': {'min': 0,    'max': 2000,
                                      'hard_min': 0,  'hard_max': 100000},
            'disolved oxygen':       {'min': 0,    'max': 20,
                                      'hard_min': 0,  'hard_max': 25},
            'batteryvoltage':        {'min': 2000, 'max': 5000,
                                      'hard_min': 0,  'hard_max': 6000},
            'turbidity': {'min': 0, 'max': 1000, 'hard_min': -10, 'hard_max': 4000},
        }

        # Known telemetry sentinels: exact values a logger writes when a channel
        # has nothing to report. Never measurements, whatever range they fall in.
        self.SENTINEL_VALUES = [-9999.0, -999.0, -1023.0, -1023.4,
                                6553.5, 65535.0, 100000.0]
        
        self.NON_ZERO_PARAMS = ['electric conductivity', 'ph', 'batteryvoltage']
        
        for d in [self.output_dir, self.plot_dir]:
            os.makedirs(d, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid')


    def _sfx(self, filename):
        """Insert output_suffix BEFORE the extension: 'station_X.csv' with
        suffix '_iqr1.5' becomes 'station_X_iqr1.5.csv', not
        'station_X.csv_iqr1.5'. Empty suffix returns the name unchanged, so the
        default run writes exactly the filenames full_exp.py already reads."""
        if not self.output_suffix:
            return filename
        root, ext = os.path.splitext(filename)
        return f"{root}{self.output_suffix}{ext}"

    def _split_indices(self, n_rows):
        """Match full_exp.py splits after its 72-row feature warm-up."""
        effective = n_rows - self.feature_warmup
        if effective < 3:
            raise ValueError(
                f"Dataset has {n_rows} rows, not enough for feature_warmup="
                f"{self.feature_warmup} plus train/validation/test partitions.")
        train_end = self.feature_warmup + int(
            effective * self.model_train_split)
        test_start = self.feature_warmup + int(
            effective * self.test_start_split)
        if train_end < 1 or test_start <= train_end or test_start >= n_rows:
            raise ValueError(
                f"Dataset has {n_rows} rows, which cannot support the configured "
                "warm-up and split fractions.")
        return train_end, test_start

    @staticmethod
    def _column_key(name):
        """Return a stable key for matching heterogeneous source headers."""
        return ' '.join(
            str(name).strip().lower().replace('_', ' ').replace('-', ' ').split())

    def _canonicalize_columns(self, df):
        """Canonicalise headers once, before any rule or mask is evaluated.

        The historical spelling ``disolved oxygen`` is intentionally retained
        because full_exp.py currently uses that target name. It should be
        migrated in both scripts together, never in only one of them.
        """
        aliases = {
            'temperature': 'temperature',
            'water temperature': 'temperature',
            'turbidity': 'turbidity',
            'conductivity': 'electric conductivity',
            'electrical conductivity': 'electric conductivity',
            'electric conductivity': 'electric conductivity',
            'ec': 'electric conductivity',
            'dissolved oxygen': 'disolved oxygen',
            'disolved oxygen': 'disolved oxygen',
            'dissolvedoxygen': 'disolved oxygen',
            'disolvedoxygen': 'disolved oxygen',
            'do': 'disolved oxygen',
            'ph': 'ph',
            'p h': 'ph',
            'batteryvoltage': 'batteryvoltage',
            'battery voltage': 'batteryvoltage',
        }
        renamed = {}
        for col in df.columns:
            key = self._column_key(col)
            renamed[col] = aliases.get(key, key)
        out = df.rename(columns=renamed)
        duplicates = out.columns[out.columns.duplicated()].unique().tolist()
        if duplicates:
            raise ValueError(
                "Column canonicalisation produced duplicate channels: "
                f"{duplicates}. Resolve the source schema explicitly.")
        return out

    def _normalize_daily_index(self, df, station_id):
        """Normalise near-daily 00:00/01:00 timestamp shifts to calendar days.

        This changes timestamps only, never row count or values. It prevents a
        later ``asfreq('D')`` call from replacing valid 00:00 observations with
        artificial 01:00 gaps when a source crosses daylight-saving offsets.
        """
        if not self.normalize_daily_timestamps or len(df) < 3:
            return df
        diffs_h = df.index.to_series().diff().dropna().dt.total_seconds() / 3600.0
        if diffs_h.empty:
            return df
        near_daily = ((diffs_h >= 20.0) & (diffs_h <= 28.0)).mean() >= 0.90
        median_h = float(diffs_h.median())
        if not (near_daily and 20.0 <= median_h <= 28.0):
            return df

        normalized = df.index.normalize()
        changed = int((normalized != df.index).sum())
        if changed == 0:
            return df
        if normalized.has_duplicates:
            dup_dates = normalized[normalized.duplicated(keep=False)].unique()
            preview = [str(x) for x in dup_dates[:5]]
            raise ValueError(
                f"Daily timestamp normalisation for {station_id} would create "
                f"duplicate calendar dates: {preview}. Aggregate upstream with "
                "an explicitly documented rule instead of silently dropping rows.")
        out = df.copy()
        out.index = normalized
        out.index.name = 'timestamp'
        print(f"  Input validation: normalised {changed} near-daily timestamps "
              "to calendar midnight (row count unchanged).")
        return out

    def _load_input_frame(self, file_path, station_id):
        """Load one CSV with strict timestamp, schema, and numeric validation."""
        header = pd.read_csv(file_path, nrows=0)
        time_candidates = [
            c for c in header.columns
            if self._column_key(c) in {'time', 'timestamp', 'datetime', 'date time'}
        ]
        if len(time_candidates) != 1:
            raise ValueError(
                f"{file_path}: expected exactly one time/timestamp column; "
                f"found {time_candidates or 'none'}.")
        time_col = time_candidates[0]
        df = pd.read_csv(file_path, index_col=time_col)
        parsed = pd.to_datetime(df.index, errors='coerce')
        bad_time = int(pd.isna(parsed).sum())
        if bad_time:
            raise ValueError(f"{file_path}: {bad_time} timestamps could not be parsed.")
        df.index = pd.DatetimeIndex(parsed, name='timestamp')

        if not df.index.is_monotonic_increasing:
            print("  Input validation: timestamps were not chronological; sorting them.")
            df = df.sort_index(kind='stable')
        if df.index.has_duplicates:
            duplicate_count = int(df.index.duplicated(keep=False).sum())
            preview = [str(x) for x in df.index[df.index.duplicated(keep=False)][:5]]
            raise ValueError(
                f"{file_path}: {duplicate_count} rows have duplicate timestamps "
                f"(examples: {preview}). Define an upstream aggregation rule.")

        df = self._canonicalize_columns(df)
        if df.shape[1] == 0:
            raise ValueError(f"{file_path}: no sensor columns were found.")

        for col in df.columns:
            raw_nonmissing = df[col].notna()
            numeric = pd.to_numeric(df[col], errors='coerce')
            invalid = int((raw_nonmissing & numeric.isna()).sum())
            if invalid:
                print(f"  Input validation: '{col}' contains {invalid} non-numeric "
                      "values; recorded as missing.")
            inf_mask = np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan))
            if inf_mask.any():
                print(f"  Input validation: '{col}' contains {int(inf_mask.sum())} "
                      "infinite values; recorded as missing.")
                numeric = numeric.mask(inf_mask)
            df[col] = numeric.astype(float)

        df = self._normalize_daily_index(df, station_id)
        if not df.index.is_monotonic_increasing or df.index.has_duplicates:
            raise AssertionError("Timestamp invariants failed after input normalisation.")
        if len(df) >= 3:
            cadence_s = df.index.to_series().diff().dropna().dt.total_seconds()
            median_s = float(cadence_s.median())
            regular = np.isclose(cadence_s.to_numpy(dtype=float), median_s,
                                 rtol=0.0, atol=1e-6)
            irregular_count = int((~regular).sum())
            unit = 'day' if median_s >= 20 * 3600 else 'hour'
            value = median_s / (86400.0 if unit == 'day' else 3600.0)
            print(f"  Input validation: median cadence {value:g} {unit}(s); "
                  f"{irregular_count} irregular interval(s).")
        return df

    @staticmethod
    def _causal_forward_fill(frame, seed=None):
        """Forward-fill using only the segment's past and an earlier-partition seed."""
        out = frame.ffill()
        if seed is not None:
            out = out.fillna(seed)
        return out

    @staticmethod
    def _constant_run_mask(df, min_len=6):
        """Flag every point in an exact constant run, including its first points."""
        out = pd.DataFrame(False, index=df.index, columns=df.columns)
        for col in df.columns:
            s = df[col]
            valid = s.notna()
            # Every missing sample starts a new group, so a run cannot bridge a gap.
            changed = s.ne(s.shift()) | ~valid | ~valid.shift(fill_value=False)
            groups = changed.cumsum()
            sizes = s.groupby(groups).transform('size')
            out[col] = valid & sizes.ge(min_len)
        return out

    def _plot_cleaning_comparison(self, df_original, df_cleaned_nans, df_final, station_id):
        """
        Generates a high-quality "narrative" plot showing the full cleaning process.
        - Original data is shown in the background.
        - Rejected points are marked with red X's.
        - The final, cleaned line is plotted prominently.
        """
        plot_name = self._sfx(f"{station_id}_cleaning_narrative.png")
        plot_path = os.path.join(self.plot_dir, plot_name)
        cols = df_original.columns
        num_params = len(cols)
        
        fig, axes = plt.subplots(
            num_params, 1, figsize=(20, 5 * num_params), sharex=True,
            squeeze=False)
        axes = axes[:, 0]
        fig.suptitle(f"Data Cleaning Narrative - Station {station_id}", fontsize=20, y=0.99)

        for i, col in enumerate(cols):
            # 1. Plot the final, cleaned, and interpolated data as the main line
            sns.lineplot(ax=axes[i], x=df_final.index, y=df_final[col], color='green', lw=1.5, label='Final Cleaned Signal')
            
            # 2. Plot the original data faintly in the background
            sns.lineplot(ax=axes[i], x=df_original.index, y=df_original[col], color='blue', alpha=0.3, lw=1, label='Original Signal')
            
            # 3. Identify and highlight the rejected points
            rejected_mask = df_cleaned_nans[col].isna() & df_original[col].notna()
            rejected_points = df_original[rejected_mask]
            
            if not rejected_points.empty:
                sns.scatterplot(ax=axes[i], x=rejected_points.index, y=rejected_points[col], color='red', marker='x', s=50, label='Rejected Points')

            axes[i].set_title(col, loc='left', fontsize=14)
            axes[i].set_ylabel("Value")
            axes[i].legend()

        plt.xlabel("Timestamp", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(plot_path, dpi=150) # Use higher DPI for publication quality
        plt.close()
        print(f"  -> Saved cleaning narrative plot to: {plot_path}")

    def _plot_distributions(self, df_before, df_after_nans, station_id):
        """Generates before-and-after plots of the data distributions."""
        plot_name = self._sfx(f"{station_id}_distributions.png")
        plot_path = os.path.join(self.plot_dir, plot_name)
        cols = df_before.columns
        num_params = len(cols)
        
        fig, axes = plt.subplots(
            num_params, 2, figsize=(12, 4 * num_params), squeeze=False)
        fig.suptitle(f"Statistical Impact of Cleaning - Station {station_id}", fontsize=16)

        for i, col in enumerate(cols):
            sns.histplot(df_before[col], kde=True, ax=axes[i, 0], color='blue')
            axes[i, 0].set_title(f"{col} (Before DQA)")
            axes[i, 0].set_xlabel("")

            sns.histplot(df_after_nans[col].dropna(), kde=True, ax=axes[i, 1], color='green')
            axes[i, 1].set_title(f"{col} (After DQA)")
            axes[i, 1].set_xlabel("")

        axes[0, 0].set_ylabel("Count")
        axes[0, 1].set_ylabel("")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  -> Saved distribution plot to: {plot_path}")

    @staticmethod
    def _detect_prefilled_runs(df, min_len=8, rtol=1e-7, atol=1e-12):
        """Flag long, non-zero, constant-slope runs as likely upstream fills.

        Slopes are computed per unit time rather than per row, and runs cannot
        bridge a source NaN. This avoids the former false-positive path where
        equally spaced values across irregular timestamps or missing intervals
        appeared to have a constant first difference.

        This remains a provenance heuristic, not proof of upstream processing;
        therefore the dedicated prefilled mask is saved for audit.
        """
        out = pd.DataFrame(False, index=df.index, columns=df.columns)
        for col in df.columns:
            series = df[col]
            valid = series.notna()
            # Split into contiguous nonmissing blocks; never bridge a source gap.
            block_ids = (~valid).cumsum()
            for _, block in series[valid].groupby(block_ids[valid]):
                if len(block) < min_len:
                    continue
                values = block.to_numpy(dtype=float)
                time_s = block.index.asi8.astype(float) / 1e9
                dt = np.diff(time_s)
                dv = np.diff(values)
                slopes = np.divide(
                    dv, dt, out=np.full_like(dv, np.nan, dtype=float),
                    where=dt > 0)
                if len(slopes) == 0:
                    continue

                run_start = 0
                for j in range(1, len(slopes) + 1):
                    continues = (
                        j < len(slopes)
                        and np.isfinite(slopes[j - 1])
                        and np.isfinite(slopes[j])
                        and np.isclose(slopes[j], slopes[j - 1],
                                       rtol=rtol, atol=atol)
                    )
                    if continues:
                        continue
                    run = slopes[run_start:j]
                    n_points = len(run) + 1
                    reference = float(np.nanmedian(run)) if len(run) else np.nan
                    constant = (
                        len(run) > 0
                        and np.isfinite(reference)
                        and np.all(np.isfinite(run))
                        and np.all(np.isclose(run, reference,
                                              rtol=rtol, atol=atol))
                    )
                    # Flat runs are real stuck-sensor evidence handled in Stage 3.
                    if constant and n_points >= min_len and abs(reference) > atol:
                        out.loc[block.index[run_start:j + 1], col] = True
                    run_start = j
        return out

    def _statistical_outlier_mask(self, df, iqr_multiplier=2.5):
        """Boolean mask of IQR outliers. Thresholds come from the training
        partition only, so the test window cannot influence its own filter.

        Returns the mask rather than a modified frame: in fault-preserving mode
        the caller flags these points and keeps the values, because an outlier
        at a degrading sensor is the calibration excursion under study.
        """
        mask_out = pd.DataFrame(False, index=df.index, columns=df.columns)
        train_end_idx, _ = self._split_indices(len(df))
        df_train = df.iloc[:train_end_idx]

        for col in df.columns:
            if "battery" in col.lower():
                continue
            Q1 = df_train[col].quantile(0.25)
            Q3 = df_train[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0 or pd.isna(IQR):
                continue
            lower_bound = Q1 - (iqr_multiplier * IQR)
            upper_bound = Q3 + (iqr_multiplier * IQR)
            mask_out[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        return mask_out

    def _nullify_statistical_outliers(self, df, iqr_multiplier=2.5):
        """Legacy destructive form, kept for the 'nullify' comparison mode."""
        df_out = df.copy()
        m = self._statistical_outlier_mask(df, iqr_multiplier)
        for col in m.columns:
            if m[col].sum() > 0:
                df_out.loc[m[col], col] = None
        return df_out

    def run_assessment(self):
        """Executes the full DQA pipeline for all files in the input directory."""
        print("--- Starting Adaptive Data Quality Assessment Pipeline (v2.3) ---")
        self.failures = []
        if not os.path.isdir(self.input_dir):
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
        # [R3 gap found in the first run] The original glob only matched
        # 'station_*.csv', so the Belgian VUB file was never processed and never
        # got an observation mask. full_exp.py therefore SKIPS the entire R3
        # imputation decomposition on VUB -- which is the one dataset where the
        # turbidity degeneracy (Sec 2.4) lives. Any .csv in the input directory
        # is now processed.
        station_files = sorted(f for f in os.listdir(self.input_dir) if f.endswith('.csv'))
        if not station_files:
            raise RuntimeError(f"No .csv files found in {self.input_dir}.")
        print(f"Found {len(station_files)} input file(s): {station_files}")
        
        summary_stats = []

        for i, filename in enumerate(station_files):
            station_id = (filename[len('station_'):] if filename.startswith('station_')
                          else filename).replace('.csv', '')
            print(f"\n--- Processing Station: {station_id} ({i+1}/{len(station_files)}) ---")
            
            file_path = os.path.join(self.input_dir, filename)
            df_original = self._load_input_frame(file_path, station_id)

            # ==============================================================
            # [Stage 0] PRE-EXISTING INTERPOLATION IN THE RAW FILE.
            #
            # obs_mask below is built from df_cleaned.notna(), so it can only
            # distinguish points THIS script nullified. A value that arrived in
            # the raw CSV already synthesised is not NaN, so it is recorded as
            # MEASURED and the benchmark will score against it as ground truth.
            #
            # Station 861513064189509 carried a 157-point run in dissolved
            # oxygen decaying from 100000 to 0 with a first difference constant
            # to ten decimal places, mask-labelled 100% measured. A sensor
            # cannot produce that; something upstream filled a logger outage by
            # straight-line interpolation between a sentinel and a real reading.
            # The same row spans (193-349, 406-495) are straight in ph and
            # batteryVoltage too, which is the signature of a whole-logger gap.
            #
            # A constant first difference over >=8 samples is not a physical
            # signal, so those points are demoted to NOT-measured here. They are
            # then interpolated like any other gap and, crucially, excluded from
            # observed-only metrics rather than scored as truth.
            # ==============================================================
            prefilled = self._detect_prefilled_runs(
                df_original, min_len=self.prefilled_min_len)
            if prefilled.values.any():
                print("  Stage 0: Pre-existing interpolation in the raw file")
                for col in df_original.columns:
                    k = int(prefilled[col].sum())
                    if k:
                        print(f"    - '{col}': {k} points sit on straight-line "
                              f"runs (constant 1st difference); "
                              f"removed from usable inputs and marked NOT-observed.")

            # Critical correction: the previous version changed only obs_mask,
            # leaving these synthetic values in model inputs and targets. They
            # must be missing before every cleaning and reconstruction stage.
            df_cleaned = df_original.mask(prefilled)
            
            preserve = (self.fault_handling == 'preserve')
            # fault_flags[col] is True where a degradation signature fired.
            # In preserve mode the VALUE SURVIVES and only the flag is set.
            fault_flags = pd.DataFrame(False, index=df_cleaned.index,
                                       columns=df_cleaned.columns)
            # 'none' rather than '': an empty string round-trips through CSV
            # as NaN, which makes the label file awkward to read back.
            fault_kind = pd.DataFrame('none', index=df_cleaned.index,
                                      columns=df_cleaned.columns)

            def _mark(col, mask, kind, sentinel=False):
                """Record a fault. Delete the value only if it cannot be modelled."""
                if mask.sum() == 0:
                    return
                fault_flags.loc[mask, col] = True
                # Stages overlap: a stuck-at-zero probe fires Stage 3 AND Stage 4,
                # and a plain assignment would let the later stage erase the
                # earlier label. That overlap is the diagnosis -- stuck+zero is a
                # DEAD probe, stuck alone is a fouled one -- so labels accumulate.
                prev = fault_kind.loc[mask, col]
                fault_kind.loc[mask, col] = np.where(
                    prev.eq('none'), kind, prev.astype(str) + ' + ' + kind)
                if sentinel or not preserve:
                    df_cleaned.loc[mask, col] = None
                verb = "Removed" if (sentinel or not preserve) else "FLAGGED (kept)"
                print(f"    - '{col}': {verb} {int(mask.sum())} {kind}.")

            print(f"  Stage 1: Physical guardrails  [mode: {self.fault_handling}]")
            for param, ranges in self.GUARDRAIL_RANGES.items():
                if param not in df_cleaned.columns:
                    continue
                lo = ranges.get('min', -np.inf)
                hi = ranges.get('max', np.inf)
                hlo = ranges.get('hard_min', -np.inf)
                hhi = ranges.get('hard_max', np.inf)
                v = df_cleaned[param]

                # Tier 1: an exact known logger sentinel, or a physically
                # impossible value. Not a measurement -> removed.
                impossible = (v < hlo) | (v > hhi)
                is_sentinel = v.isin(self.SENTINEL_VALUES)
                _mark(param, (impossible | is_sentinel).fillna(False),
                      'sentinel / non-physical readings', sentinel=True)

                # Tier 2: outside the instrument's spec range but physically
                # possible. A real reading from a drifting probe -> flagged, kept.
                out_of_spec = (((v < lo) | (v > hi)) & ~impossible
                               & ~is_sentinel).fillna(False)
                _mark(param, out_of_spec, 'out-of-range readings (drift)')

            print("  Stage 2: Adaptive statistical filter (IQR)")
            # An IQR outlier at a degrading sensor IS the calibration excursion
            # this paper is about, so in preserve mode it is flagged, not cut.
            iqr_mask = self._statistical_outlier_mask(
                df_cleaned, iqr_multiplier=self.iqr_multiplier)
            for col in iqr_mask.columns:
                _mark(col, iqr_mask[col], 'statistical outliers (excursion)')

            print("  Stage 3: Stuck value filter  [biofouling signature]")
            window_size = 6
            stuck_mask = self._constant_run_mask(df_cleaned, min_len=window_size)
            for col in df_cleaned.columns:
                _mark(col, stuck_mask[col], 'stuck / zero-variance readings')

            print("  Stage 4: Anomalous zero filter")
            for param in self.NON_ZERO_PARAMS:
                if param in df_cleaned.columns:
                    _mark(param, df_cleaned[param] == 0, 'anomalous zero readings')

            n_flag = int(fault_flags.values.sum())
            n_kept = int((fault_flags & df_cleaned.notna()).values.sum())
            print(f"    => {n_flag} fault points detected, {n_kept} retained in the "
                  f"data as degraded readings, {n_flag - n_kept} unmodellable.")

            # Create a copy before interpolation for plotting purposes
            df_cleaned_pre_interpolation = df_cleaned.copy()
            
            # ==============================================================
            # [Gate A] A column that Stages 1-4 emptied completely cannot be
            # interpolated from anything -- df.interpolate() leaves it all-NaN
            # and the cleaned CSV silently carries a column of NaN. In the first
            # run this happened to station 861513064189988 for BOTH electric
            # conductivity and temperature (0.0% valid), and the job still
            # exited 0. Fail here instead of shipping an empty column downstream.
            # ==============================================================
            dead = [c for c in df_cleaned.columns if df_cleaned[c].isna().all()]
            if dead:
                print(f"  [Gate A] ERROR: Stages 1-4 removed 100% of {dead} for "
                      f"station {station_id}. These columns cannot be recovered by "
                      f"interpolation and will be all-NaN in the cleaned file.")
                print(f"           Either loosen the filter that emptied them "
                      f"(check the Stage 1-4 counts above), or drop this station "
                      f"from the study and say so in the manuscript. Do NOT model "
                      f"an all-NaN target.")
                # One row per target is required by full_exp.py's exclusion
                # lookup. A list-valued target silently failed to match.
                for col in dead:
                    self.failures.append((station_id, 'empty_after_cleaning', col))

            print("\n  Stage 5: Reporting Data Validity...")
            # Capture provenance before filling. True means present in the
            # source after synthetic upstream fills and nonmeasurements were
            # removed; it does not prove the sensor itself was healthy.
            obs_mask = df_cleaned.notna() & ~prefilled.reindex(
                index=df_cleaned.index, columns=df_cleaned.columns,
                fill_value=False)
            # Denominator is the complete expected time grid, not merely the
            # originally nonmissing subset. The former calculation overstated
            # completeness whenever the source already contained NaNs.
            validity_percent = 100.0 * df_cleaned.notna().mean()
            station_summary = {'station_id': station_id}
            station_summary.update({
                f"{col}_usable_before_fill_%": round(float(pct), 3)
                for col, pct in validity_percent.items()
            })
            summary_stats.append(station_summary)
            print(validity_percent.to_string())

            print("\n  Stage 6: Split-aware reconstruction (no evaluation leakage)...")
            # ==============================================================
            # [R3] obs_mask was captured before any value was filled.
            # True  = a source value survived provenance/nonmeasurement checks.
            # False = missing, detected upstream fill, or a value removed by
            #         this script. It does not assert that a retained sensor was
            #         healthy; fault masks carry that separate information.
            #
            # Without this mask the benchmark cannot answer the reviewer's
            # question -- "are the reported gains driven by the imputation
            # rather than by the model?" -- because after Stage 6 the two kinds
            # of point are indistinguishable in the CSV.
            # ==============================================================
            # 1. Determine the model-train, validation, and test boundaries.
            n_rows = len(df_cleaned)
            train_end_idx, split_idx = self._split_indices(n_rows)

            # 2. Split before any learned or future-aware reconstruction.
            df_train_part = df_cleaned.iloc[:train_end_idx].copy()
            df_val_part = df_cleaned.iloc[train_end_idx:split_idx].copy()
            df_test_part = df_cleaned.iloc[split_idx:].copy()

            # 3. Offline model-training reconstruction may use neighbours from
            # inside the training partition only. It cannot see validation/test.
            df_train_final = df_train_part.interpolate(
                method='time', limit_direction='both')

            # Validation is an evaluation partition too: fill it only from its
            # own past, seeded by the completed training history.
            train_seed = df_train_final.iloc[-1]
            df_val_final = self._causal_forward_fill(df_val_part, train_seed)

            # --------------------------------------------------------------
            # [R3, second issue] CAUSALITY OF THE TEST-PARTITION FILL.
            #
            # The submitted pipeline filled test gaps with
            #     interpolate(method='time', limit_direction='both')
            # applied to the whole test block at once. A gap at time t is then
            # filled using observations at t+k. That information does not exist
            # at inference time on a live deployment, so the test partition is
            # not a faithful simulation of operation -- this is precisely the
            # ambiguity the reviewer flagged, and the manuscript currently reads
            # both ways.
            #
            # causal_test_fill=True uses forward-fill only (t-k -> t), which is
            # what an operational system can actually do. Keep the old behaviour
            # available so the two can be compared and the difference reported.
            # --------------------------------------------------------------
            if self.causal_test_fill:
                test_seed = df_val_final.iloc[-1]
                df_test_final = self._causal_forward_fill(df_test_part, test_seed)
                print("    - Validation and test partitions filled CAUSALLY with "
                      "past-only forward-fill. No future evaluation values were used.")
            else:
                df_test_final = df_test_part.interpolate(
                    method='time', limit_direction='both')
                print("    - WARNING: test partition filled with a bidirectional "
                      "pass. Gaps are filled using FUTURE test observations, which "
                      "are unavailable at inference. Set causal_test_fill=True "
                      "before reporting deployment-relevant numbers.")

            # 4. Re-combine for saving/plotting
            df_final = pd.concat([df_train_final, df_val_final, df_test_final])
            if not df_final.index.equals(df_cleaned.index):
                raise AssertionError("Cleaning changed row order or timestamps.")
            if np.isinf(df_final.to_numpy(dtype=float, na_value=np.nan)).any():
                raise AssertionError("Cleaned output contains infinite values.")

            # [R3] Report the imputation load, split by partition, so the
            # manuscript can state it rather than leave it to be discovered.
            reconstructed = (~obs_mask) & df_final.notna()
            unresolved = df_final.isna()
            imp_train = 100.0 * reconstructed.iloc[:train_end_idx].mean()
            imp_val = 100.0 * reconstructed.iloc[train_end_idx:split_idx].mean()
            imp_test = 100.0 * reconstructed.iloc[split_idx:].mean()
            print("    - Reconstructed fraction (%) by partition:")
            print(pd.DataFrame({'train_%': imp_train.round(2),
                                'validation_%': imp_val.round(2),
                                'test_%': imp_test.round(2)}).to_string())
            unresolved_pct = 100.0 * unresolved.mean()
            if (unresolved_pct > 0).any():
                print("    - Unresolved NaN fraction (%) after reconstruction:")
                print(unresolved_pct[unresolved_pct > 0].round(2).to_string())
            if (imp_test > 30).any():
                heavy = imp_test[imp_test > 30].index.tolist()
                print(f"    - WARNING: >30% of TEST points are imputed for "
                      f"{heavy}. Any metric on these targets is largely a "
                      f"measurement of the interpolator. Report the "
                      f"observed-only metric as primary for them.")

            print(f"    - Boundaries: model train [0, {self.model_train_split:.2f}), "
                  f"validation [{self.model_train_split:.2f}, "
                  f"{self.test_start_split:.2f}), test "
                  f"[{self.test_start_split:.2f}, 1.00) of the post-warm-up "
                  f"feature rows; raw row indices are 0:{train_end_idx}, "
                  f"{train_end_idx}:{split_idx}, and {split_idx}:{n_rows}.")
            

            # --- PLOTTING FOR ALL STATIONS ---
            self._plot_cleaning_comparison(df_original, df_cleaned_pre_interpolation, df_final, station_id)
            self._plot_distributions(df_original, df_cleaned_pre_interpolation, station_id)
            
            print("\n  Stage 7: Saving Cleaned Data...")
            output_path = os.path.join(self.output_dir, f"cleaned_{self._sfx(filename)}")
            df_final.to_csv(output_path)
            print(f"  -> Saved cleaned data to: {output_path}")

            # ==============================================================
            # [Gate B] Test-window usability. A target can be non-empty and
            # still be unusable: forward fill turns a sparse test window into a
            # staircase of constant plateaus, and an MAE computed on a staircase
            # measures the fill, not the model.
            #
            # First run, station 861513064189509, dissolved oxygen: only 21.4%
            # of the evaluation window was measured, 84.5% of it sat inside
            # constant runs, and the longest run was 60 of 84 points. That target
            # must not enter the C2 results table.
            #
            # The window checked here is full_exp.py's ACTUAL evaluation window:
            # create_tabular_features drops LONG_WINDOW_SIZE=72 leading rows, and
            # the test split is the last 15% of what remains.
            # ==============================================================
            n_rows = len(df_final)
            eval_start = split_idx
            if eval_start < n_rows:
                for col in df_final.columns:
                    seg = df_final[col].iloc[eval_start:]
                    m_seg = obs_mask[col].iloc[eval_start:]
                    if seg.isna().all():
                        continue
                    obs_pct = 100.0 * m_seg.mean()
                    grp = (seg != seg.shift()).cumsum()
                    sizes = seg.groupby(grp).transform('size')
                    flat = sizes >= 6
                    flat_pct = 100.0 * flat.mean()
                    longest = int(sizes.max())

                    # In preserve mode a flat run has TWO possible causes and the
                    # distinction decides the verdict:
                    #   flat AND imputed  -> the forward fill made it flat. An MAE
                    #                        there scores the interpolator.
                    #   flat AND measured -> the SENSOR is flat. That is a real
                    #                        stuck-probe fault: exactly the signal
                    #                        this paper exists to study, and it
                    #                        must NOT be filtered away. But it is
                    #                        still degenerate for a headline MAE,
                    #                        because a constant predictor wins.
                    flat_measured = 100.0 * (flat & m_seg.astype(bool)).mean()
                    flat_filled = 100.0 * (flat & ~m_seg.astype(bool)).mean()

                    # Run-length is blind to a sensor that ALTERNATES between a
                    # small number of stuck levels: 0, 0, 60.28, 0, 60.28 has no
                    # long run, so `flat` barely fires, yet the column carries
                    # almost no information. Station 861513064189509's dissolved
                    # oxygen test window is 63% zeros and 23% one repeated
                    # constant -- 86% two values -- and it passed the run-length
                    # tiers as a mere "FAULT". It then supplied 98.5% of that
                    # station's entire reported MAE advantage. Measure value
                    # concentration directly, independent of ordering.
                    obs_vals = seg[m_seg.astype(bool)].dropna()
                    if len(obs_vals) > 0:
                        vc = obs_vals.value_counts()
                        dom1 = 100.0 * vc.iloc[0] / len(obs_vals)
                        n_distinct = int(obs_vals.nunique())
                    else:
                        dom1, n_distinct = 0.0, 0

                    if dom1 > 60 and len(obs_vals) > 0:
                        # One value owns the majority of the measured window. A
                        # constant predictor is near-optimal, so any MAE here
                        # ranks models on how well they reproduce a stuck level,
                        # not on forecasting skill. The readings stay in the data
                        # (they are the fault evidence) but no headline metric.
                        print(f"  [Gate B] DEGENERATE '{col}': one value accounts for "
                              f"{dom1:.1f}% of the {int(len(obs_vals))} measured points "
                              f"in the evaluation window ({n_distinct} distinct values "
                              f"total). The channel is pinned, not forecastable: a "
                              f"constant predictor is near-optimal, so an MAE here "
                              f"measures stuck-level reproduction rather than skill. "
                              f"Readings are KEPT as fault evidence; report no headline "
                              f"MAE/MASE for {station_id}/{col} and cite it as a "
                              f"detected sensor failure.")
                        self.failures.append((station_id, 'degenerate_stuck_target', col))
                    elif obs_pct < 40:
                        print(f"  [Gate B] BLOCK '{col}': only {obs_pct:.1f}% of the "
                              f"{len(seg)}-point evaluation window was measured "
                              f"({int(m_seg.sum())} points). Longest constant run "
                              f"{longest}. Exclude this target from the results "
                              f"table for {station_id}; do not report an MAE on it.")
                        self.failures.append((station_id, 'unusable_test_window', col))
                    elif flat_measured > 90:
                        # A genuinely stuck sensor across the whole window. Keep the
                        # data -- it is the fault under study -- but a headline MAE
                        # here is meaningless: persistence scores ~0.
                        print(f"  [Gate B] DEGENERATE '{col}': {flat_measured:.1f}% of "
                              f"the evaluation window is MEASURED but constant "
                              f"(longest run {longest}). The sensor is stuck, not "
                              f"interpolated. This is a real fault and the readings "
                              f"are kept for the fault-stratified analysis, but a "
                              f"naive-persistence baseline scores ~0 error here, so "
                              f"no headline MAE/MASE may be reported for it. Exclude "
                              f"from the {station_id} results table and cite it as a "
                              f"detected sensor failure instead.")
                        self.failures.append((station_id, 'degenerate_stuck_target', col))
                    elif flat_filled > 25:
                        print(f"  [Gate B] CAUTION '{col}': {flat_filled:.1f}% of the "
                              f"evaluation window sits inside FORWARD-FILL plateaus "
                              f"(longest run {longest}, {obs_pct:.1f}% measured). "
                              f"Report the observed-only metric as primary.")
                    elif flat_measured > 25:
                        print(f"  [Gate B] FAULT '{col}': {flat_measured:.1f}% of the "
                              f"evaluation window is measured-but-constant "
                              f"(longest run {longest}) -- a genuine stuck-probe "
                              f"episode, preserved for the degraded-vs-healthy "
                              f"comparison. Report MAE_fault alongside the headline.")
                    elif obs_pct < 80:
                        print(f"  [Gate B] NOTE '{col}': {obs_pct:.1f}% measured in "
                              f"the evaluation window. Observed-only metric is the "
                              f"honest headline number.")

            # [FAULT-PRESERVING MODE] Emit the fault mask. This is what makes
            # the degradation available to the benchmark as a variable rather
            # than as damage: full_exp.py reports metrics separately on
            # fault-flagged and healthy points, which is the direct measurement
            # of "does the corrector rescue a degraded sensor" that the paper
            # claims and never actually made.
            fault_path = os.path.join(self.output_dir, f"fault_mask_{self._sfx(filename)}")
            fault_flags.to_csv(fault_path)
            kind_path = os.path.join(self.output_dir, f"fault_kind_{self._sfx(filename)}")
            fault_kind.to_csv(kind_path)
            print(f"  -> Saved fault mask to: {fault_path}")

            # ------------------------------------------------------------------
            # C-T5: per-fault-type masks. fault_kind carries accumulating labels
            # (a dead probe reads 'stuck / zero-variance readings + anomalous
            # zero readings'), so membership is tested by SUBSTRING, not
            # equality -- a dead probe belongs to both the stuck and the zero
            # stratum, which is the correct reading of the diagnosis.
            #
            # The benchmark uses these to report MAE_stuck / MAE_excursion /
            # MAE_zero separately. That matters because the paper's claim is
            # about rescuing degraded sensors, and 'degraded' is not one thing:
            # a stuck probe gives the corrector a constant to work from, while an
            # excursion gives it a transient. Pooling them hides which failure
            # mode the correction actually addresses.
            # ------------------------------------------------------------------
            FAULT_KIND_STRATA = {
                'stuck':     'stuck / zero-variance',
                'zero':      'anomalous zero',
                'excursion': 'statistical outliers (excursion)',
                'sentinel':  'sentinel / non-physical',
                'drift':     'out-of-range readings (drift)',
            }
            kind_str = fault_kind.astype(str)
            for short, needle in FAULT_KIND_STRATA.items():
                sub = kind_str.apply(lambda s: s.str.contains(needle, regex=False))
                n_hit = int(sub.values.sum())
                if n_hit == 0:
                    continue
                sub_path = os.path.join(
                    self.output_dir, f"fault_mask_{short}_{self._sfx(filename)}")
                sub.to_csv(sub_path)
                per_col = 100.0 * sub.mean()
                print(f"  -> [C-T5] '{short}' stratum: {n_hit} cells "
                      f"({', '.join(f'{c}={v:.1f}%' for c, v in per_col.items() if v > 0)})"
                      f" -> {os.path.basename(sub_path)}")

            fp = 100.0 * fault_flags.mean()
            print("  -> Fault-flagged fraction (%) per parameter:")
            print(fp[fp > 0].round(2).to_string() if (fp > 0).any() else "     (none)")

            # [R3] Emit the observation mask alongside the cleaned data.
            # full_exp.py reads this via load_observation_mask() and uses it to
            # report every headline metric a second time on observed-only points.
            mask_path = os.path.join(self.output_dir, f"interp_mask_{self._sfx(filename)}")
            obs_mask.to_csv(mask_path)
            print(f"  -> Saved observation mask to: {mask_path}")

            # Separate source-provenance evidence from sensor-fault evidence.
            # Upstream linear fills are not counted as observed sensor faults.
            prefilled_path = os.path.join(
                self.output_dir, f"prefilled_mask_{self._sfx(filename)}")
            prefilled.to_csv(prefilled_path)
            print(f"  -> Saved detected upstream-fill mask to: {prefilled_path}")

        # --- Final Summary Report ---
        summary_df = pd.DataFrame(summary_stats).set_index('station_id')
        print("\n\n" + "="*80)
        print("--- FINAL DATA QUALITY SUMMARY ---")
        print("Percentage of expected rows usable before reconstruction:")
        print(summary_df.to_string())
        print("="*80)
        summary_path = os.path.join(
            self.output_dir, self._sfx('data_quality_summary.csv'))
        summary_df.to_csv(summary_path)
        print(f"Saved machine-readable summary to: {summary_path}")

        # ==================================================================
        # Two different things were being conflated here. A target that cannot
        # carry a headline metric is not the same as a run that must not
        # proceed. The first run halted the entire pipeline over one all-NaN
        # column at station 861513064189988 -- a station that is not one of the
        # paper's three testbeds and is never loaded by full_exp.py.
        #
        # So gate outcomes are now written as an EXCLUSION LIST that the
        # benchmark reads and honours, and the run only hard-fails when a
        # target the manuscript actually reports is unusable.
        # ==================================================================
        # Suffixing is essential: IQR sweep runs must never overwrite the
        # canonical exclusion list consumed by full_exp.py.
        excl_path = os.path.join(
            self.output_dir, self._sfx('target_exclusions.csv'))
        if self.failures:
            ex = pd.DataFrame(self.failures,
                              columns=['station_id', 'reason', 'target'])
            ex = ex.drop_duplicates(
                subset=['station_id', 'reason', 'target']).reset_index(drop=True)
            ex['headline_metric_allowed'] = False
            ex['keep_for_fault_analysis'] = ex['reason'].eq('degenerate_stuck_target')
            ex.to_csv(excl_path, index=False)
            print("\n" + "="*80)
            print(f"TARGET EXCLUSIONS  (written to {os.path.basename(excl_path)})")
            print("="*80)
            print(ex.to_string(index=False))
            print()
            print("  empty_after_cleaning     -- nothing survives; the column is all-NaN.")
            print("  unusable_test_window     -- too few measured points to score anything.")
            print("  degenerate_stuck_target  -- the sensor is genuinely stuck. The readings")
            print("                              ARE kept and are the most valuable fault")
            print("                              evidence in the study, but a constant")
            print("                              predictor scores ~0 error, so no headline")
            print("                              MAE/MASE may be reported. Cite it as a")
            print("                              detected sensor failure and report the")
            print("                              fault-stratified numbers instead.")
            print()
            print("full_exp.py reads this file and drops these (station, target) pairs")
            print("from its results tables automatically.")
        else:
            pd.DataFrame(columns=['station_id', 'reason', 'target',
                                  'headline_metric_allowed',
                                  'keep_for_fault_analysis']).to_csv(excl_path, index=False)
            print("\nAll data quality gates passed; no target exclusions.")

        # Hard-fail only if a target the MANUSCRIPT reports is unusable. Station
        # 861513064189988 is not among the paper's testbeds, so its failures are
        # recorded and excluded but do not halt the chain.
        reported = {s for s in self.REPORTED_STATIONS}
        blocking = [f for f in self.failures if f[0] in reported]
        if blocking:
            print("\n" + "!"*80)
            print("BLOCKING: a target reported in the manuscript is unusable")
            print("!"*80)
            for station, kind, detail in blocking:
                print(f"  station {station:>18s} | {kind:22s} | {detail}")
            print("\nThese sit in one of the paper's three testbeds, so the exclusion")
            print("has to be a stated decision in the manuscript rather than a silent")
            print("drop. Decide, record it in the response letter, add the target to")
            print("MANUAL_EXCLUSIONS below to acknowledge it, then re-run.")
            unack = [f for f in blocking
                     if (f[0], f[2]) not in self.acknowledged_exclusions]
            if unack:
                raise SystemExit(2)
            print("\nAll blocking exclusions are explicitly acknowledged; continuing.")

        print("\nAssessment complete.")

if __name__ == '__main__':
    # C-T4 controls. The sweep is OFF by default: it re-runs the whole
    # assessment once per multiplier, and the canonical run must finish first.
    # R9 / H1: default flipped ON. Round 8 left this at '0', the sweep never
    # ran, and no iqr_sweep_flag_rates.csv was produced -- so the
    # preprocessing-robustness blocker stayed open on an unset env var. Set
    # RUN_IQR_SWEEP=0 explicitly to skip it.
    RUN_IQR_SWEEP = os.environ.get('RUN_IQR_SWEEP', '1') == '1'
    IQR_SWEEP_VALUES = [1.5, 2.0, 2.5, 3.0]

    INPUT_DIR = 'processed_data'
    OUTPUT_DIR = 'cleaned_data'
    PLOT_DIR = 'quality_assessment_plots'
    
    # [R3] These fractions are applied after the same 72-row feature warm-up as
    # full_exp.py. This makes the raw row boundaries coincide with its actual
    # 70/15/15 model-train/validation/test split.
    assessor = DataQualityAssessor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        plot_dir=PLOT_DIR,
        model_train_split=0.70,  # thresholds and training reconstruction stop here
        train_split=0.85,        # backward-compatible name for test-start boundary
        feature_warmup=72,       # matches full_exp.py LONG_WINDOW_SIZE
        causal_test_fill=True,   # past-only ffill; no future test values used
        # The three testbeds whose numbers appear in the manuscript. A gate
        # failure inside these halts the run until you acknowledge it; a failure
        # at any other station is recorded in target_exclusions.csv and skipped.
        reported_stations=['861513064190226',   # 'stable'
                           '861513064189509',   # 'challenged'
                           '861513064190234',  # 'challenged_2'
                           'VUB_Belgian_Data'],
        # Exclusions you have decided on and will state in the manuscript.
        # Adding a pair here means "I know, it is deliberate, do not halt".
        #   ('861513064189509', 'disolved oxygen') -- 21% measured; excluded.
        acknowledged_exclusions=[
            ('VUB_Belgian_Data', 'turbidity'),
            # DECIDED, and stated in the revised manuscript + response letter.
            # C_2 (861513064189509) dissolved oxygen: the test window is 63%
            # zeros and 23% a single repeated constant (86% two values, 9
            # distinct in 68 measured points), and 27 of 95 submitted points
            # exceeded the physical ceiling for DO entirely. The submitted DO
            # MAEs (27.8-3391) are larger than the full 0-25 mg/L range of the
            # quantity, which is the arithmetic signature of scoring against
            # sentinels. This target is reported as a detected sensor failure,
            # not as a forecasting result.
            ('861513064189509', 'disolved oxygen'),
            # C_1 (861513064190234) dissolved oxygen: 100% measured-but-constant
            # across the evaluation window (single value, longest run 108). A
            # genuinely stuck probe; kept as fault evidence, no headline metric.
            # Never appeared in the C_1 table (N=3: Temp/pH/EC), so no published
            # number changes.
            ('861513064190234', 'disolved oxygen'),
            # F (861513064189988) is not a forecasting testbed in the paper: it
            # appears only as the row 'F' in the DQA summary table, which is
            # precisely a report of its failure. DO and temperature are empty
            # after cleaning and EC is 100% stuck.
            ('861513064189988', 'disolved oxygen'),
            ('861513064189988', 'temperature'),
            ('861513064189988', 'electric conductivity'),
        ],
        # Keep the degradation IN the data. Stages 1-4 now label faults instead
        # of deleting them; see the FAULT-PRESERVING MODE note in __init__.
        # Set to 'nullify' only to reproduce the submitted pipeline for the
        # side-by-side comparison the revision needs.
        fault_handling='preserve'
    )
    assessor.run_assessment()

    # ======================================================================
    # C-T4: IQR MULTIPLIER SENSITIVITY SWEEP
    # ======================================================================
    # The multiplier was a function default (2.5) that nothing ever varied, so
    # the sensitivity of every downstream number to it was unmeasured. Part (a)
    # -- the nullification/flag rate -- is measured here. Part (b) -- the effect
    # on final M2P-PTAC MAE -- requires re-running the benchmark against each
    # setting's masks, which full_exp_revised.py does when pointed at the
    # suffixed mask directory (see IQR_SWEEP there).
    #
    # Sweep runs are written with an _iqr<k> suffix so they cannot overwrite the
    # canonical masks the benchmark consumes.
    if RUN_IQR_SWEEP:
        print("\n" + "=" * 80)
        print("### C-T4: IQR MULTIPLIER SENSITIVITY SWEEP ###")
        print("=" * 80)
        sweep_rows = []
        for mult in IQR_SWEEP_VALUES:
            print(f"\n--- IQR multiplier = {mult} ---")
            sw = DataQualityAssessor(
                input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, plot_dir=PLOT_DIR,
                model_train_split=0.70, train_split=0.85,
                feature_warmup=72,
                causal_test_fill=True,
                reported_stations=[],          # a sweep must not halt on gates
                acknowledged_exclusions=[],
                fault_handling='preserve',
                iqr_multiplier=mult,
                output_suffix=f"_iqr{mult}")
            try:
                sw.run_assessment()
            except Exception as e:
                print(f"    [C-T4] multiplier {mult} failed: {e}")
                continue
            # Flag rate per station, read back from the masks just written.
            for f in sorted(os.listdir(OUTPUT_DIR)):
                if not f.startswith("fault_mask_") or f"_iqr{mult}.csv" not in f:
                    continue
                if any(f.startswith(f"fault_mask_{s}_") for s in
                       ('stuck', 'zero', 'excursion', 'sentinel', 'drift')):
                    continue          # per-kind files, not the aggregate
                m = pd.read_csv(os.path.join(OUTPUT_DIR, f), index_col=0)
                num = m.select_dtypes(include=['bool', 'number'])
                sweep_rows.append(dict(
                    iqr_multiplier=mult, mask_file=f,
                    n_cells=int(num.size),
                    n_flagged=int(num.astype(bool).values.sum()),
                    flag_rate=float(num.astype(bool).values.mean())))
        if sweep_rows:
            S = pd.DataFrame(sweep_rows)
            S.to_csv(os.path.join(OUTPUT_DIR, "iqr_sweep_flag_rates.csv"), index=False)
            agg = S.groupby('iqr_multiplier').flag_rate.agg(['mean', 'min', 'max'])
            print("\n[C-T4] flag rate by multiplier (mean over station files):")
            print(agg.round(5).to_string())
            print("\n[C-T4] wrote iqr_sweep_flag_rates.csv. A multiplier that "
                  "changes the flag rate a lot but leaves final MAE unchanged "
                  "means the outlier filter is not what drives the result -- "
                  "which is worth stating either way.")
