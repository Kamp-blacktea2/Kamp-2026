"""A-C: phase prediction, boundary candidates, and secondary lag diagnostics.

Quality counts are never loaded by this module.
"""

from common import *
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def errors(y, pred):
    return float(np.mean(np.abs(y - pred))), float(np.sqrt(np.mean((y - pred) ** 2)))


def predict_phase(y, phase, cut, period):
    # The target at t uses observed t-1: one-step prediction, not free-running forecast.
    design = y[:-1, None]
    if period:
        design = np.column_stack([design, np.eye(period)[phase[1:]]])
    scaler = StandardScaler().fit(design[: cut - 1])
    model = Ridge(alpha=1).fit(scaler.transform(design[: cut - 1]), y[1:cut])
    pred = model.predict(scaler.transform(design[cut - 1 :]))
    return errors(y[cut:], pred)


def phase_analysis(raw, segments):
    metrics, profiles, lags, controls, runs, voltage, templates = [], [], [], [], [], [], []
    for s in segments.itertuples():
        part = segment_part(raw, s)
        x = part[NAMES].to_numpy()
        idx = part.idx.to_numpy(int)
        base = dict(condition=s.condition, segment=s.segment, date=s.date, n=s.n)
        for j, name in enumerate(NAMES + ["four_values"]):
            z = x[:, j : j + 1] if j < 4 else x
            starts = np.r_[0, np.flatnonzero(np.any(z[1:] != z[:-1], axis=1)) + 1]
            for a, b in zip(starts, np.r_[starts[1:], len(z)]):
                runs.append(dict(base, variable=name, excel_start=s.excel_start + a, length=b - a))
        if s.n < 200:
            continue
        cut = int(s.n * 0.6)
        for j, name in enumerate(NAMES):
            y = x[:, j]
            # Training-only phase medians remove fixed position levels.
            q4 = (idx - 1) % 4
            med4 = np.array([np.median(y[:cut][q4[:cut] == q]) for q in range(4)])
            residual = y - med4[q4]
            for signal, z in (("raw", y), ("difference", np.diff(y)), ("phase_residual", residual)):
                for lag in range(1, 65):
                    if len(z) - lag >= 100:
                        lags.append(
                            dict(
                                base,
                                variable=name,
                                signal=signal,
                                lag=lag,
                                pairs=len(z) - lag,
                                correlation=corr(z[:-lag], z[lag:]),
                                exact_rate=float(np.mean(z[:-lag] == z[lag:])),
                            )
                        )
            for origin in ("idx", "segment"):
                positions = idx - 1 if origin == "idx" else np.arange(s.n)
                m0, r0 = errors(y[cut:], np.repeat(np.median(y[:cut]), s.n - cut))
                m1, r1 = predict_phase(y, positions * 0, cut, 0)
                phase_errors = {}
                for p in CONFIG["periods"]:
                    q = positions % p
                    mae, rmse = predict_phase(y, q, cut, p)
                    phase_errors[p] = mae
                    metrics.append(
                        dict(
                            base,
                            origin=origin,
                            variable=name,
                            p=p,
                            cut=cut,
                            train_last_excel=s.excel_start + cut - 1,
                            test_first_excel=s.excel_start + cut,
                            test_n=s.n - cut,
                            m0_mae=m0,
                            m1_mae=m1,
                            m2_mae=mae,
                            m0_rmse=r0,
                            m1_rmse=r1,
                            m2_rmse=rmse,
                            improvement=(m1 - mae) / m1 if m1 else np.nan,
                        )
                    )
                    for half, mask in (
                        ("train", np.arange(s.n) < cut),
                        ("test", np.arange(s.n) >= cut),
                    ):
                        for phase in range(p):
                            v = y[mask & (q == phase)]
                            profiles.append(
                                dict(
                                    base,
                                    origin=origin,
                                    variable=name,
                                    p=p,
                                    half=half,
                                    phase=phase,
                                    count=len(v),
                                    median=np.median(v),
                                    mean=np.mean(v),
                                    iqr=np.quantile(v, 0.75) - np.quantile(v, 0.25),
                                )
                            )
                if origin == "idx" and s.n >= 256:
                    rng = np.random.default_rng(42)
                    null = []
                    for trial in range(100):
                        offsets = rng.integers(0, 4, (s.n + 63) // 64)
                        q = ((idx - 1) % 4 + offsets[np.arange(s.n) // 64]) % 4
                        mae, _ = predict_phase(y, q, cut, 4)
                        null.append(mae)
                        controls.append(
                            dict(
                                base,
                                variable=name,
                                trial=trial,
                                block=64,
                                actual_mae=phase_errors[4],
                                surrogate_mae=mae,
                                actual_better=phase_errors[4] < mae,
                            )
                        )
            if name == "voltage":
                for q in range(4):
                    v = y[q4 == q]
                    for signal, z in (("raw", v), ("difference", np.diff(v))):
                        for lag in range(1, 9):
                            if len(z) >= 3 * lag and len(z) - lag >= 20:
                                voltage.append(
                                    dict(
                                        base,
                                        phase=q,
                                        signal=signal,
                                        group_lag=lag,
                                        raw_lag=4 * lag,
                                        pairs=len(z) - lag,
                                        correlation=corr(z[:-lag], z[lag:]),
                                        exact_rate=float(np.mean(z[:-lag] == z[lag:])),
                                    )
                                )
            for width in (4, 8, 16):
                for half, a, b in (("train", 0, cut), ("test", cut, s.n)):
                    z = y[a:b]
                    z = z[: len(z) // width * width].reshape(-1, width)
                    cs = [corr(v, w) for v, w in zip(z[:-1], z[1:])]
                    finite = np.array(cs)[np.isfinite(cs)]
                    templates.append(
                        dict(
                            base,
                            variable=name,
                            width=width,
                            half=half,
                            window_count=len(z),
                            finite_pairs=len(finite),
                            correlation_median=np.median(finite) if len(finite) else np.nan,
                            high_correlation_count=int(np.sum(finite >= 0.7)),
                        )
                    )
        print(f"phase {s.segment} finished", flush=True)
    for rows, name in (
        (metrics, "phase_metrics"),
        (profiles, "phase_profiles"),
        (lags, "lag_metrics"),
        (controls, "phase_controls"),
        (runs, "value_runs"),
        (voltage, "voltage_phase"),
        (templates, "window_profiles"),
    ):
        save(rows, name)


def candidates(raw, segments):
    groups, leftovers, boundaries, forces = [], [], [], []
    training_scale = raw.loc[raw.train, NAMES].std(ddof=0).to_numpy()
    for s in segments[segments.condition == "all"].itertuples():
        part = segment_part(raw, s)
        for r in range(4):
            starts = np.flatnonzero((part.idx.to_numpy(int) - 1) % 4 == r)
            starts = starts[starts + 3 < s.n]
            for condition in ("all", "controlled100"):
                used = set()
                for a in starts:
                    block = part.iloc[a : a + 4]
                    if condition == "controlled100" and not block.controlled100.all():
                        continue
                    used.update(block.excel_row.tolist())
                    groups.append(
                        dict(
                            condition=condition,
                            r=r,
                            segment=s.segment,
                            date=s.date,
                            excel_start=int(block.excel_row.iloc[0]),
                            excel_end=int(block.excel_row.iloc[-1]),
                            idx_start=int(block.idx.iloc[0]),
                            train=bool(block.train.iloc[0]),
                            candidate_id=f"{condition}_idx_r{r}_{s.segment}_{block.excel_row.iloc[0]}",
                        )
                    )
                eligible = part if condition == "all" else part[part.controlled100]
                for row in eligible.itertuples():
                    if row.excel_row not in used:
                        leftovers.append(
                            dict(
                                condition=condition,
                                r=r,
                                segment=s.segment,
                                date=s.date,
                                excel_row=row.excel_row,
                                idx=row.idx,
                            )
                        )
    group_df = pd.DataFrame(groups)
    save(group_df, "candidate_map")
    save(leftovers, "unassigned_rows")
    # Compare transitions only inside each retained contiguous segment.
    for s in segments.itertuples():
        part = segment_part(raw, s)
        d = np.abs(np.diff(part[NAMES].to_numpy(), axis=0))
        incoming = (part.idx.to_numpy(int)[1:] - 1) % 4
        for r in range(4):
            for j, name in enumerate(NAMES):
                inside, outside = d[incoming != r, j], d[incoming == r, j]
                if len(inside) and len(outside):
                    boundaries.append(
                        dict(
                            condition=s.condition,
                            segment=s.segment,
                            date=s.date,
                            r=r,
                            variable=name,
                            internal_n=len(inside),
                            boundary_n=len(outside),
                            internal_mean=np.mean(inside),
                            boundary_mean=np.mean(outside),
                            internal_median=np.median(inside),
                            boundary_median=np.median(outside),
                            standardized_difference=(np.mean(outside) - np.mean(inside))
                            / training_scale[j],
                        )
                    )
        f = part.force.to_numpy()
        for threshold in CONFIG["force_thresholds"]:
            mask = f > threshold
            starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
            ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
            for a, b in zip(starts, ends):
                lo, hi = s.excel_start + a, s.excel_start + b
                for r in range(4):
                    g = group_df[(group_df.condition == s.condition) & (group_df.r == r)]
                    contained = (g.excel_start >= lo) & (g.excel_end <= hi)
                    touched = (g.excel_start <= hi) & (g.excel_end >= lo)
                    forces.append(
                        dict(
                            condition=s.condition,
                            segment=s.segment,
                            date=s.date,
                            threshold=threshold,
                            excel_start=lo,
                            excel_end=hi,
                            length=b - a + 1,
                            start_phase=int((part.idx.iloc[a] - 1) % 4),
                            end_phase=int((part.idx.iloc[b] - 1) % 4),
                            length_mod4=(b - a + 1) % 4,
                            r=r,
                            complete_groups=int(contained.sum()),
                            partial_groups=int((touched & ~contained).sum()),
                            aligned=int((b - a + 1) % 4 == 0 and (part.idx.iloc[a] - 1) % 4 == r),
                            phase_shift_alignment_reference=0.25 if (b - a + 1) % 4 == 0 else 0,
                        )
                    )
    save(boundaries, "boundary_candidates")
    save(forces, "force_runs")
    counts = group_df.groupby(["condition", "r", "date"]).size().rename("groups").reset_index()
    remaining = (
        pd.DataFrame(leftovers)
        .groupby(["condition", "r", "date"])
        .size()
        .rename("unassigned")
        .reset_index()
    )
    counts = counts.merge(remaining, how="outer").fillna(0)
    counts["records"] = 4 * counts.groups + counts.unassigned
    save(counts, "candidate_counts")


def main():
    raw, segments = load()
    json_save(CONFIG, HERE / "config.json")
    save(
        [
            dict(
                variable=c,
                unique=raw[c].nunique(),
                missing=raw[c].isna().sum(),
                first=str(raw[c].iloc[0]),
            )
            for c in raw.columns
        ],
        "input_audit",
    )
    candidates(raw, segments)
    phase_analysis(raw, segments)
    print("STRUCTURE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
