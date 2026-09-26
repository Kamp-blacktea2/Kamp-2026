"""Supplementary diagnostics specified in the plan, without quality selection."""

from common import *


def main():
    raw, segments = load()
    x = raw[NAMES].to_numpy()
    rows, group_scores = read("row_scores"), read("candidate_scores")
    group_overlap = read("candidate_row_scores")
    extras, boundary, values, anomaly = [], [], [], []
    for s in segments.itertuples():
        part = segment_part(raw, s)
        y = part[NAMES].to_numpy()
        for j, name in enumerate(NAMES):
            for value, n in part[name].value_counts().items():
                if name == "time":
                    values.append(
                        dict(
                            condition=s.condition,
                            segment=s.segment,
                            variable=name,
                            value=value,
                            count=n,
                        )
                    )
            if 100 < s.n < 200:
                for signal, z in (("raw", y[:, j]), ("difference", np.diff(y[:, j]))):
                    for lag in range(1, min(64, len(z) - 100) + 1):
                        extras.append(
                            dict(
                                condition=s.condition,
                                segment=s.segment,
                                date=s.date,
                                variable=name,
                                signal=signal,
                                lag=lag,
                                pairs=len(z) - lag,
                                correlation=corr(z[:-lag], z[lag:]),
                                exact_rate=np.mean(z[:-lag] == z[lag:]),
                            )
                        )
            if s.n >= 100:
                d = np.abs(np.diff(y[:, j]))
                for p in (2, 8, 16):
                    phase = (part.idx.to_numpy(int)[1:] - 1) % p
                    for r in range(p):
                        a, b = d[phase == r], d[phase != r]
                        boundary.append(
                            dict(
                                condition=s.condition,
                                segment=s.segment,
                                variable=name,
                                width=p,
                                r=r,
                                boundary_mean=a.mean(),
                                internal_mean=b.mean(),
                                boundary_n=len(a),
                                internal_n=len(b),
                            )
                        )
    save(extras, "short_segment_lags")
    save(boundary, "boundary_width_sensitivity")
    save(values, "time_value_counts")
    train_first = {}
    for row in raw[raw.train].itertuples():
        train_first.setdefault(tuple(getattr(row, name) for name in NAMES), row.excel_row)
    for name in ("ae_42", "if_standard_42"):
        for index in np.flatnonzero(rows[f"{name}_flag"].to_numpy(bool)):
            record = dict(
                model=name,
                excel_row=index + 2,
                date=raw.date.iloc[index],
                split="train" if raw.train.iloc[index] else "test",
                first_train_exact_excel=train_first.get(tuple(x[index])),
                score=rows[f"{name}_score"].iloc[index],
            )
            record.update(dict(zip(NAMES, x[index])))
            anomaly.append(record)
    save(anomaly, "flagged_row_details")
    purged = []
    for key, part in group_scores.groupby("key"):
        part = part.merge(
            group_overlap[["candidate_id"] + [f"overlap_{w}_any" for w in (1, 4, 16, 100)]],
            on="candidate_id",
        )
        for width in (1, 4, 16, 100):
            subset = part[~part.train & ~part[f"overlap_{width}_any"]]
            purged.append(
                dict(
                    key=key,
                    width=width,
                    remaining=len(subset),
                    density_flags=int(subset.density_flag99.sum()),
                    if_flags95=int(subset.if_flag95.sum()) if "_S8_" in key else np.nan,
                )
            )
    save(purged, "purged_candidate_models")
    g = read("candidate_map")
    summaries = []
    for (condition, r), part in g.groupby(["condition", "r"]):
        z = group_values(raw, part)
        for j, name in enumerate(NAMES):
            for phase in range(4):
                for split in (True, False):
                    a = z[part.train.to_numpy(bool) == split, phase, j]
                    summaries.append(
                        dict(
                            condition=condition,
                            r=r,
                            slot=phase,
                            variable=name,
                            split="train" if split else "test",
                            n=len(a),
                            median=np.median(a) if len(a) else np.nan,
                            iqr=np.quantile(a, 0.75) - np.quantile(a, 0.25) if len(a) else np.nan,
                        )
                    )
    save(summaries, "candidate_slot_profiles")
    print("DIAGNOSTICS COMPLETE")


if __name__ == "__main__":
    main()
