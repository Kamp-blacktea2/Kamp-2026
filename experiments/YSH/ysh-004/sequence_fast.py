"""Exact L2 change-point objective and blocked within-segment nearest-window search."""

import time

import numpy as np
import pandas as pd

from analyze import HERE, NAMES, X3, condition_indices, load_config, load_inputs, save, scale


def l2_partition(z, min_size, penalty):
    """Solve the same penalized L2 objective as PELT using vectorized exact DP."""
    n, d = z.shape
    if n < 2 * min_size:
        return []
    prefix = np.vstack([np.zeros(d), np.cumsum(z, axis=0)])
    squared = np.r_[0, np.cumsum(np.sum(z * z, axis=1))]
    best = np.full(n + 1, np.inf)
    previous = np.full(n + 1, -1, int)
    best[0] = -penalty
    for end in range(min_size, n + 1):
        starts = np.arange(end - min_size + 1)
        valid = np.isfinite(best[starts])
        starts = starts[valid]
        sums = prefix[end] - prefix[starts]
        cost = squared[end] - squared[starts] - np.sum(sums * sums, axis=1) / (end - starts)
        total = best[starts] + cost + penalty
        which = int(np.argmin(total))
        best[end] = total[which]
        previous[end] = starts[which]
    points = []
    end = n
    while previous[end] > 0:
        points.append(int(previous[end]))
        end = int(previous[end])
    return points[::-1]


def scan_windows(part, j, m):
    windows = np.lib.stride_tricks.sliding_window_view(part[:, j], m)
    count = len(windows)
    centered = windows - windows.mean(axis=1, keepdims=True)
    norms = np.sum(centered * centered, axis=1)
    _, ids = np.unique(windows, axis=0, return_inverse=True)
    nearest = np.full(count, np.inf)
    where = np.full(count, -1, int)
    nonexact = np.full(count, np.inf)
    where_nonexact = np.full(count, -1, int)
    all_indices = np.arange(count)[None, :]
    for start in range(0, count, 128):
        stop = min(count, start + 128)
        distances = norms[start:stop, None] + norms[None, :] - 2 * centered[start:stop] @ centered.T
        np.maximum(distances, 0, out=distances)
        distances[np.abs(np.arange(start, stop)[:, None] - all_indices) < m] = np.inf
        pos = np.argmin(distances, axis=1)
        nearest[start:stop] = np.sqrt(distances[np.arange(stop - start), pos] / m)
        where[start:stop] = pos
        distances[ids[start:stop, None] == ids[None, :]] = np.inf
        pos = np.argmin(distances, axis=1)
        nonexact[start:stop] = np.sqrt(distances[np.arange(stop - start), pos] / m)
        where_nonexact[start:stop] = pos
    return windows, nearest, where, nonexact, where_nonexact


def main(config=None, values=None, segments=None):
    started = time.perf_counter()
    if config is None:
        config = load_config(HERE / "config.json")
    if values is None or segments is None:
        _, values, segments, _, _ = load_inputs(config)
    summary = []
    boundaries = []
    effects = []
    motif_rows = []
    motif_summary = []
    for condition in ("all", "controlled100"):
        idx = condition_indices(segments, condition, values)
        common = {
            rep: scale(values[idx][:, cols])[0] for rep, cols in (("X4", [0, 1, 2, 3]), ("X3", X3))
        }
        for r in segments[segments.condition == condition].itertuples():
            part = values[r.excel_start - 2 : r.excel_end - 1]
            print(
                f"SEQUENCE {condition} {r.segment} n={len(part)} elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
            for rep, cols in (("X4", [0, 1, 2, 3]), ("X3", X3)):
                z = common[rep].transform(part[:, cols])
                dimension = z.shape[1]
                for min_size, coef in ((32, 3), (16, 3), (64, 3), (32, 1), (32, 10)):
                    setting = f"min{min_size}_pen{coef}"
                    common_fields = dict(
                        condition=condition,
                        model="PELT",
                        representation=rep,
                        fit_id=f"{condition}_{rep}_standard",
                        n=len(part),
                        setting_id=setting,
                        segment_id=r.segment,
                        excel_start=r.excel_start,
                        excel_end=r.excel_end,
                    )
                    if len(part) < 2 * min_size:
                        summary.append(
                            dict(**common_fields, reason="segment shorter than 2*min_size")
                        )
                        continue
                    points = l2_partition(z, min_size, coef * dimension * np.log(len(part)))
                    summary.append(dict(**common_fields, boundaries=len(points)))
                    for point in points:
                        excel = r.excel_start + point
                        boundaries.append(dict(**common_fields, boundary_excel=excel))
                        if setting == "min32_pen3":
                            left = part[max(0, point - 32) : point]
                            right = part[point : min(len(part), point + 32)]
                            row = dict(
                                **common_fields,
                                boundary_excel=excel,
                                n_before=len(left),
                                n_after=len(right),
                            )
                            for j, name in enumerate(NAMES):
                                row[f"{name}_mean_delta"] = float(
                                    right[:, j].mean() - left[:, j].mean()
                                )
                                row[f"{name}_median_delta"] = float(
                                    np.median(right[:, j]) - np.median(left[:, j])
                                )
                                row[f"{name}_iqr_before"] = float(
                                    np.subtract(*np.percentile(left[:, j], [75, 25]))
                                )
                                row[f"{name}_iqr_after"] = float(
                                    np.subtract(*np.percentile(right[:, j], [75, 25]))
                                )
                                row[f"{name}_madiff_before"] = float(
                                    np.mean(np.abs(np.diff(left[:, j])))
                                )
                                row[f"{name}_madiff_after"] = float(
                                    np.mean(np.abs(np.diff(right[:, j])))
                                )
                            effects.append(row)
            for j, name in enumerate(NAMES):
                lengths = [16, 32, 64] + ([15, 17] if name == "voltage" else [])
                for m in lengths:
                    base = dict(
                        condition=condition,
                        model="motif",
                        representation=name,
                        fit_id=r.segment,
                        n=len(part),
                        setting_id=f"m{m}",
                        segment_id=r.segment,
                        excel_start=r.excel_start,
                        excel_end=r.excel_end,
                    )
                    if len(part) < 2 * m:
                        motif_summary.append(dict(**base, reason="no nonoverlap pair"))
                        continue
                    windows, nearest, where, nonexact, where_nonexact = scan_windows(part, j, m)
                    finite = np.isfinite(nonexact)
                    motif_summary.append(
                        dict(
                            **base,
                            windows=len(windows),
                            exact_nearest=int(np.sum(np.all(windows == windows[where], axis=1))),
                            nonexact_available=int(finite.sum()),
                            nearest_median=float(np.median(nearest[np.isfinite(nearest)])),
                            nonexact_median=(
                                float(np.median(nonexact[finite])) if finite.any() else np.nan
                            ),
                            nonexact_p95=(
                                float(np.quantile(nonexact[finite], 0.95))
                                if finite.any()
                                else np.nan
                            ),
                        )
                    )
                    for kind, order in (
                        ("similar", np.argsort(nonexact)),
                        ("dissimilar", np.argsort(-nonexact)),
                    ):
                        selected = []
                        selected_pairs = 0
                        for pos in order:
                            if not finite[pos]:
                                continue
                            other = int(where_nonexact[pos])
                            if any(
                                abs(candidate - old) < m
                                for candidate in (pos, other)
                                for old in selected
                            ):
                                continue
                            a = windows[pos]
                            b = windows[other]
                            ac = a - a.mean()
                            bc = b - b.mean()
                            an = np.linalg.norm(ac)
                            bn = np.linalg.norm(bc)
                            row = dict(
                                **{**base, "setting_id": f"m{m}_{kind}"},
                                a_excel=int(r.excel_start + pos),
                                b_excel=int(r.excel_start + other),
                                distance=float(nonexact[pos]),
                                nearest_any=float(nearest[pos]),
                                nearest_any_exact=bool(np.array_equal(a, windows[where[pos]])),
                                exact_all=bool(
                                    np.array_equal(part[pos : pos + m], part[other : other + m])
                                ),
                                corr=float(ac @ bc / (an * bn)) if an * bn > 1e-15 else np.nan,
                                raw_rmse=float(np.sqrt(np.mean((a - b) ** 2))),
                                amplitude=float(bn / an) if an > 0 else np.nan,
                            )
                            for jj, nm in enumerate(NAMES):
                                row[f"a_{nm}_median"] = float(np.median(part[pos : pos + m, jj]))
                                row[f"b_{nm}_median"] = float(
                                    np.median(part[other : other + m, jj])
                                )
                            motif_rows.append(row)
                            selected.extend([int(pos), other])
                            selected_pairs += 1
                            if selected_pairs == 10:
                                break
        # Checkpoints distinguish a partial computation from a completed run.
        save(summary, "sequence_summary.partial.csv")
        save(boundaries, "change_points.partial.csv")
        save(motif_summary, "motif_summary.partial.csv")
    save(summary, "sequence_summary.csv")
    save(boundaries, "change_points.csv")
    save(effects, "boundary_effects.csv")
    save(motif_rows, "motif_pairs.csv")
    save(motif_summary, "motif_summary.csv")
    print(f"DONE sequence {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
