"""Independent reconciliations from raw rows, saved fits, and recorded predictions."""

from common import *
import joblib
import platform
from threadpoolctl import threadpool_limits


def main():
    raw, segments = load()
    x = raw[NAMES].to_numpy()
    groups, unused = read("candidate_map"), read("unassigned_rows")
    checks = []

    def check(name, assertion, detail=None):
        passed = bool(assertion)
        checks.append(dict(check=name, passed=passed, detail=detail))
        if not passed:
            raise AssertionError(f"{name}: {detail}")

    check(
        "raw unchanged", digest(ROOT / "data" / "Welding_Data_Set_01.xlsx") == CONFIG["raw_sha256"]
    )
    check(
        "dictionary unchanged",
        digest(ROOT / "docs" / "data_dictionary.md") == CONFIG["dictionary_sha256"],
    )
    for (condition, r), part in groups.groupby(["condition", "r"]):
        assigned = np.concatenate(
            [np.arange(a, b + 1) for a, b in zip(part.excel_start, part.excel_end)]
        )
        residual = unused.loc[
            (unused.condition == condition) & (unused.r == r), "excel_row"
        ].to_numpy()
        expected = (
            raw.excel_row.to_numpy()
            if condition == "all"
            else raw.loc[raw.controlled100, "excel_row"].to_numpy()
        )
        combined = np.r_[assigned, residual]
        check(
            f"coverage {condition} r{r}",
            np.array_equal(np.sort(combined), expected)
            and len(np.unique(combined)) == len(combined),
            dict(groups=len(part), rows=len(combined)),
        )
        for row in part.itertuples():
            block = raw.iloc[row.excel_start - 2 : row.excel_end - 1]
            assert len(block) == 4 and np.all(np.diff(block.idx) == 1)
            assert block.date.nunique() == 1 and block.train.nunique() == 1
            assert (block.idx.iloc[0] - 1) % 4 == r
        if condition == "controlled100":
            original = groups[(groups.condition == "all") & (groups.r == r)]
            expected_starts = [
                a for a in original.excel_start if raw.iloc[a - 2 : a + 2].controlled100.all()
            ]
            check(
                f"controlled original groups r{r}",
                sorted(expected_starts) == sorted(part.excel_start),
            )
    check("all candidate continuity and dates", True, len(groups))

    metrics = read("phase_metrics")
    rotated = metrics[metrics.origin == "idx"].merge(
        metrics[metrics.origin == "segment"],
        on=["condition", "segment", "variable", "p"],
        suffixes=("_idx", "_segment"),
    )
    check(
        "phase rotation prediction invariance",
        np.allclose(rotated.m2_mae_idx, rotated.m2_mae_segment, atol=1e-10),
    )
    # Solve regularized normal equations directly, independently of Ridge/predict_phase.
    for row in metrics[(metrics.origin == "idx") & metrics.p.isin([2, 4, 16])].itertuples():
        s = segments[segments.segment == row.segment].iloc[0]
        part = raw.iloc[int(s.excel_start) - 2 : int(s.excel_end) - 1]
        y = part[row.variable].to_numpy()
        phase = (part.idx.to_numpy(int) - 1) % row.p
        a = np.column_stack([y[:-1], np.eye(row.p)[phase[1:]]])
        cut = int(row.cut)
        mean, scale = a[: cut - 1].mean(axis=0), a[: cut - 1].std(axis=0)
        scale[scale == 0] = 1
        a = np.column_stack([np.ones(len(a)), (a - mean) / scale])
        penalty = np.eye(a.shape[1])
        penalty[0, 0] = 0
        beta = np.linalg.solve(a[: cut - 1].T @ a[: cut - 1] + penalty, a[: cut - 1].T @ y[1:cut])
        pred = a[cut - 1 :] @ beta
        assert np.isclose(np.mean(np.abs(y[cut:] - pred)), row.m2_mae, atol=1e-9)
    check("independent phase regression", True, "all idx p2/p4/p16 settings")
    lag_table = read("lag_metrics")
    for row in lag_table[
        (lag_table.signal == "difference") & lag_table.lag.isin([4, 8, 12, 16])
    ].itertuples():
        s = segments[segments.segment == row.segment].iloc[0]
        y = np.diff(
            raw.iloc[int(s.excel_start) - 2 : int(s.excel_end) - 1][row.variable].to_numpy()
        )
        value = (
            np.corrcoef(y[: -row.lag], y[row.lag :])[0, 1]
            if np.std(y[: -row.lag]) > 1e-12 and np.std(y[row.lag :]) > 1e-12
            else np.nan
        )
        assert np.isclose(value, row.correlation, equal_nan=True, atol=1e-10)
    check("difference lag direct recomputation", True)
    force = read("force_runs")
    for row in force.itertuples():
        part = raw.iloc[row.excel_start - 2 : row.excel_end - 1]
        assert len(part) == row.length and (part.force > row.threshold).all()
        assert (part.idx.iloc[0] - 1) % 4 == row.start_phase
    check("force runs source values", True, len(force))

    # Integer tuple IDs avoid reusing the byte-string matching implementation.
    values_to_id = {}
    ids = []
    for row in x:
        key = tuple(row)
        ids.append(values_to_id.setdefault(key, len(values_to_id)))
    overlaps = read("overlap_rows")
    for width in (1, 4, 16, 100):
        train_keys = set()
        test_windows = []
        for s in segments[segments.condition == "all"].itertuples():
            positions = list(range(s.excel_start - 2, s.excel_end - 1))
            for offset in range(len(positions) - width + 1):
                indices = positions[offset : offset + width]
                key = tuple(ids[i] for i in indices)
                if s.date <= CONFIG["train_last_date"]:
                    train_keys.add(key)
                else:
                    test_windows.append((indices, key))
        covered = set()
        for indices, key in test_windows:
            if key in train_keys:
                covered.update(indices)
        check(
            f"independent overlap width{width}",
            covered == set(np.flatnonzero(overlaps[f"overlap_{width}"])),
            len(covered),
        )

    from sklearn.preprocessing import StandardScaler, RobustScaler

    candidate_scores = read("candidate_scores")
    for path in (HERE / "outputs" / "models").glob("*.joblib"):
        bundle = joblib.load(path)
        if "gmm" in bundle:
            table = candidate_scores[candidate_scores.key == path.stem]
            starts = table.excel_start.to_numpy(int) - 2
            v = x[starts[:, None] + np.arange(4)]
            features = (
                np.column_stack([v.mean(axis=1), np.ptp(v, axis=1)])
                if bundle["features"] == "S8"
                else v.reshape(len(v), 16)
            )
            train = table.train.to_numpy(bool)
            check(
                f"fit rows {path.stem}",
                bundle["train_ids"] == table.loc[train, "candidate_id"].tolist(),
            )
            scale_class = StandardScaler if path.stem.endswith("standard") else RobustScaler
            independently_fitted = scale_class().fit(features[train])
            check(
                f"train-only scale {path.stem}",
                np.allclose(
                    independently_fitted.transform(features), bundle["scaler"].transform(features)
                ),
            )
            z = bundle["scaler"].transform(features)
            assert np.array_equal(bundle["gmm"].predict(z), table.state)
            if "if" in bundle:
                score = -bundle["if"].score_samples(z)
                assert np.allclose(score, table.if_score)
                assert np.isclose(np.quantile(score[train], 0.95), table.if_threshold95.iloc[0])
        else:
            recorded = read("row_scores")
            score = -bundle["model"].score_samples(bundle["scaler"].transform(x))
            assert np.allclose(score, recorded[f"{path.stem}_score"])
            assert bundle["train_excel_rows"] == raw.loc[raw.train, "excel_row"].tolist()
    check("serialized candidate predictions and IF thresholds", True)
    scores, models = read("row_scores"), read("row_models")
    train = raw.train.to_numpy(bool)
    for row in models.itertuples():
        score = scores[f"{row.model}_score"].to_numpy()
        threshold = (
            score[train].mean() + 8 * score[train].std(ddof=0)
            if row.model.startswith("ae")
            else np.quantile(score[train], 0.95)
        )
        assert np.isclose(threshold, row.threshold, atol=1e-12)
        flags = score >= row.threshold
        assert np.array_equal(flags, scores[f"{row.model}_flag"])
        assert flags[train].sum() == row.train_flags and flags[~train].sum() == row.test_flags
        if row.model.startswith("ae"):
            arrays = np.load(HERE / "outputs" / "models" / f"{row.model}_reconstruction.npz")
            expected_input = ((x - x[train].min(axis=0)) / np.ptp(x[train], axis=0)).astype(
                np.float32
            )
            assert np.allclose(expected_input, arrays["input"], atol=1e-7)
            assert np.allclose(
                ((arrays["input"] - arrays["output"]) ** 2).mean(axis=1), score, atol=1e-10
            )
    check("AE input scaling, reconstruction MSE and fixed thresholds", True)
    check("IF fixed thresholds and score counts", True)
    audit = read("result_audit")
    original = pd.read_excel(ROOT / "data" / "Welding_Data_Set_01.xlsx", sheet_name="result")
    check(
        "Result original counts and types",
        np.array_equal(audit.defect, original.defect)
        and np.array_equal(audit["defect type"], original["defect type"]),
    )
    daily = read("daily_quality_comparison")
    check(
        "missing is not zero",
        daily.loc[daily.date == "2020-03-27", "type_1"].isna().all()
        and daily.loc[daily.date == "2020-03-31", "type_3"].isna().all()
        and daily.loc[daily.date == "2020-04-07", "type_1"].iloc[0] == 0,
    )
    for day in daily.itertuples():
        for name in models.model:
            expected = scores.loc[scores.date == day.date, f"{name}_flag"].sum()
            assert getattr(day, f"{name}_flag_count") == expected
    check("daily score aggregations", True)
    candidate_rows = read("candidate_row_scores")
    for name in models.model:
        values = scores[f"{name}_score"].to_numpy()[
            candidate_rows.excel_start.to_numpy(int)[:, None] - 2 + np.arange(4)
        ]
        assert np.allclose(values.mean(axis=1), candidate_rows[f"{name}_score_mean"])
        assert np.allclose(values.max(axis=1), candidate_rows[f"{name}_score_max"])
    check("candidate mean/max aggregations", True)
    for row in read("short_segment_lags").itertuples():
        s = segments[segments.segment == row.segment].iloc[0]
        y = raw.iloc[int(s.excel_start) - 2 : int(s.excel_end) - 1][row.variable].to_numpy()
        if row.signal == "difference":
            y = y[1:] - y[:-1]
        assert len(y) - row.lag == row.pairs and row.pairs >= 100
        value = np.corrcoef(y[: -row.lag], y[row.lag :])[0, 1]
        assert np.isclose(value, row.correlation, equal_nan=True, atol=1e-10)
    check("short segment lag pair counts and correlations", True)
    candidate_counts = read("candidate_counts")
    for day in daily.itertuples():
        for r in range(4):
            n = len(
                groups[(groups.condition == "all") & (groups.r == r) & (groups.date == day.date)]
            )
            assert getattr(day, f"r{r}_groups") == n
    check("daily candidate counts", True)
    check(
        "no quality labels in model inputs",
        not any(c.startswith("type_") or c == "defect" for c in groups.columns),
    )
    json_save(
        {"passed": True, "checks": checks, "check_count": len(checks)},
        HERE / "outputs" / "verification.json",
    )
    import sklearn, scipy

    manifest = dict(
        completed=True,
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        scipy=scipy.__version__,
        sklearn=sklearn.__version__,
        config=CONFIG,
        device="CPU",
        threads=4,
        raw_sha256=digest(ROOT / "data" / "Welding_Data_Set_01.xlsx"),
        dictionary_sha256=digest(ROOT / "docs" / "data_dictionary.md"),
        code_sha256={p.name: digest(p) for p in HERE.glob("*.py")},
        model_sha256={p.name: digest(p) for p in (HERE / "outputs" / "models").iterdir()},
        table_sha256={p.name: digest(p) for p in TABLES.glob("*.csv")},
        plan_sha256=digest(HERE / "실험계획.md"),
        source_pdf="not found in project",
        verification="outputs/verification.json",
    )
    json_save(manifest, HERE / "outputs" / "manifest.json")
    print(f"VERIFIED {len(checks)} checks", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
