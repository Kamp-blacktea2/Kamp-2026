"""D-E: train-only candidate models and official-architecture AE reference."""

from common import *
import itertools
import joblib
import time
import warnings
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from threadpoolctl import threadpool_limits

MODELS = HERE / "outputs" / "models"


def overlap(raw, segments):
    x = raw[NAMES].to_numpy()
    flags = raw[["excel_row", "date", "train"]].copy()
    audit = []
    for width in (1, 4, 16, 100):
        keys = set()
        for s in segments[segments.condition == "all"].itertuples():
            if s.date > CONFIG["train_last_date"]:
                continue
            for a in range(s.excel_start - 2, s.excel_end - width):
                keys.add(x[a : a + width].tobytes())
        covered = np.zeros(len(raw), bool)
        matches = eligible = 0
        for s in segments[segments.condition == "all"].itertuples():
            if s.date <= CONFIG["train_last_date"]:
                continue
            for a in range(s.excel_start - 2, s.excel_end - width):
                eligible += 1
                if x[a : a + width].tobytes() in keys:
                    matches += 1
                    covered[a : a + width] = True
        flags[f"overlap_{width}"] = covered
        audit.append(
            dict(
                width=width,
                eligible_test_windows=eligible,
                matching_test_windows=matches,
                covered_test_rows=int(covered.sum()),
                test_rows=int((~raw.train).sum()),
                remaining_test_rows=int((~raw.train & ~covered).sum()),
            )
        )
    save(flags, "overlap_rows")
    save(audit, "overlap_audit")
    return flags


def candidate_models(raw):
    groups = read("candidate_map")
    scores, summary, grids, sensitivity, profiles, fit_stats = [], [], [], [], [], []
    for (condition, r), part in groups.groupby(["condition", "r"], sort=False):
        part = part.reset_index(drop=True)
        values = group_values(raw, part)
        representations = {
            "S8": np.column_stack([values.mean(axis=1), np.ptp(values, axis=1)]),
            "P16": values.reshape(len(values), 16),
        }
        train = part.train.to_numpy(bool)
        reference_labels = {}
        for representation, features in representations.items():
            if train.sum() < 100 or len(np.unique(features[train], axis=0)) < 20:
                summary.append(
                    dict(
                        condition=condition,
                        r=r,
                        representation=representation,
                        status="insufficient training groups",
                        train_n=int(train.sum()),
                    )
                )
                continue
            for scale_name, scaler_class in (
                ("standard", StandardScaler),
                ("robust", RobustScaler),
            ):
                key = f"{condition}_r{r}_{representation}_{scale_name}"
                scaler = scaler_class().fit(features[train])
                z = scaler.transform(features)
                models, bics = [], []
                for k in range(1, 5):
                    model = GaussianMixture(
                        k,
                        covariance_type="full",
                        reg_covar=1e-3,
                        n_init=5,
                        max_iter=500,
                        random_state=42,
                    ).fit(z[train])
                    bic = model.bic(z[train])
                    models.append(model)
                    bics.append(bic)
                    grids.append(dict(key=key, k=k, bic=bic, converged=bool(model.converged_)))
                chosen = next(k for k, bic in enumerate(bics) if bic <= min(bics) + 2)
                model = models[chosen]
                labels = model.predict(z)
                reference_labels[(representation, scale_name)] = labels
                summary.append(
                    dict(
                        key=key,
                        condition=condition,
                        r=r,
                        representation=representation,
                        scale=scale_name,
                        k=chosen + 1,
                        train_n=int(train.sum()),
                        test_n=int((~train).sum()),
                        unique_train=len(np.unique(features[train], axis=0)),
                        bic=bics[chosen],
                        status="fitted",
                        converged=bool(model.converged_),
                    )
                )
                frame = part.copy()
                frame["key"] = key
                frame["state"] = labels
                frame["posterior"] = model.predict_proba(z).max(axis=1)
                frame["negative_log_density"] = -model.score_samples(z)
                density_threshold = np.quantile(frame.loc[train, "negative_log_density"], 0.99)
                frame["density_threshold99"] = density_threshold
                frame["density_flag99"] = frame.negative_log_density >= density_threshold
                bundle = {
                    "scaler": scaler,
                    "gmm": model,
                    "train_ids": part.loc[train, "candidate_id"].tolist(),
                    "features": representation,
                    "density_threshold99": density_threshold,
                }
                if representation == "S8":
                    forest = IsolationForest(
                        n_estimators=300,
                        max_samples=min(256, train.sum()),
                        random_state=42,
                        n_jobs=4,
                    ).fit(z[train])
                    score = -forest.score_samples(z)
                    frame["if_score"] = score
                    for q in (0.95, 0.99):
                        threshold = np.quantile(score[train], q)
                        frame[f"if_threshold{int(q * 100)}"] = threshold
                        frame[f"if_flag{int(q * 100)}"] = score >= threshold
                    bundle["if"] = forest
                    for seed in (7, 2026):
                        f2 = IsolationForest(
                            n_estimators=300,
                            max_samples=min(256, train.sum()),
                            random_state=seed,
                            n_jobs=4,
                        ).fit(z[train])
                        v2 = -f2.score_samples(z)
                        a, b = score >= np.quantile(score[train], 0.95), v2 >= np.quantile(
                            v2[train], 0.95
                        )
                        sensitivity.append(
                            dict(
                                key=key,
                                comparison=f"if_seed{seed}",
                                ari=np.nan,
                                spearman=spearmanr(score, v2).statistic,
                                jaccard=np.sum(a & b) / np.sum(a | b),
                            )
                        )
                scores.append(frame)
                for state in range(chosen + 1):
                    for split, eligible in (("train", train), ("test", ~train)):
                        v = values[(labels == state) & eligible]
                        if len(v):
                            row = dict(key=key, state=state, split=split, n_groups=len(v))
                            for j, name in enumerate(NAMES):
                                row[f"{name}_median"] = float(np.median(v[:, :, j]))
                                row[f"{name}_mean_range"] = float(
                                    np.mean(np.ptp(v[:, :, j], axis=1))
                                )
                            profiles.append(row)
                for seed in (7, 2026):
                    other = GaussianMixture(
                        chosen + 1, reg_covar=1e-3, n_init=5, max_iter=500, random_state=seed
                    ).fit(z[train])
                    for split, mask in (("all", np.ones(len(z), bool)), ("test", ~train)):
                        if mask.sum() >= 2:
                            sensitivity.append(
                                dict(
                                    key=key,
                                    comparison=f"gmm_seed{seed}_{split}",
                                    ari=adjusted_rand_score(labels[mask], other.predict(z[mask])),
                                    spearman=np.nan,
                                    jaccard=np.nan,
                                )
                            )
                joblib.dump(bundle, MODELS / f"{key}.joblib", compress=3)
                fit_stats.append(
                    dict(
                        key=key,
                        train_excel_starts=part.loc[train, "excel_start"].tolist(),
                        center=getattr(scaler, "mean_", getattr(scaler, "center_", None)).tolist(),
                        scale=scaler.scale_.tolist(),
                    )
                )
            a, b = (
                reference_labels[(representation, "standard")],
                reference_labels[(representation, "robust")],
            )
            sensitivity.append(
                dict(
                    key=f"{condition}_r{r}_{representation}",
                    comparison="gmm_scale",
                    ari=adjusted_rand_score(a, b),
                    spearman=np.nan,
                    jaccard=np.nan,
                )
            )
        print(f"candidate models {condition} r{r} finished", flush=True)
    score_table = pd.concat(scores, ignore_index=True)
    # Same original candidate correspondence, not concatenating independent samples.
    for r, rep, scale in itertools.product(range(4), ("S8", "P16"), ("standard", "robust")):
        a = score_table[score_table.key == f"all_r{r}_{rep}_{scale}"]
        b = score_table[score_table.key == f"controlled100_r{r}_{rep}_{scale}"]
        matched = a.merge(b, on="excel_start", suffixes=("_all", "_controlled"))
        if len(matched) >= 2:
            sensitivity.append(
                dict(
                    key=f"r{r}_{rep}_{scale}",
                    comparison="all_vs_controlled100",
                    ari=adjusted_rand_score(matched.state_all, matched.state_controlled),
                    n=len(matched),
                    spearman=np.nan,
                    jaccard=np.nan,
                )
            )
    save(score_table, "candidate_scores")
    for rows, name in (
        (summary, "model_summary"),
        (grids, "gmm_bic"),
        (profiles, "candidate_profiles"),
        (sensitivity, "candidate_sensitivity"),
    ):
        save(rows, name)
    json_save(fit_stats, HERE / "outputs" / "candidate_fit.json")


def row_models(raw):
    x = raw[NAMES].to_numpy()
    train = raw.train.to_numpy(bool)
    result = raw[["excel_row", "date", "train"]].copy()
    model_rows, learning = [], []
    for scale_name, scaler_class in (("standard", StandardScaler), ("robust", RobustScaler)):
        scaler = scaler_class().fit(x[train])
        z = scaler.transform(x)
        for seed in CONFIG["seeds"]:
            forest = IsolationForest(
                n_estimators=300, max_samples=256, random_state=seed, n_jobs=4
            ).fit(z[train])
            score = -forest.score_samples(z)
            name = f"if_{scale_name}_{seed}"
            threshold = np.quantile(score[train], 0.95)
            result[f"{name}_score"] = score
            result[f"{name}_flag"] = score >= threshold
            model_rows.append(
                dict(
                    model=name,
                    threshold=threshold,
                    train_flags=int((score[train] >= threshold).sum()),
                    test_flags=int((score[~train] >= threshold).sum()),
                    train_score_mean=score[train].mean(),
                    train_score_std=score[train].std(),
                )
            )
            joblib.dump(
                {
                    "model": forest,
                    "scaler": scaler,
                    "threshold": threshold,
                    "train_excel_rows": raw.loc[train, "excel_row"].tolist(),
                },
                MODELS / f"{name}.joblib",
                compress=3,
            )
    try:
        import torch
    except ImportError as error:
        json_save({"status": "blocked", "reason": str(error)}, HERE / "outputs" / "ae_status.json")
    else:
        torch.set_num_threads(4)
        scaler = MinMaxScaler().fit(x[train])
        z = torch.tensor(scaler.transform(x), dtype=torch.float32)
        train_tensor = z[train]
        json_save(
            {
                "min": scaler.data_min_.tolist(),
                "max": scaler.data_max_.tolist(),
                "train_excel_rows": raw.loc[train, "excel_row"].tolist(),
                "test_outside_any_count": int(((z[~train] < 0) | (z[~train] > 1)).any(dim=1).sum()),
            },
            HERE / "outputs" / "ae_fit.json",
        )
        for seed in CONFIG["seeds"]:
            torch.manual_seed(seed)
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 3),
                torch.nn.RReLU(),
                torch.nn.Linear(3, 2),
                torch.nn.RReLU(),
                torch.nn.Linear(2, 3),
                torch.nn.RReLU(),
                torch.nn.Linear(3, 4),
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            loader = torch.utils.data.DataLoader(
                train_tensor, batch_size=64, shuffle=True, num_workers=0
            )
            for epoch in range(50):
                model.train()
                loss_sum = 0.0
                for batch in loader:
                    optimizer.zero_grad()
                    loss = torch.mean((model(batch) - batch) ** 2)
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(batch)
                learning.append(
                    dict(seed=seed, epoch=epoch + 1, training_loss=loss_sum / len(train_tensor))
                )
            model.eval()
            with torch.no_grad():
                reconstruction = model(z)
                score = ((reconstruction - z) ** 2).mean(dim=1).numpy().astype(float)
            threshold = score[train].mean() + 8 * score[train].std(ddof=0)
            name = f"ae_{seed}"
            result[f"{name}_score"] = score
            result[f"{name}_flag"] = score >= threshold
            model_rows.append(
                dict(
                    model=name,
                    threshold=threshold,
                    train_flags=int((score[train] >= threshold).sum()),
                    test_flags=int((score[~train] >= threshold).sum()),
                    train_score_mean=score[train].mean(),
                    train_score_std=score[train].std(),
                )
            )
            torch.save(model.state_dict(), MODELS / f"{name}.pt")
            np.savez_compressed(
                MODELS / f"{name}_reconstruction.npz",
                input=z.numpy(),
                output=reconstruction.numpy(),
            )
            print(f"AE seed {seed} finished", flush=True)
        json_save(
            {"status": "complete", "torch": torch.__version__, "device": "cpu"},
            HERE / "outputs" / "ae_status.json",
        )
    save(result, "row_scores")
    save(model_rows, "row_models")
    save(learning, "ae_learning")
    stability = []
    for a, b in itertools.combinations([c[:-6] for c in result if c.endswith("_score")], 2):
        u, v = result[f"{a}_flag"].to_numpy(bool), result[f"{b}_flag"].to_numpy(bool)
        stability.append(
            dict(
                a=a,
                b=b,
                spearman=spearmanr(result[f"{a}_score"], result[f"{b}_score"]).statistic,
                jaccard=np.sum(u & v) / np.sum(u | v) if np.any(u | v) else np.nan,
            )
        )
    save(stability, "row_sensitivity")


def aggregate_scores(raw, overlap_flags):
    groups = read("candidate_map")
    rows = read("row_scores")
    starts = groups.excel_start.to_numpy(int) - 2
    locations = starts[:, None] + np.arange(4)
    for col in rows:
        if col.endswith("_score"):
            z = rows[col].to_numpy()[locations]
            groups[f"{col}_mean"] = z.mean(axis=1)
            groups[f"{col}_max"] = z.max(axis=1)
        elif col.endswith("_flag"):
            groups[f"{col}_any"] = rows[col].to_numpy(bool)[locations].any(axis=1)
    for width in (1, 4, 16, 100):
        groups[f"overlap_{width}_any"] = (
            overlap_flags[f"overlap_{width}"].to_numpy(bool)[locations].any(axis=1)
        )
    save(groups, "candidate_row_scores")
    audit = []
    for width in (1, 4, 16, 100):
        keep = ~rows.train & ~overlap_flags[f"overlap_{width}"]
        entry = dict(unit="row", width=width, condition="all", r=-1, remaining=int(keep.sum()))
        for col in rows:
            if col.endswith("_flag"):
                entry[col] = int(rows.loc[keep, col].sum())
        audit.append(entry)
        for (condition, r), part in groups.groupby(["condition", "r"]):
            keep_g = ~part.train & ~part[f"overlap_{width}_any"]
            entry = dict(
                unit="candidate", width=width, condition=condition, r=r, remaining=int(keep_g.sum())
            )
            for col in part:
                if col.endswith("_flag_any"):
                    entry[col] = int(part.loc[keep_g, col].sum())
            audit.append(entry)
    save(audit, "purged_scores")


def main():
    start = time.time()
    MODELS.mkdir(parents=True, exist_ok=True)
    raw, segments = load()
    with threadpool_limits(limits=4):
        flags = overlap(raw, segments)
        candidate_models(raw)
        row_models(raw)
        aggregate_scores(raw, flags)
    json_save(
        {"completed": True, "seconds": time.time() - start}, HERE / "outputs" / "model_run.json"
    )
    print("MODELS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
