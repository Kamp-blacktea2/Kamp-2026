"""Check whether lag-3 component recurrence has a stable Raw4 phase."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
from common import OUT, write_csv

def main():
    rows = pd.read_csv(OUT / "physical_row_followup.csv").sort_values("excel_row")
    features = ["F", "I", "V", "t"]
    records = []
    examples = []
    for (date, seg), frame in rows.groupby(["date", "safe_segment"]):
        frame = frame.sort_values("excel_row")
        if len(frame) < 40:
            continue
        labels = frame.assigned_component.to_numpy()
        x = frame[features].to_numpy(float)
        relative = np.arange(len(frame))
        phase = relative % 3
        majority = pd.Series(labels).value_counts(normalize=True).iloc[0]
        purity = sum(pd.Series(labels[phase == j]).value_counts().max() for j in range(3)) / len(labels)
        mid = (len(labels) // 6) * 3
        left = [pd.Series(labels[:mid][phase[:mid] == j]).mode().iloc[0] for j in range(3)]
        right = [pd.Series(labels[mid:][phase[mid:] == j]).mode().iloc[0] for j in range(3)]
        phase_agree = sum(int(left[j] == right[j]) for j in range(3))
        for lag in (2, 3, 4, 6, 9, 12, 16):
            rec = {"date": date, "regime": frame.regime.iloc[0], "safe_segment": seg,
                   "rows": len(frame), "lag": lag, "component_match": float(np.mean(labels[lag:] == labels[:-lag])),
                   "exact_raw4_match": float(np.mean(np.all(x[lag:] == x[:-lag], axis=1))),
                   "phase3_adjusted_mi": float(adjusted_mutual_info_score(phase, labels)),
                   "phase3_purity": float(purity), "majority_component_share": float(majority),
                   "phase3_half_dominant_agree": phase_agree}
            for j, name in enumerate(features):
                rec[name + "_mean_abs_diff"] = float(np.mean(np.abs(x[lag:, j] - x[:-lag, j])))
            records.append(rec)
        examples.append({"date": date, "safe_segment": seg,
                         "start_excel_row": int(frame.excel_row.iloc[0]),
                         "first_24_components": "|".join(map(str, labels[:24])),
                         "first_24_I": "|".join(map(str, x[:24, 1]))})
    out = pd.DataFrame(records)
    local = []
    for (date, seg), frame in rows.groupby(["date", "safe_segment"]):
        frame = frame.sort_values("excel_row")
        if len(frame) < 96:
            continue
        for start in range(0, len(frame) - 95, 48):
            sub = frame.iloc[start:start+96]
            label = sub.assigned_component.to_numpy()
            phase3 = np.arange(96) % 3
            local.append({
                "date": date, "regime": sub.regime.iloc[0], "safe_segment": seg,
                "start_excel_row": int(sub.excel_row.iloc[0]),
                "end_excel_row": int(sub.excel_row.iloc[-1]),
                "lag2_rate": float(np.mean(label[2:] == label[:-2])),
                "lag3_rate": float(np.mean(label[3:] == label[:-3])),
                "lag4_rate": float(np.mean(label[4:] == label[:-4])),
                "phase3_adjusted_mi": float(adjusted_mutual_info_score(phase3, label)),
            })
    local = pd.DataFrame(local)
    local["lag3_peak"] = local.lag3_rate - (local.lag2_rate + local.lag4_rate) / 2
    write_csv(local, "physical_cycle_local_windows.csv")
    print("local 96-row windows", len(local),
          "median phase3 AMI", round(float(local.phase3_adjusted_mi.median()), 3),
          "max phase3 AMI", round(float(local.phase3_adjusted_mi.max()), 3),
          "peak3>0.1 and AMI>0.1", int(((local.lag3_peak > .1) & (local.phase3_adjusted_mi > .1)).sum()))
    write_csv(out, "physical_cycle_probe.csv")
    write_csv(pd.DataFrame(examples), "physical_cycle_examples.csv")
    print(out[out.lag.isin([2,3,4])][["date","lag","component_match","exact_raw4_match","I_mean_abs_diff","phase3_adjusted_mi","phase3_purity","majority_component_share","phase3_half_dominant_agree"]].round(3).to_string(index=False))
    print("overall")
    print(out.groupby("lag").apply(lambda g: pd.Series({
        "component_match": np.average(g.component_match, weights=g.rows-out.loc[g.index, "lag"]),
        "exact_raw4_match": np.average(g.exact_raw4_match, weights=g.rows-out.loc[g.index, "lag"]),
        "I_mean_abs_diff": np.average(g.I_mean_abs_diff, weights=g.rows-out.loc[g.index, "lag"]),
    }), include_groups=False).round(4).to_string())

if __name__ == "__main__":
    main()