"""Statistics for the strategy-first manuscript (frontier arm, ten rounds).

The FR suite already answers "how much of this play is a canonical rule?".
This script supplies the four things the strategy-first argument needs and the
FR suite does not compute:

  A.  **What the read-out is worth.**  A risk-coverage sweep on the synthetic
      corpus, where the generating strategy is known, so the 0.90 abstention
      floor is reported as a chosen operating point on a curve rather than as
      a constant.  Ten-round posteriors are not bimodal the way thirty-round
      ones are (`Analysis/README.md`), so the floor here is a convention and
      has to be defended by its risk-coverage behaviour instead.

  B.  **Provenance per label, not per game.**  T_FR27 gives the share of games
      reached deductively; what the argument needs is the share of each
      *label* that was reached deductively.  The two differ sharply, and the
      WSLS column is where they differ most.

  C.  **The strategy mix as the dependent variable.**  Payoff scale, prompt
      language and assigned persona are re-expressed as manipulations of the
      distribution over strategies, with a total-variation distance and a
      permutation null that shuffles whole dyads.

  D.  **Mining the abstention set.**  The extended library of named
      non-canonical rules and two-phase clock rules, alternation statistics,
      a cluster search, and the within-game coherence check that decides
      whether an unnameable trajectory is a hidden strategy or drift.

Writes `tables/T_S*.csv`; `34_fig_paper_strategy.py` reads them and draws no
number of its own.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from pdlib import residual, unclassified as unc
from pdlib.natstyle import DATADIR, FRONTIER, TABDIR
from pdlib.rulebase import consistent_mask
from pdlib.seqcode import STRATEGIES

MAX_LEN = 10
FLOOR = unc.THRESHOLD                      # 0.90
LABELS5 = ["AllC", "TFT", "WSLS", "AllD", "Ambiguous"]
BUCKETS = ["exact", "ambiguous", "confident", "unclassified"]
N_BOOT = 2000
N_PERM = 2000
SEED = 0


# ==========================================================================
# helpers
# ==========================================================================
def _boot_share(labels: np.ndarray, dyads: np.ndarray, categories,
                n_boot: int = N_BOOT, seed: int = SEED):
    """Category shares with a dyad-clustered bootstrap CI.

    Both agents of a dyad appear as rows and share a history, so resampling
    rows would understate every interval; the resample is over dyads.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(dyads, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(order))
    cat_idx = {c: i for i, c in enumerate(categories)}
    codes = np.array([cat_idx.get(v, -1) for v in labels])

    obs = np.array([(codes == i).mean() for i in range(len(categories))])
    draws = np.empty((n_boot, len(categories)))
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([order[starts[p]:ends[p]] for p in pick])
        c = codes[rows]
        draws[b] = [(c == i).mean() for i in range(len(categories))]
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return obs, lo, hi


def _mix(labels: np.ndarray, categories) -> np.ndarray:
    idx = {c: i for i, c in enumerate(categories)}
    out = np.zeros(len(categories))
    for v in labels:
        j = idx.get(v, -1)
        if j >= 0:
            out[j] += 1
    return out / max(out.sum(), 1)


def _max_tv(mixes) -> float:
    """Largest total-variation distance between any two level distributions."""
    m = np.asarray(mixes)
    best = 0.0
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            best = max(best, 0.5 * np.abs(m[i] - m[j]).sum())
    return best


def mix_shift_test(df: pd.DataFrame, factor: str, categories=LABELS5,
                   n_perm: int = N_PERM, seed: int = SEED):
    """Does the strategy mix move across the levels of `factor`?

    The statistic is the largest total-variation distance between two levels.
    The null shuffles the factor label across *dyads*, so a dyad keeps both of
    its agent-rows together and the two rows never land in different levels.
    """
    lab = df.archetype.to_numpy()
    lev = df[factor].to_numpy()
    dyad = df.game_uid.to_numpy()

    levels = list(pd.unique(lev))
    obs = _max_tv([_mix(lab[lev == L], categories) for L in levels])

    key = pd.DataFrame({"game_uid": dyad, "lev": lev}).drop_duplicates("game_uid")
    rng = np.random.default_rng(seed)
    uid_index = pd.Series(np.arange(len(key)), index=key.game_uid.to_numpy())
    row_of_dyad = uid_index.reindex(dyad).to_numpy()
    lev_by_dyad = key.lev.to_numpy()

    null = np.empty(n_perm)
    for b in range(n_perm):
        shuffled = rng.permutation(lev_by_dyad)[row_of_dyad]
        null[b] = _max_tv([_mix(lab[shuffled == L], categories) for L in levels])
    return obs, float((null >= obs).mean()), float(np.percentile(null, 95))


def sequences(rounds: pd.DataFrame):
    """(game_uid, agent) keyed own/opponent action lists, in round order."""
    r = rounds.sort_values(["game_uid", "agent", "round"])
    g = r.groupby(["game_uid", "agent"], sort=False)
    own = g.action.apply(list)
    opp = g.opp_action.apply(list)
    keys = list(own.index)
    return keys, list(own), list(opp)


# ==========================================================================
# A.  what the read-out is worth
# ==========================================================================
def readout_validation():
    """Risk-coverage on synthetic play, where the generating rule is known."""
    test = np.load(DATADIR / "clf_test_h10.npz", allow_pickle=True)
    ood = np.load(DATADIR / "clf_ood_h10_Noise01.npz", allow_pickle=True)

    rows = []
    for name, d in (("test (0-5% noise)", test), ("unseen 10% noise", ood)):
        proba, y = d["proba"], d["y"]
        conf = proba.max(1)
        correct = proba.argmax(1) == y
        for thr in [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]:
            keep = conf >= thr
            rows.append({
                "corpus": name, "threshold": thr,
                "coverage": float(keep.mean()),
                "accuracy_kept": float(correct[keep].mean()) if keep.any() else np.nan,
                "accuracy_all": float(correct.mean()),
                "error_kept": float(1 - correct[keep].mean()) if keep.any() else np.nan,
            })
    out = pd.DataFrame(rows)
    out.to_csv(TABDIR / "T_S01_risk_coverage.csv", index=False)

    # exact-rule and set-valued behaviour of the deductive stage, same corpus
    X, y = test["X"], test["y"]
    m = consistent_mask(X)
    n_fit = m.sum(axis=1)
    truth_in_set = m[np.arange(len(y)), y]
    setrows = pd.DataFrame({
        "n_rule_fits": ["0 (no rule)", "1 (unique)", ">1 (a set)"],
        "share": [float((n_fit == 0).mean()), float((n_fit == 1).mean()),
                  float((n_fit > 1).mean())],
        "truth_recovered": [np.nan,
                            float(truth_in_set[n_fit == 1].mean()),
                            float(truth_in_set[n_fit > 1].mean())],
    })
    setrows.to_csv(TABDIR / "T_S02_deductive_stage.csv", index=False)
    print("\nA. read-out validation")
    print(out[out.threshold.isin([0.0, 0.9])].to_string(index=False))
    print(setrows.to_string(index=False))
    return out


# ==========================================================================
# B.  provenance per label
# ==========================================================================
def label_provenance(arche: pd.DataFrame):
    """For each strategy name, how it was arrived at.

    A label reached by exact matching is a deduction; a label reached by the
    network is a nearest-neighbour statement about play that matches no rule.
    Pooling the two is what lets a vocabulary look better supported than it is.
    """
    a = arche.copy()
    rows = []
    for lab in STRATEGIES:
        exact = a[(a.assignment == "exact") & (a.archetype == lab)]
        approx = a[(a.assignment == "approx") & (a.archetype == lab)]
        inset = a[(a.assignment == "ambiguous")
                  & a.rule_set.str.split("+").apply(lambda s: lab in s)]
        n_named = len(exact) + len(approx)
        rows.append({
            "label": lab,
            "n_assigned": n_named,
            "share_of_corpus": n_named / len(a),
            "n_exact": len(exact),
            "n_lstm": len(approx),
            "prov_exact": len(exact) / n_named if n_named else np.nan,
            "n_in_ambiguous_set": len(inset),
            "mean_dev_lstm": float(approx.min_deviations.mean()) if len(approx) else np.nan,
            "mean_conf_lstm": float(approx.confidence.mean()) if len(approx) else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABDIR / "T_S03_label_provenance.csv", index=False)

    per_model = []
    for mdl in FRONTIER:
        s = a[a.model == mdl]
        for lab in STRATEGIES:
            ex = int(((s.assignment == "exact") & (s.archetype == lab)).sum())
            ap = int(((s.assignment == "approx") & (s.archetype == lab)).sum())
            per_model.append({"model": mdl, "label": lab,
                              "exact": ex / len(s), "lstm": ap / len(s),
                              "total": (ex + ap) / len(s),
                              "prov_exact": ex / (ex + ap) if ex + ap else np.nan})
    pm = pd.DataFrame(per_model)
    pm.to_csv(TABDIR / "T_S04_label_provenance_by_model.csv", index=False)

    print("\nB. provenance per label (pooled)")
    print(out.round(4).to_string(index=False))
    return out, pm


def census(arche: pd.DataFrame):
    """Four-way read-out per model, and the same at three abstention floors."""
    a = unc.add_buckets(arche, FLOOR)
    tab = (a.groupby(["model", "bucket"]).size().unstack("bucket")
           .reindex(FRONTIER).reindex(columns=BUCKETS).fillna(0))
    tab = tab.div(tab.sum(axis=1), axis=0)
    tab.loc["all models"] = (a.bucket.value_counts(normalize=True)
                             .reindex(BUCKETS).fillna(0))
    tab.to_csv(TABDIR / "T_S05_census.csv")

    sens = []
    for f in (0.80, 0.90, 0.95):
        b = unc.add_buckets(arche, f)
        sens.append({"floor": f,
                     **b.bucket.value_counts(normalize=True).reindex(BUCKETS).to_dict()})
    pd.DataFrame(sens).to_csv(TABDIR / "T_S06_floor_sensitivity.csv", index=False)

    print("\nB. census")
    print(tab.round(3).to_string())
    return tab


# ==========================================================================
# C.  the mix as the dependent variable
# ==========================================================================
FACTORS = {"scale_nominal": "payoff scale",
           "language": "prompt language",
           "personality": "own persona",
           "dyad": "persona pairing"}


def mix_by_condition(arche: pd.DataFrame):
    long, tests = [], []
    for mdl in FRONTIER:
        s = arche[arche.model == mdl]
        for fac in FACTORS:
            for lev, d in s.groupby(fac):
                obs, lo, hi = _boot_share(d.archetype.to_numpy(),
                                          d.game_uid.to_numpy(), LABELS5)
                for k, lab in enumerate(LABELS5):
                    long.append({"model": mdl, "factor": fac, "level": lev,
                                 "archetype": lab, "share": obs[k],
                                 "lo": lo[k], "hi": hi[k], "n": len(d)})
            tv, p, null95 = mix_shift_test(s, fac)
            # "Ambiguous" is a statement about identifiability rather than
            # about behaviour, so the same test is repeated over the four
            # strategies alone; if the shift were an artefact of games moving
            # in and out of the ambiguous bin, this column would collapse.
            s4 = s[s.archetype != "Ambiguous"]
            tv4, p4, null4 = mix_shift_test(s4, fac, categories=STRATEGIES)
            tests.append({"model": mdl, "factor": fac, "max_tv": tv,
                          "p_perm": p, "null_95": null95,
                          "max_tv_named4": tv4, "p_perm_named4": p4,
                          "null_95_named4": null4,
                          "n_levels": s[fac].nunique()})
    pd.DataFrame(long).to_csv(TABDIR / "T_S07_mix_by_condition.csv", index=False)
    t = pd.DataFrame(tests)
    t.to_csv(TABDIR / "T_S08_mix_shift_tests.csv", index=False)

    print("\nC. does the strategy mix move?")
    print(t.round(4).to_string(index=False))
    return t


# ==========================================================================
# D.  mining the abstention set
# ==========================================================================
def library_mining(rounds: pd.DataFrame, arche: pd.DataFrame):
    """The extended vocabulary, run per bucket against a shuffled null."""
    lib = unc.library_table(rounds, seed=SEED)
    lib = lib.merge(arche[["game_uid", "agent", "model", "archetype",
                           "assignment", "confidence"]],
                    on=["game_uid", "agent"], how="left")
    lib = unc.add_buckets(lib.rename(columns={"assignment": "assignment"}), FLOOR)
    lib["exact_extended"] = lib.dev_extended == 0
    lib["exact_extended_null"] = lib.dev_extended_null == 0
    lib["exact_canonical"] = lib.dev_canonical == 0
    lib.to_parquet(DATADIR / "frontier_library.parquet", index=False)

    by_bucket = (lib.groupby("bucket")
                 .agg(n=("game_uid", "size"),
                      extended_exact=("exact_extended", "mean"),
                      extended_null=("exact_extended_null", "mean"),
                      noncanonical_nearest=("is_canonical",
                                            lambda s: float(1 - s.mean())),
                      mean_dev_extended=("dev_extended", "mean"),
                      mean_dev_canonical=("dev_canonical", "mean"))
                 .reindex(BUCKETS))
    by_bucket["excess"] = by_bucket.extended_exact - by_bucket.extended_null
    by_bucket.to_csv(TABDIR / "T_S09_library_by_bucket.csv")

    by_model = (lib[lib.bucket == "unclassified"].groupby("model")
                .agg(n=("game_uid", "size"),
                     extended_exact=("exact_extended", "mean"),
                     extended_null=("exact_extended_null", "mean"),
                     noncanonical_nearest=("is_canonical",
                                           lambda s: float(1 - s.mean())))
                .reindex(FRONTIER))
    by_model["excess"] = by_model.extended_exact - by_model.extended_null
    by_model.to_csv(TABDIR / "T_S10_library_by_model.csv")

    hit = lib[(lib.bucket == "unclassified") & lib.exact_extended]
    rules = (hit.best_family.value_counts(normalize=True).rename("share")
             .to_frame().reset_index().rename(columns={"index": "rule"}))
    rules["n"] = hit.best_family.value_counts().to_numpy()
    rules.to_csv(TABDIR / "T_S11_library_rules.csv", index=False)

    near = (lib[lib.bucket == "unclassified"].best_family
            .value_counts(normalize=True).rename("share").to_frame().reset_index()
            .rename(columns={"index": "rule"}))
    near.to_csv(TABDIR / "T_S12_nearest_rule_unclassified.csv", index=False)

    print("\nD1. extended library")
    print(by_bucket.round(4).to_string())
    print("\n  rules that fit the abstention set exactly (top 8)")
    print(rules.head(8).to_string(index=False))
    return lib


def alternation(rounds: pd.DataFrame, lib: pd.DataFrame):
    """Switching structure, and the alternating C/D pattern specifically.

    `alternation_excess` is the observed switch rate minus the 2p(1-p) a coin
    with the same cooperation rate would produce, so it is positive only for
    play that alternates more than its own base rate explains.
    """
    seq = unc.sequence_stats(rounds)
    seq = seq.merge(lib[["game_uid", "agent", "model", "bucket", "best_family"]],
                    on=["game_uid", "agent"], how="left")
    seq.to_parquet(DATADIR / "frontier_sequence_stats.parquet", index=False)

    rows = []
    for b in BUCKETS:
        d = seq[seq.bucket == b]
        if not len(d):
            continue
        rng = np.random.default_rng(SEED)
        uid = d.game_uid.to_numpy()
        uniq = np.unique(uid)
        vals = d.alternation_excess.to_numpy()
        draws = np.empty(N_BOOT)
        idx_of = {u: np.flatnonzero(uid == u) for u in uniq}
        for i in range(N_BOOT):
            pick = rng.choice(uniq, len(uniq), replace=True)
            draws[i] = vals[np.concatenate([idx_of[p] for p in pick])].mean()
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append({"bucket": b, "n": len(d),
                     "switch_rate": float(d.switch_rate.mean()),
                     "expected_iid": float(d.exp_switch_iid.mean()),
                     "alternation_excess": float(vals.mean()),
                     "lo": float(lo), "hi": float(hi),
                     "share_alternating": float((vals > 0.15).mean()),
                     "max_run": float(d.max_run.mean())})
    out = pd.DataFrame(rows)
    out.to_csv(TABDIR / "T_S13_alternation.csv", index=False)

    per_model = (seq[seq.bucket == "unclassified"].groupby("model")
                 .agg(n=("game_uid", "size"),
                      alternation_excess=("alternation_excess", "mean"),
                      share_alternating=("alternation_excess",
                                         lambda s: float((s > 0.15).mean())))
                 .reindex(FRONTIER))
    per_model.to_csv(TABDIR / "T_S14_alternation_by_model.csv")

    print("\nD2. alternation")
    print(out.round(4).to_string(index=False))
    return seq


def cluster_search(rounds: pd.DataFrame, arche: pd.DataFrame):
    """Is the abstention set a few hidden archetypes, or one continuum?"""
    prof = (rounds.groupby(["game_uid", "agent"])
            .apply(residual.conditional_profile, include_groups=False)
            .reset_index())
    prof = prof.merge(arche[["game_uid", "agent", "model", "assignment",
                             "confidence"]], on=["game_uid", "agent"])
    prof = unc.add_buckets(prof, FLOOR)
    prof.to_parquet(DATADIR / "frontier_profiles.parquet", index=False)

    cols = ["pC_R", "pC_S", "pC_T", "pC_P", "self_persist", "opp_match"]
    U = prof[prof.bucket == "unclassified"][cols].dropna().to_numpy()

    rng = np.random.default_rng(SEED)
    rows = []
    for k in range(2, 9):
        km = KMeans(k, n_init=10, random_state=SEED).fit(U)
        sil = silhouette_score(U, km.labels_, sample_size=min(4000, len(U)),
                               random_state=SEED)
        aris = []
        for _ in range(20):
            idx = rng.integers(0, len(U), len(U))
            lab = KMeans(k, n_init=5, random_state=SEED).fit(U[idx]).labels_
            aris.append(adjusted_rand_score(km.labels_[idx], lab))
        rows.append({"k": k, "silhouette": float(sil),
                     "ari_mean": float(np.mean(aris)),
                     "ari_sd": float(np.std(aris))})
    out = pd.DataFrame(rows)
    out.to_csv(TABDIR / "T_S15_cluster_search.csv", index=False)

    km = KMeans(3, n_init=10, random_state=SEED).fit(U)
    centres = pd.DataFrame(km.cluster_centers_, columns=cols)
    centres["share"] = np.bincount(km.labels_, minlength=3) / len(U)
    centres.to_csv(TABDIR / "T_S16_cluster_centres.csv", index=False)

    # distance from every bucket to the nearest canonical corner of the
    # reactive square, which is a model-free statement about how far the play
    # sits from the vocabulary
    react = unc.corner_distance(unc.reactive_coordinates(rounds))
    react = react.merge(prof[["game_uid", "agent", "bucket", "model"]],
                        on=["game_uid", "agent"], how="left")
    corner = (react.groupby("bucket").d_nearest_corner
              .agg(["size", "mean", "median",
                    ("q25", lambda s: s.quantile(0.25)),
                    ("q75", lambda s: s.quantile(0.75))]).reindex(BUCKETS))
    corner.to_csv(TABDIR / "T_S17_corner_distance.csv")

    print("\nD3. cluster search on the abstention set")
    print(out.round(3).to_string(index=False))
    print(corner.round(3).to_string())
    return prof, react


def reactive_identifiability(rounds: pd.DataFrame, arche: pd.DataFrame):
    """Can a reactive rule be proved at all over ten rounds?

    The obvious objection to §"a reactive rule is deduced in 2% of
    agent-games" is that ten rounds are too few to prove one.  Two controls
    answer it without leaving the data.

    The first is synthetic: the generating rule is known, so the share of
    TFT-generated and WSLS-generated trajectories that the deductive stage
    pins uniquely is a direct measurement of the instrument at this horizon.

    The second is on the LLM corpus itself.  A reactive rule is only
    distinguishable from a constant one if the opponent gave it something to
    react to: if the opponent never defects, TFT and AllC prescribe the same
    ten actions and no method can separate them.  We therefore report the
    share of agent-games whose opponent history *affords* the distinction -
    at least one C and at least one D in rounds 1..n-1 - and re-express the
    deduced-reactive rate inside that subset.
    """
    test = np.load(DATADIR / "clf_test_h10.npz", allow_pickle=True)
    X, y = test["X"], test["y"]
    m = consistent_mask(X)
    n_fit = m.sum(axis=1)
    rows = []
    for k, s in enumerate(STRATEGIES):
        sel = y == k
        uniq = sel & (n_fit == 1)
        rows.append({
            "generating_rule": s, "n": int(sel.sum()),
            "deduced_uniquely": float(uniq[sel].mean()),
            "deduced_correctly": float((m[uniq, k]).mean()) if uniq.any() else np.nan,
            "matched_by_none": float((n_fit[sel] == 0).mean()),
        })
    syn = pd.DataFrame(rows)
    syn.to_csv(TABDIR / "T_S21_synthetic_deduction_by_rule.csv", index=False)

    r = rounds.sort_values(["game_uid", "agent", "round"])
    g = r.groupby(["game_uid", "agent"], sort=False)
    afford = g.opp_action.apply(
        lambda s: ("C" in set(s.iloc[:-1])) and ("D" in set(s.iloc[:-1])))
    afford = afford.rename("affords_reaction").reset_index()

    a = arche.merge(afford, on=["game_uid", "agent"], how="left")
    a["deduced_reactive"] = (a.assignment == "exact") & a.archetype.isin(["TFT", "WSLS"])
    a["deduced_constant"] = (a.assignment == "exact") & a.archetype.isin(["AllC", "AllD"])

    out = []
    for mdl in FRONTIER + ["all models"]:
        s = a if mdl == "all models" else a[a.model == mdl]
        sub = s[s.affords_reaction]
        out.append({
            "model": mdl, "n": len(s),
            "affords_reaction": float(s.affords_reaction.mean()),
            "deduced_reactive_all": float(s.deduced_reactive.mean()),
            "deduced_reactive_given_afford": float(sub.deduced_reactive.mean()),
            "deduced_constant_given_afford": float(sub.deduced_constant.mean()),
        })
    llm = pd.DataFrame(out)
    llm.to_csv(TABDIR / "T_S22_reactive_affordance.csv", index=False)

    print("\nB2. can a reactive rule be proved at ten rounds?")
    print(syn.round(4).to_string(index=False))
    print(llm.round(4).to_string(index=False))
    return syn, llm


def motifs(rounds: pd.DataFrame, arche: pd.DataFrame, width: int = 4):
    """Unsupervised motif mining: which action patterns recur beyond chance?

    The library search asks "does a rule I already named fit?".  This asks the
    prior question with no vocabulary at all: count every length-`width`
    window of the focal player's own actions, and compare its frequency with a
    null that shuffles that player's own actions within its own game.  The null
    holds the cooperation rate fixed, so a lift above one is temporal
    structure and nothing else.  Alternation - the CDCD / DCDC windows - is the
    pattern a memory-one vocabulary cannot express at all, so it is the one to
    read first.
    """
    a = unc.add_buckets(arche, FLOOR)
    bucket = dict(zip(zip(a.game_uid, a.agent), a.bucket))

    r = rounds.sort_values(["game_uid", "agent", "round"])
    g = r.groupby(["game_uid", "agent"], sort=False)
    rng = np.random.default_rng(SEED)

    obs, null = {}, {}
    for key, acts in zip(g.groups.keys(), g.action.apply(list)):
        b = bucket.get(key)
        if b is None:
            continue
        seq = "".join(acts)
        sh = "".join(rng.permutation(list(acts)))
        for t in range(len(seq) - width + 1):
            obs[(b, seq[t:t + width])] = obs.get((b, seq[t:t + width]), 0) + 1
            null[(b, sh[t:t + width])] = null.get((b, sh[t:t + width]), 0) + 1

    rows = []
    for b in BUCKETS:
        tot_o = sum(v for (bb, _), v in obs.items() if bb == b)
        tot_n = sum(v for (bb, _), v in null.items() if bb == b)
        if not tot_o:
            continue
        for (bb, pat), n in obs.items():
            if bb != b:
                continue
            p_o = n / tot_o
            p_n = null.get((b, pat), 0) / max(tot_n, 1)
            rows.append({"bucket": b, "motif": pat, "n": n, "freq": p_o,
                         "freq_null": p_n,
                         "lift": p_o / p_n if p_n > 0 else np.nan})
    out = pd.DataFrame(rows).sort_values(["bucket", "lift"], ascending=[True, False])
    out.to_csv(TABDIR / "T_S19_motifs.csv", index=False)

    # memory-two contingency: the structure a memory-one rule cannot carry
    r2 = r.dropna(subset=["prev_action", "prev_opp_action"]).copy()
    r2["ctx"] = r2.prev_action + r2.prev_opp_action
    r2 = r2.merge(a[["game_uid", "agent", "bucket"]], on=["game_uid", "agent"])
    ctx = (r2.pivot_table(index="bucket", columns="ctx", values="coop",
                          aggfunc="mean").reindex(BUCKETS))
    ctx.to_csv(TABDIR / "T_S20_memory1_contingency.csv")

    alt = out[out.motif.isin(["CDCD", "DCDC"])]
    print("\nD5. motif mining (own-action windows, lift over a within-game shuffle)")
    print(alt.round(3).to_string(index=False))
    print("  top motifs in the abstention set")
    print(out[out.bucket == "unclassified"].head(6).round(3).to_string(index=False))
    return out


def coherence(rounds: pd.DataFrame, arche: pd.DataFrame):
    """Does the first half of a game predict the second?

    A trajectory generated by a fixed rule is coherent across the game by
    construction.  Ten rounds are too few for the reactive coordinates to be
    estimated twice, so the split is on the cooperation rate itself: five
    rounds against five.
    """
    r = rounds.copy()
    r["half"] = np.where(r["round"] <= 5, "first", "second")
    half = (r.groupby(["game_uid", "agent", "half"]).coop.mean()
            .unstack("half").reset_index().dropna())
    half = half.merge(arche[["game_uid", "agent", "model", "assignment",
                             "confidence"]], on=["game_uid", "agent"])
    half = unc.add_buckets(half, FLOOR)

    rows = []
    for b in BUCKETS:
        d = half[half.bucket == b]
        if len(d) < 20:
            continue
        rows.append({"bucket": b, "n": len(d),
                     "r_halves": float(np.corrcoef(d["first"], d["second"])[0, 1]),
                     "mean_abs_shift": float((d["second"] - d["first"]).abs().mean())})
    out = pd.DataFrame(rows)
    out.to_csv(TABDIR / "T_S18_within_game_coherence.csv", index=False)
    print("\nD4. within-game coherence")
    print(out.round(3).to_string(index=False))
    return out


# ==========================================================================
def main():
    arche = pd.read_parquet(DATADIR / "frontier_archetypes.parquet")
    rounds = pd.read_parquet(DATADIR / "frontier_rounds.parquet")
    print(f"{len(arche):,} agent-games   {len(rounds):,} agent-rounds   "
          f"abstention floor {FLOOR:.2f}")

    readout_validation()
    label_provenance(arche)
    census(arche)
    reactive_identifiability(rounds, arche)
    mix_by_condition(arche)
    lib = library_mining(rounds, arche)
    alternation(rounds, lib)
    cluster_search(rounds, arche)
    motifs(rounds, arche)
    coherence(rounds, arche)
    print("\nwrote tables/T_S01 .. T_S20")


if __name__ == "__main__":
    main()
