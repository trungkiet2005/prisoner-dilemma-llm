"""Mine the two thirds of LLM play that no canonical rule explains.

F27 -- how far a widening hypothesis class gets, against a permutation null
F28 -- what the residual behaviour actually looks like: memory depth,
       stochastic memory-one coordinates, and where the rule breaks
F29 -- regime switching: when models change strategy mid-game
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from pdlib.residual import (CANONICAL_4, CORE_NAMES, FULL_NAMES, NAMED, STATES,
                            bic_memory_depth, conditional_profile, encode,
                            hypothesis_coverage, shuffle_null)
from pdlib.rulebase import RULES
from pdlib.style import (CMAP_DIV, CMAP_SEQ, C_COOP, C_DEFECT, DATADIR, INK,
                         INK2, LANG_ORDER, MODEL, MODEL_ORDER, MUTED,
                         STRATEGY, STRATEGY_ORDER, SURFACE, TABDIR, panel_tag,
                         savefig, use_paper_style)

use_paper_style()

FAM = {"frontier": 10, "small": 30}
LADDER = [("canonical4", "4 canonical rules\nAllC · AllD · TFT · WSLS"),
          ("memory1_32", "all 32 deterministic\nmemory-one rules"),
          ("two_regime", "…plus one mid-game\nregime switch")]


# --------------------------------------------------------------------------
def build(rounds, arche):
    """Assemble the residual set: trajectories no canonical rule explains."""
    none = arche.loc[arche.assignment == "approx", ["game_uid", "agent"]]
    res = {}
    for fam, ml in FAM.items():
        sub = rounds[rounds.family == fam].merge(none, on=["game_uid", "agent"])
        sub = sub.sort_values(["game_uid", "agent", "round"])
        g = sub.groupby(["game_uid", "agent"], sort=False)
        own = g.action.apply(list)
        opp = g.opp_action.apply(list)
        X, lens = encode(list(own), list(opp), ml)
        cov = hypothesis_coverage(X, ml)
        null = shuffle_null(list(own), list(opp), ml)
        keys = pd.DataFrame(list(own.index), columns=["game_uid", "agent"])
        res[fam] = dict(keys=keys, X=X, lens=lens, cov=cov, null=null,
                        rounds=sub, own=own, opp=opp)
    return res


# --------------------------------------------------------------------------
def fig_ladder(res, arche):
    fig = plt.figure(figsize=(11.2, 6.6))
    gs = fig.add_gridspec(2, 3, hspace=0.66, wspace=0.42)

    rows = []
    for fam in FAM:
        d = res[fam]
        for key, _ in LADDER:
            rows.append({"family": fam, "class": key, "n": len(d["X"]),
                         "observed": float(d["cov"][key].mean()),
                         "null": float(d["null"][key].mean())})
    lad = pd.DataFrame(rows)
    lad["enrichment"] = lad.observed / lad.null.clip(lower=1e-6)
    lad.to_csv(TABDIR / "T20_hypothesis_ladder.csv", index=False)

    # (a) coverage ladder with the permutation null --------------------------
    ax = fig.add_subplot(gs[0, :2])
    x = np.arange(len(LADDER))
    w = 0.2
    for k, (fam, col) in enumerate((("frontier", C_COOP), ("small", C_DEFECT))):
        s = lad[lad.family == fam].set_index("class").reindex([c for c, _ in LADDER])
        ax.bar(x + (k - 0.5) * 2 * w - w / 2, s.observed, width=w,
               color=col, edgecolor=SURFACE, linewidth=1.0,
               label=f"{fam}: observed")
        ax.bar(x + (k - 0.5) * 2 * w + w / 2, s.null, width=w, color=col,
               alpha=0.35, edgecolor=SURFACE, linewidth=1.0, hatch="///",
               label=f"{fam}: shuffled null")
        for xi, (o, nu) in enumerate(zip(s.observed, s.null)):
            ax.text(xi + (k - 0.5) * 2 * w - w / 2, o + 0.015, f"{o:.2f}",
                    ha="center", fontsize=6.4, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in LADDER], fontsize=7)
    ax.set_ylabel("share of the residual explained")
    ax.set_ylim(0, 0.78)
    ax.legend(ncol=2, fontsize=6.5, loc="upper left")
    ax.set_title("Widening the rule vocabulary does not rescue the residual")
    panel_tag(ax, "a", dx=-0.085)

    # (b) what is left --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    left = []
    for fam in FAM:
        d = res[fam]
        n = len(d["X"])
        c4 = d["cov"]["canonical4"]
        m1 = d["cov"]["memory1_32"] & ~c4
        tr = d["cov"]["two_regime"] & ~d["cov"]["memory1_32"]
        rest = ~(d["cov"]["memory1_32"] | d["cov"]["two_regime"])
        left.append({"family": fam, "non-canonical\nmemory-one": m1.mean(),
                     "regime switch": tr.mean(), "still unexplained": rest.mean()})
    lf = pd.DataFrame(left).set_index("family")
    bottom = np.zeros(len(lf))
    cols = ["#4a3aa7", "#eda100", MUTED]
    for c, col in zip(lf.columns, cols):
        v = lf[c].to_numpy()
        ax.bar(np.arange(len(lf)), v, bottom=bottom, color=col,
               edgecolor=SURFACE, linewidth=1.2, width=0.6, label=c)
        for xi, (b, q) in enumerate(zip(bottom, v)):
            if q > 0.08:
                ax.text(xi, b + q / 2, f"{q:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="semibold")
        bottom = bottom + v
    ax.set_xticks(range(len(lf)))
    ax.set_xticklabels(lf.index)
    ax.set_ylim(0, 1)
    ax.grid(False)
    ax.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.set_title("Composition of the residual")
    panel_tag(ax, "b", dx=-0.30)

    # (c) which non-canonical rules do appear ---------------------------------
    ax = fig.add_subplot(gs[1, :2])
    tally = {}
    for fam in FAM:
        d = res[fam]
        per = d["cov"]["per_rule"]
        m = d["cov"]["memory1_32"] & ~d["cov"]["canonical4"]
        share = per[m].mean(axis=0) * m.mean()
        tally[fam] = pd.Series(share, index=FULL_NAMES)
    tal = pd.DataFrame(tally)
    tal["total"] = tal.sum(axis=1)
    top = tal.sort_values("total", ascending=False).head(10)
    y = np.arange(len(top))[::-1]
    ax.barh(y - 0.18, top.frontier, height=0.34, color=C_COOP,
            edgecolor=SURFACE, linewidth=1.0, label="frontier")
    ax.barh(y + 0.18, top["small"], height=0.34, color=C_DEFECT,
            edgecolor=SURFACE, linewidth=1.0, label="open-weight")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k}   {NAMED.get(k, '')}" for k in top.index],
                       fontsize=7, family="monospace")
    ax.set_xlabel("share of the residual  (opening · action after R, S, T, P)")
    ax.legend(fontsize=6.8, loc="lower right")
    ax.set_title("The memory-one rules that do fit are mostly D-openers")
    panel_tag(ax, "c", dx=-0.30)

    # (d) where the nearest canonical rule breaks -----------------------------
    ax = fig.add_subplot(gs[1, 2])
    rows = []
    for fam in FAM:
        r = res[fam]["rounds"]
        r = r.merge(arche[["game_uid", "agent", "lstm_archetype"]],
                    on=["game_uid", "agent"], how="left")
        r = r[r.prev_letter != "E"]
        pred = [RULES[a][s] for a, s in zip(r.lstm_archetype, r.prev_letter)]
        r = r.assign(violates=(np.array(pred) != r.action.to_numpy()))
        for mdl, s in r.groupby("model"):
            for st in STATES:
                q = s[s.prev_letter == st]
                if len(q) > 30:
                    rows.append({"model": mdl, "state": st,
                                 "violation_rate": float(q.violates.mean())})
    viol = pd.DataFrame(rows)
    viol.to_csv(TABDIR / "T21_deviation_fingerprint.csv", index=False)
    piv = viol.pivot(index="model", columns="state", values="violation_rate") \
              .reindex(index=MODEL_ORDER, columns=STATES)
    im = ax.imshow(piv.to_numpy(), cmap=CMAP_SEQ, aspect="auto", vmin=0, vmax=1)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8,
                        color="white" if v > 0.55 else INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["after R", "after S", "after T", "after P"], fontsize=6.8)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([m.split("-")[0] for m in MODEL_ORDER])
    ax.grid(False)
    ax.set_title("Where the nearest rule breaks")
    panel_tag(ax, "d", dx=-0.36)

    fig.suptitle("Anatomy of the play that no canonical strategy explains",
                 x=0.02, ha="left", fontweight="bold", color=INK)
    savefig(fig, "F27_residual_anatomy")
    return lad


# --------------------------------------------------------------------------
def fig_structure(res, rounds, arche):
    fig = plt.figure(figsize=(11.2, 6.8))
    gs = fig.add_gridspec(2, 3, hspace=0.68, wspace=0.42)

    none = arche.loc[arche.assignment == "approx", ["game_uid", "agent"]]
    r = rounds.merge(none, on=["game_uid", "agent"]).sort_values(
        ["game_uid", "agent", "round"])
    r["prev2"] = r.groupby(["game_uid", "agent"], sort=False)["prev_letter"].shift(1)
    late = r[r["round"] >= 3]

    # (a) memory depth --------------------------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    rows = []
    for mdl in MODEL_ORDER:
        b = bic_memory_depth(late[late.model == mdl])
        if b:
            rows.append({"model": mdl, **b})
    bic = pd.DataFrame(rows).set_index("model")
    for c in ["memoryless", "memory1", "memory2", "memory1_time"]:
        bic[f"d_{c}"] = (bic[f"BIC_{c}"] - bic.base) / bic.n
    bic.to_csv(TABDIR / "T22_memory_depth.csv")

    x = np.arange(len(bic))
    w = 0.26
    for k, (c, col, name) in enumerate((
            ("memory1", "#4a3aa7", "memory-one"),
            ("memory1_time", "#eda100", "memory-one + game phase"),
            ("memory2", "#1baf7a", "memory-two"))):
        ax.bar(x + (k - 1) * w, bic[f"d_{c}"], width=w * 0.9, color=col,
               edgecolor=SURFACE, linewidth=0.9, label=name)
    ax.axhline(0, color=INK, lw=1.0)
    ax.text(len(bic) - 0.4, 0.012, "memoryless baseline", fontsize=6.5,
            color=MUTED, ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("-")[0] for m in bic.index], rotation=15,
                       ha="right")
    ax.set_ylabel("ΔBIC per round\n(more negative = better)")
    ax.legend(ncol=3, fontsize=6.5, loc="lower left")
    ax.set_title("History-dependent, and deeper than memory-one")
    ax.margins(y=0.14)
    panel_tag(ax, "a", dx=-0.11)

    # (b) stochastic memory-one coordinates ----------------------------------
    ax = fig.add_subplot(gs[0, 2])
    prof = (late.groupby(["model", "prev_letter"]).coop.mean().unstack()
            .reindex(index=MODEL_ORDER, columns=STATES))
    for mdl in MODEL_ORDER:
        ax.plot(range(4), prof.loc[mdl], "-o", color=MODEL[mdl], ms=4,
                mec=SURFACE, mew=1.0, label=mdl.split("-")[0])
    for name, vec, col in (("TFT", [1, 0, 1, 0], STRATEGY["TFT"]),
                           ("WSLS", [1, 0, 0, 1], STRATEGY["WSLS"])):
        ax.plot(range(4), vec, ":", color=col, lw=1.6, alpha=0.8)
        ax.annotate(name, (3, vec[3]), textcoords="offset points",
                    xytext=(6, 0), fontsize=6.5, color=col, va="center")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["R", "S", "T", "P"])
    ax.set_xlim(-0.2, 3.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(cooperate | previous outcome)")
    ax.set_xlabel("previous joint outcome")
    ax.legend(ncol=3, fontsize=5.4, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    ax.set_title("Residual is a blurred rule")
    panel_tag(ax, "b", dx=-0.34)

    # (c) is there any cluster structure at all? ------------------------------
    # Per-trajectory conditional profiles are only meaningful where every
    # state was actually visited a few times, which needs the 30-round games;
    # over ten rounds four states share ten observations and the estimates are
    # mostly prior.
    feats = (r[r.family == "small"]
             .groupby(["game_uid", "agent"], sort=False)
             .apply(conditional_profile, include_groups=False))
    enough = (feats[[f"n_{st}" for st in STATES]] >= 2).all(axis=1)
    meta = arche.set_index(["game_uid", "agent"])
    feats = feats.join(meta[["model", "language", "scale_nominal", "efficiency"]])
    sub = feats[enough]
    print(f"  30-round residual {len(feats):,}; all four states observed "
          f"{enough.sum():,} ({enough.mean():.1%})")

    cols = ["pC_R", "pC_S", "pC_T", "pC_P", "self_persist", "opp_match",
            "coop_rate"]
    Zraw = sub[cols].to_numpy()
    Z = StandardScaler().fit_transform(Zraw)

    sil = []
    rng = np.random.default_rng(0)
    samp = rng.choice(len(Z), min(4000, len(Z)), replace=False)
    for k in range(2, 8):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
        sil.append({"k": k, "silhouette": silhouette_score(Z[samp], km.labels_[samp])})
        if k == 5:
            sub = sub.assign(cluster=km.labels_)
    pd.DataFrame(sil).to_csv(TABDIR / "T23_residual_cluster_search.csv", index=False)
    sub.groupby("cluster")[cols].mean().round(3).to_csv(
        TABDIR / "T23b_residual_cluster_centres.csv")

    ax = fig.add_subplot(gs[1, :2])
    P = sub[["pC_R", "pC_S", "pC_T", "pC_P"]].to_numpy()
    mu = P.mean(0)
    U, S, Vt = np.linalg.svd(P - mu, full_matrices=False)
    Y = (P - mu) @ Vt[:2].T
    var = (S ** 2 / (S ** 2).sum())[:2]
    hb = ax.hexbin(Y[:, 0], Y[:, 1], gridsize=34, cmap=CMAP_SEQ, mincnt=1,
                   linewidths=0)
    for name, vec in (("AllC", [1, 1, 1, 1]), ("AllD", [0, 0, 0, 0]),
                      ("TFT", [1, 0, 1, 0]), ("WSLS", [1, 0, 0, 1]),
                      ("GRIM", [1, 0, 0, 0])):
        p = (np.array(vec, float) - mu) @ Vt[:2].T
        ax.plot(*p, "*", ms=15, color=STRATEGY.get(name, INK), mec=SURFACE,
                mew=1.4, zorder=6)
        ax.annotate(name, p, textcoords="offset points", xytext=(8, 4),
                    fontsize=7, fontweight="bold",
                    color=STRATEGY.get(name, INK2), zorder=6)
    for mdl in MODEL_ORDER:
        m = (sub.model == mdl).to_numpy()
        if m.sum() < 20:
            continue
        c = Y[m].mean(0)
        ax.plot(*c, "o", ms=10, color=MODEL[mdl], mec=SURFACE, mew=1.6, zorder=7)
        ax.annotate(mdl.split("-")[0], c, textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=6.8,
                    color=MODEL[mdl], fontweight="semibold", zorder=7)
    ax.set_xlabel(f"PC1 of the memory-one profile  ({var[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2  ({var[1]:.0%})")
    ax.grid(True, axis="both")
    ax.set_title("The residual fills the space between the canonical rules\n"
                 "30-round games only, where per-trajectory profiles are estimable",
                 fontsize=8.6)
    cb = fig.colorbar(hb, ax=ax, fraction=0.024, pad=0.015)
    cb.outline.set_visible(False)
    cb.set_label("trajectories", fontsize=7)
    panel_tag(ax, "c", dx=-0.085, dy=1.14)

    # (d) no natural number of clusters --------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    sl = pd.DataFrame(sil)
    ax.plot(sl.k, sl.silhouette, "-o", color=C_DEFECT, ms=6, mec=SURFACE, mew=1.2)
    ax.set_xlabel("number of clusters k")
    ax.set_ylabel("silhouette")
    ax.set_ylim(0, max(0.35, sl.silhouette.max() * 1.25))
    ax.text(0.5, 0.94, "no elbow: the residual is a continuum,\n"
            "not a set of discrete archetypes", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.8, color=INK2)
    ax.set_title("No natural k")
    panel_tag(ax, "d", dx=-0.32)

    fig.suptitle("What the unexplained play is made of", x=0.02, ha="left",
                 fontweight="bold", color=INK)
    savefig(fig, "F28_residual_structure")
    return bic, sub


# --------------------------------------------------------------------------
def fig_regimes(res, arche):
    fig = plt.figure(figsize=(11.2, 4.4))
    gs = fig.add_gridspec(1, 3, wspace=0.42)

    # (a) when the switch happens --------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for fam, col in (("frontier", C_COOP), ("small", C_DEFECT)):
        d = res[fam]
        cut = d["cov"]["switch_round"]
        m = d["cov"]["two_regime"] & ~d["cov"]["memory1_32"]
        v = cut[m] / FAM[fam]
        ax.hist(v, bins=np.linspace(0, 1, 16), density=True, histtype="step",
                lw=2.0, color=col, label=fam)
    ax.set_xlabel("switch point, as a fraction of the game")
    ax.set_ylabel("density")
    ax.legend(fontsize=6.8)
    ax.set_title("Switches cluster early")
    panel_tag(ax, "a", dx=-0.26)

    # (b) dominant transitions ------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    trans = _transitions(res)
    trans.to_csv(TABDIR / "T24_regime_transitions.csv", index=False)
    top = trans.groupby(["from", "to"]).share.sum().sort_values(
        ascending=False).head(6)
    y = np.arange(len(top))[::-1]
    ax.barh(y, top.to_numpy(), color=[STRATEGY.get(a, MUTED) for a, _ in top.index],
            edgecolor=SURFACE, linewidth=1.0, height=0.64)
    for yi, v in zip(y, top.to_numpy()):
        ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a} → {b}" for a, b in top.index], fontsize=7.5)
    ax.set_xlim(0, float(top.max()) * 1.3)
    ax.set_xlabel("share of the residual")
    ax.set_title("Dominant transitions")
    panel_tag(ax, "b", dx=-0.34)

    # (c) per-model regime-switch share ---------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    rows = []
    for fam in FAM:
        d = res[fam]
        k = d["keys"].copy()
        k["switch"] = d["cov"]["two_regime"] & ~d["cov"]["memory1_32"]
        k = k.merge(arche[["game_uid", "agent", "model"]],
                    on=["game_uid", "agent"])
        rows.append(k)
    k = pd.concat(rows)
    sh = k.groupby("model").switch.mean().reindex(MODEL_ORDER)
    y = np.arange(len(sh))[::-1]
    ax.barh(y, sh.to_numpy(), color=[MODEL[m] for m in sh.index], height=0.62,
            edgecolor=SURFACE, linewidth=1.0)
    for yi, v in zip(y, sh.to_numpy()):
        ax.text(v + 0.008, yi, f"{v:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([m.split("-")[0] for m in sh.index])
    ax.set_xlim(0, float(sh.max()) * 1.3)
    ax.set_xlabel("share of the model's residual")
    ax.set_title("Who switches mid-game")
    panel_tag(ax, "c", dx=-0.34)

    fig.suptitle("Mid-game regime switching in the residual", x=0.02, y=0.99,
                 ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    savefig(fig, "F29_regime_switching")


def _transitions(res):
    from pdlib.residual import CORE_OK, FULL_OK
    NM = {"CCCCC": "AllC", "DDDDD": "AllD", "CCDCD": "TFT", "CCDDC": "WSLS",
          "CCDDD": "GRIM"}
    rows = []
    for fam, ml in FAM.items():
        d = res[fam]
        X = d["X"]
        pre = np.cumprod(FULL_OK[:, X], axis=2)
        suf = np.cumprod(CORE_OK[:, X][:, :, ::-1], axis=2)[:, :, ::-1]
        ki = [(FULL_NAMES.index(k), v) for k, v in NM.items()]
        mi = [(CORE_NAMES.index(k[1:]), v) for k, v in NM.items()]
        mask = d["cov"]["two_regime"] & ~d["cov"]["memory1_32"]
        tally = {}
        for i in np.where(mask)[0]:
            done = False
            for t in range(int(d["lens"][i]) - 1):
                for ka, na in ki:
                    if not pre[ka, i, t]:
                        continue
                    for mb, nb in mi:
                        if na != nb and suf[mb, i, t + 1]:
                            tally[(na, nb)] = tally.get((na, nb), 0) + 1
                            done = True
                            break
                    if done:
                        break
                if done:
                    break
        n = len(X)
        for (a, b), c in tally.items():
            rows.append({"family": fam, "from": a, "to": b, "n": c,
                         "share": c / n})
    return pd.DataFrame(rows)


def main():
    rounds = pd.read_parquet(DATADIR / "rounds.parquet")
    arche = pd.read_parquet(DATADIR / "llm_archetypes.parquet")
    res = build(rounds, arche)
    lad = fig_ladder(res, arche)
    bic, feats = fig_structure(res, rounds, arche)
    fig_regimes(res, arche)
    print()
    print(lad.round(3).to_string(index=False))
    print()
    print(bic[[c for c in bic.columns if c.startswith("d_")]].round(3).to_string())


if __name__ == "__main__":
    main()
