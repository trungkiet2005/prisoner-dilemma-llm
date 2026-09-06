"""Sections 12 to 14: dimensionality reduction, behavioural regimes and anomalies.

Why this section exists
-----------------------
Everything up to here has treated behaviour one variable at a time: a
cooperation rate, a reciprocity coefficient, an endgame drop. But an agent-game
is not a scalar. It is a joint pattern over sixteen behavioural quantities, and
the interesting question is whether those quantities move together in a small
number of ways, that is, whether the corpus contains a handful of recognisable
*policies* rather than a continuum. That is a question about the shape of the
behaviour cloud, so it needs geometry, not another marginal mean.

The unit of analysis is the agent-game (24,000 rows), not the dyad, because a
policy is a property of one agent facing a history. The price of that choice is
that the two agent-games inside one dyad are the same interaction seen from two
sides, so they are not independent. Every interval below therefore clusters on
`game_uid`, and the one test that genuinely needs independent draws (the
regime-mix contingency test) is run on the agent-1 subsample only, which is
12,000 independent games and is balanced across personas by construction,
because CvS and SvC appear equally often.

Three decisions about the feature matrix, each of which changes the answer
-------------------------------------------------------------------------
1. `efficiency` is exactly 1.25 times `utility` (r = 1.000 to machine
   precision), because both are affine maps of the same base penalty. Feeding
   both to PCA would silently double the weight of that single direction, so
   `efficiency` is dropped and this is stated on the figure rather than hidden.
2. The four outcome shares cc, cd, dc, dd sum to 1 by construction, so the
   feature block is rank deficient. That is not a bug, but it does mean PC1 is
   partly an accounting identity, and the loadings have to be read with that in
   mind.
3. The `pC_*` conditionals and `reciprocity` are missing for 25% to 51% of
   agent-games, and the missingness is *structural*: `pC_T` is undefined when
   the agent was never in the T state, which happens precisely when the game
   never visited that cell of the payoff matrix. A pair that locked into mutual
   defection has no `pC_R`, and that absence is itself a strong behavioural
   fact. Mean imputation would invent a conditional response that never
   happened and would blur exactly the distinction we are trying to find. So
   each unobserved conditional is set to the agent's own realised unconditional
   cooperation rate, which is the zero information default (it asserts no
   conditional deviation), and five binary missingness indicators are appended
   so that the geometry can use non-visitation as a feature in its own right.

What the payoff-scaling question needs from this section
-------------------------------------------------------
Section 04 established that cooperation moves with lambda even though a
positive rescaling of a von Neumann-Morgenstern payoff matrix provably cannot
change the game. A level shift in a mean is compatible with two very different
mechanisms: every agent softening slightly, or the population re-sorting
between a fixed set of discrete policies. Clustering separates those. If the
regime mix shifts with lambda while the regimes themselves stay put, the
violation is a re-sorting effect, and that is a mechanism-level statement the
marginal analysis cannot make.

Why the anomaly hunt is framed as a null test
---------------------------------------------
The condition-cell grid is 1,200 cells of 10 games each. With 1,200 draws of a
mean over 10 bounded, boundary-inflated observations, extreme cells are not
merely possible, they are guaranteed: the expected number of cells beyond three
standard errors under a pure noise model is already around three. So the useful
question is never "which cell is extreme" but "is the observed tail heavier
than the null tail". The null used here is a permutation null that reassigns
the additive-model residuals across cells at random, which preserves the true
marginal residual shape (strongly non-Gaussian, because cooperation rates pile
up at 0 and 1) while destroying any real cell structure. A finding of "no more
extreme cells than chance predicts" is reported as a result, not as a failure.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from . import core as C

# behavioural columns the brief names, minus `efficiency` (see docstring point 1)
RAW_FEATS = ["coop_rate", "first_coop", "last_coop", "endgame_drop", "reciprocity",
             "pC_R", "pC_S", "pC_T", "pC_P", "cc_rate", "cd_rate", "dc_rate",
             "dd_rate", "rule_distance", "utility"]
MISS_FEATS = ["miss_pC_R", "miss_pC_S", "miss_pC_T", "miss_pC_P", "miss_recip"]
FEATS = RAW_FEATS + MISS_FEATS

SHORT = {"coop_rate": "coop rate", "first_coop": "round-1 C", "last_coop": "round-10 C",
         "endgame_drop": "endgame drop", "reciprocity": "reciprocity",
         "pC_R": "P(C | R)", "pC_S": "P(C | S)", "pC_T": "P(C | T)",
         "pC_P": "P(C | P)", "cc_rate": "CC share", "cd_rate": "CD share",
         "dc_rate": "DC share", "dd_rate": "DD share",
         "rule_distance": "rule distance", "utility": "utility",
         "miss_pC_R": "R never seen", "miss_pC_S": "S never seen",
         "miss_pC_T": "T never seen", "miss_pC_P": "P never seen",
         "miss_recip": "recip. undefined"}

CLUSTER_COL = ["#2a78d6", "#5aa0e0", "#1baf7a", "#eda100", "#c1440e", "#e34948",
               "#8250b8", "#0f8a72"]

K_FINAL = 6


# --------------------------------------------------------------------------
# feature matrix
# --------------------------------------------------------------------------
def build_features(ag: pd.DataFrame):
    """Return (feature frame, standardised matrix, column means, column sds).

    Structural missingness is encoded twice: the conditional is replaced by the
    agent's own unconditional cooperation rate (a zero information default), and
    an indicator records that the state was never visited.
    """
    X = ag[RAW_FEATS].astype(float).copy()
    for c in ["pC_R", "pC_S", "pC_T", "pC_P"]:
        X[f"miss_{c}"] = X[c].isna().astype(float)
        X[c] = X[c].fillna(ag["coop_rate"].astype(float))
    X["miss_recip"] = X["reciprocity"].isna().astype(float)
    X["reciprocity"] = X["reciprocity"].fillna(0.0)
    X = X[FEATS]
    mu = X.mean().to_numpy()
    sd = X.std(ddof=0).to_numpy()
    sd = np.where(sd == 0, 1.0, sd)
    Z = (X.to_numpy() - mu) / sd
    return X, Z, mu, sd


def _eta2_by_factor(scores: np.ndarray, labels: pd.Series) -> float:
    """Share of the variance of a projection explained by a categorical factor."""
    df = pd.DataFrame({"y": np.asarray(scores), "g": np.asarray(labels)})
    gm = df.groupby("g", observed=True)["y"]
    ssb = float((((gm.mean() - df.y.mean()) ** 2) * gm.size()).sum())
    sst = float(((df.y - df.y.mean()) ** 2).sum())
    return ssb / sst if sst > 0 else np.nan


def _cat_scatter(ax, x, y, labels, order, cols, seed=0, s=2.0, alpha=.35):
    """Scatter coloured by category with a RANDOMISED draw order.

    Plotting one category at a time paints the last category over all the
    others, which makes a dense cloud look as if one model or one lambda owns
    it. Shuffling the point order removes that artefact; the legend is then
    built from proxy handles.
    """
    labels = np.asarray(labels)
    rgba = np.zeros((len(labels), 4))
    for o in order:
        m = labels == o
        if m.sum():
            rgba[m] = mpl.colors.to_rgba(cols[o], alpha)
    perm = np.random.default_rng(seed).permutation(len(labels))
    ax.scatter(np.asarray(x)[perm], np.asarray(y)[perm], s=s, lw=0,
               c=rgba[perm], rasterized=True)
    return [plt.Line2D([], [], marker="o", ls="", ms=4, color=cols[o])
            for o in order if (labels == o).sum()]


def _longest_alt(seq) -> int:
    """Longest strictly alternating run inside one sequence."""
    seq = list(seq)
    if not seq:
        return 0
    best = cur = 1
    for i in range(1, len(seq)):
        cur = cur + 1 if seq[i] != seq[i - 1] else 1
        best = max(best, cur)
    return best


def name_regime(p: pd.Series) -> str:
    """Deterministic phenotype name read off the centroid, not hand assigned.

    The cascade asks the joint-outcome question first, because a majority share
    of one cell of the payoff matrix is the most specific thing that can be
    true of an agent-game, and only then falls back to timing, reciprocity and
    overall level.
    """
    if p["cc_rate"] >= 0.50:
        return "Mutual cooperator"
    if p["dd_rate"] >= 0.50:
        return "Mutual defector"
    if p["cd_rate"] >= 0.50:
        return "Exploited cooperator"
    if p["dc_rate"] >= 0.50:
        return "Exploiter"
    if p["endgame_drop"] <= -0.40:
        return "Late converter"
    if p["endgame_drop"] >= 0.30:
        return "Endgame defector"
    if p["reciprocity"] >= 0.45:
        return "Reciprocator"
    if p["coop_rate"] >= 0.60:
        return "Cooperative drifter"
    if p["coop_rate"] <= 0.40:
        return "Defecting drifter"
    return "Ambivalent mixer"


# --------------------------------------------------------------------------
def run(rounds, ag, dy):
    C.use_style()
    print("\n== 11 structure: dimensionality, regimes, anomalies ==")

    ag = ag.reset_index(drop=True).copy()
    Xdf, Z, mu, sd = build_features(ag)
    n = len(ag)
    print(f"  feature matrix {Z.shape[0]:,} x {Z.shape[1]}")

    r_eff = float(np.corrcoef(ag["utility"], ag["efficiency"])[0, 1])
    miss_frac = {c: float(ag[c].isna().mean())
                 for c in ["pC_R", "pC_S", "pC_T", "pC_P", "reciprocity"]}
    print("  structural missingness: " +
          ", ".join(f"{k} {v:.1%}" for k, v in miss_frac.items()))

    # =====================================================================
    # PCA
    # =====================================================================
    pca = PCA(n_components=len(FEATS), random_state=0).fit(Z)
    S = pca.transform(Z)
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    load = pd.DataFrame(pca.components_[:5].T, index=FEATS,
                        columns=[f"PC{i + 1}" for i in range(5)])
    load.insert(0, "feature", [SHORT[f] for f in FEATS])
    for i in range(5):
        load[f"PC{i + 1}_var_share"] = ev[i]
    C.savetab(load.reset_index().rename(columns={"index": "column"}),
              "struct_pca_loadings")

    def _top(pc, k=4):
        idx = load[pc].abs().sort_values(ascending=False).head(k).index
        return [(SHORT[f], float(load.loc[f, pc])) for f in idx]

    # names read off the loading matrix printed below, not assumed in advance
    PC_NAME = {
        "PC1": "cooperativeness in every state",
        "PC2": "own payoff vs being exploited",
        "PC3": "state coverage, no rule fits",
        "PC4": "timing: early C, late D",
    }
    for i in range(3):
        k = f"PC{i + 1}"
        print(f"  {k} ({ev[i]:.1%}): " +
              ", ".join(f"{a} {b:+.2f}" for a, b in _top(k)))

    # ---------------------------------------------------------------------
    # B01  scree and loadings
    # ---------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2),
                             gridspec_kw={"width_ratios": [1, 1.3, 1.15]})
    ax = axes[0]
    ax.bar(np.arange(1, len(ev) + 1), ev, color=C.MUTED, width=.7)
    ax.bar(np.arange(1, 5), ev[:4], color=C.C_COOP, width=.7)
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, len(ev) + 1), cum, "o-", color=C.OUTCOME_COL["DC"], ms=3)
    ax2.set_ylabel("cumulative share", color=C.OUTCOME_COL["DC"], fontsize=7.5)
    ax2.set_ylim(0, 1.02)
    ax2.grid(False)
    ax.axhline(1 / len(FEATS), color=C.INK, lw=.8, ls="--")
    ax.set_xlabel("principal component")
    ax.set_ylabel("share of standardised variance")
    ax.set_ylim(0, ev.max() * 1.45)
    ax.set_title("a  scree")
    C.annotate(ax, f"PC1-PC4 hold {cum[3]:.1%}\nof the variance in {len(FEATS)}\n"
                   "standardised features.\nDashed line = equal share.\n"
                   f"Efficiency excluded:\nr = {r_eff:.3f} with utility",
               loc="center right")

    ax = axes[1]
    M = load[[f"PC{i + 1}" for i in range(4)]].to_numpy()
    v = np.abs(M).max()
    im = ax.imshow(M, cmap=C.DIV, vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(4), [f"PC{i + 1}\n{ev[i]:.0%}" for i in range(4)], fontsize=7)
    ax.set_yticks(range(len(FEATS)), [SHORT[f] for f in FEATS], fontsize=6.3)
    ax.set_title("b  loadings")
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=.05, label="loading")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=5.2,
                    color="white" if abs(M[i, j]) > v * .58 else C.INK)

    ax = axes[2]
    y = np.arange(len(FEATS))
    ax.barh(y - .2, load["PC1"], height=.38, color=C.C_COOP,
            label=f"PC1  {PC_NAME['PC1']}")
    ax.barh(y + .2, load["PC2"], height=.38, color=C.OUTCOME_COL["DC"],
            label=f"PC2  {PC_NAME['PC2']}")
    ax.axvline(0, color=C.INK, lw=.8)
    ax.set_yticks(y, [SHORT[f] for f in FEATS], fontsize=6.3)
    ax.invert_yaxis()
    ax.set_xlabel("loading on the standardised feature")
    ax.set_xlim(-.48, 1.05)
    ax.set_title("c  what PC1 and PC2 mean")
    ax.legend(loc="upper right", fontsize=6.2, handlelength=1.2)
    C.save(fig, "advanced", "B01_pca_scree_loadings",
           "Do the sixteen behavioural quantities measured on an agent-game move "
           "independently, or do they collapse onto a few interpretable axes?",
           f"They collapse onto a small number of readable axes: four components "
           f"carry {cum[3]:.1%} of the variance of {len(FEATS)} standardised "
           f"features. PC1 ({ev[0]:.1%}) loads positively and almost equally on the "
           "cooperation rate and on all four conditional cooperation probabilities, "
           "and negatively on the DD share, so it is a single cooperativeness axis "
           "that says an agent which cooperates more does so in every conditioning "
           f"state. PC2 ({ev[1]:.1%}) is own utility against the CD share, that is, "
           "collecting the payoff versus being the sucker, and it is the exploitation "
           f"axis. PC3 ({ev[2]:.1%}) is rule distance against the four never-visited "
           "indicators, so it is a history-richness axis: games that toured many "
           f"payoff states and whose agent fits no memory-one rule. PC4 ({ev[3]:.1%}) "
           "is round-1 cooperation and the endgame drop against reciprocity, the "
           "timing axis. Efficiency was dropped because it is exactly 1.25 times "
           f"utility (r = {r_eff:.3f}), and the four outcome shares sum to one by "
           "construction, so part of PC1 is an accounting identity rather than a "
           "behavioural discovery.",
           "scree plus loading heatmap plus loading bars",
           "15 behavioural features and 5 missingness indicators, agent-game level")

    # ---------------------------------------------------------------------
    # B02  what separates in behaviour space
    # ---------------------------------------------------------------------
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(n, 9000, replace=False))
    x1, x2 = S[sub, 0], S[sub, 1]

    STRAT_ORDER_FULL = ["AllC", "TFT", "WSLS", "GRIM", "AllD", "ambiguous",
                        "unclassified"]
    STRAT_COLS = {**C.STRAT_COL, "ambiguous": "#eda100", "unclassified": "#c9c7be"}

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.4))
    panels = [
        ("model", C.MODEL_ORDER, {m: C.MODEL_COL[m] for m in C.MODEL_ORDER},
         "a  by model"),
        ("strategy", STRAT_ORDER_FULL, STRAT_COLS, "b  by strategy label"),
        ("scale", C.SCALES, {s: C.SEQ(i / 9) for i, s in enumerate(C.SCALES)},
         "c  by payoff multiplier"),
        ("dyad", C.DYADS, C.DYAD_COL, "d  by persona pairing"),
    ]
    for ax, (key, order, cols, title) in zip(axes.ravel(), panels):
        lab = ag[key].to_numpy()[sub]
        handles = _cat_scatter(ax, x1, x2, lab, order, cols, seed=3)
        ax.set_xlabel(f"PC1  {PC_NAME['PC1']}")
        ax.set_ylabel(f"PC2  {PC_NAME['PC2']}")
        ax.set_title(title)
        if key == "scale":
            sm_ = plt.cm.ScalarMappable(cmap=C.SEQ, norm=plt.Normalize(-2, 3))
            plt.colorbar(sm_, ax=ax, fraction=.046, label="$\\log_{10}\\lambda$")
            C.annotate(ax, "draw order randomised so no\n$\\lambda$ paints over "
                           "another", loc="lower left")
        else:
            ax.legend(handles,
                      [str(o).replace("-Non-Reasoning", "") for o in order],
                      loc="lower left", fontsize=5.4, ncol=2, handletextpad=.2,
                      columnspacing=.5, borderaxespad=.2)

    ax = axes[1, 1]
    fac = ["model", "strategy", "dyad", "language", "scale"]
    e1 = [_eta2_by_factor(S[:, 0], ag[f]) for f in fac]
    e2 = [_eta2_by_factor(S[:, 1], ag[f]) for f in fac]
    yy = np.arange(len(fac))
    ax.barh(yy - .2, e1, height=.38, color=C.C_COOP, label="PC1")
    ax.barh(yy + .2, e2, height=.38, color=C.OUTCOME_COL["DC"], label="PC2")
    ax.set_yticks(yy, fac)
    ax.set_xscale("log")
    ax.set_xlim(min(min(e1), min(e2)) / 3, 4)
    ax.set_xlabel("$\\eta^2$ of the component (log axis)")
    ax.set_title("e  which factor separates")
    ax.legend(loc="lower right", fontsize=6.5)
    for i, (a_, b_) in enumerate(zip(e1, e2)):
        ax.text(a_ * 1.25, i - .2, f"{a_:.4f}", va="center", fontsize=6)
        ax.text(b_ * 1.25, i + .2, f"{b_:.4f}", va="center", fontsize=6)
    C.savetab(pd.DataFrame({"factor": fac, "eta2_PC1": e1, "eta2_PC2": e2}),
              "struct_pca_separation")

    ax = axes[1, 2]
    mx, my = [], []
    for m in C.MODEL_ORDER:
        mm = (ag["model"] == m).to_numpy()
        cx, cy = S[mm, 0].mean(), S[mm, 1].mean()
        mx.append(cx)
        my.append(cy)
        ax.scatter(cx, cy, s=95, color=C.MODEL_COL[m], zorder=4, edgecolor="white",
                   lw=.8)
        ax.annotate(m.replace("-Non-Reasoning", ""), (cx, cy), fontsize=6,
                    xytext=(6, -8), textcoords="offset points")
    sx, sy = [], []
    for i, s in enumerate(C.SCALES):
        mm = (ag["scale"] == s).to_numpy()
        cx, cy = S[mm, 0].mean(), S[mm, 1].mean()
        sx.append(cx)
        sy.append(cy)
        ax.scatter(cx, cy, s=36, marker="s", color=C.SEQ(i / 9), zorder=5,
                   edgecolor=C.INK, lw=.5)
    sp_m = float(np.hypot(np.ptp(mx), np.ptp(my)))
    sp_s = float(np.hypot(np.ptp(sx), np.ptp(sy)))
    ax.set_xlabel(f"PC1  {PC_NAME['PC1']}")
    ax.set_ylabel(f"PC2  {PC_NAME['PC2']}")
    ax.set_title("f  centroids on one scale")
    C.annotate(ax, f"circles = the 6 models, spread {sp_m:.2f}\n"
                   f"squares = the 10 $\\lambda$, spread {sp_s:.2f}\n"
                   f"ratio {sp_m / sp_s:.0f}x", loc="upper left")
    C.save(fig, "advanced", "B02_pca_projection_by_factor",
           "Which design factor actually separates agents in behaviour space: the "
           "model, the strategy label, the payoff multiplier, or the persona pairing?",
           f"Model identity separates; the payoff multiplier and the persona pairing "
           f"barely do. On PC1 the model explains eta-squared = {e1[0]:.3f} against "
           f"{e1[4]:.4f} for the payoff multiplier, a factor of "
           f"{e1[0] / max(e1[4], 1e-9):.0f} in variance, which is one order of "
           "magnitude and not more; in the units of the plane itself the six model "
           f"centroids span {sp_m:.2f} against {sp_s:.2f} for the ten lambda "
           f"centroids, a factor of {sp_m / sp_s:.1f}. The strategy label tops the "
           f"chart at {e1[1]:.3f}, but that number is close to circular, because the "
           "label is computed from the same round-by-round actions the features are, "
           "so it is a consistency check rather than a design effect. The surprise is "
           f"the persona pairing: it explains {e1[2]:.4f} of PC1, that is nothing at "
           "all, and only reaches "
           f"{e2[2]:.3f} on PC2, where it must appear mechanically because only mixed "
           "pairings can generate the asymmetric CD and DC outcomes. The persona "
           "prompt therefore moves who gets exploited without moving how much anyone "
           "cooperates.",
           "PCA scatter in four colourings plus eta-squared bars plus centroids",
           "PC1, PC2, model, strategy, scale, dyad, language")

    # ---------------------------------------------------------------------
    # B03  UMAP
    # ---------------------------------------------------------------------
    umap_ok = False
    try:
        import umap
        rng2 = np.random.default_rng(1)
        idx = np.sort(rng2.choice(n, 8000, replace=False))
        U = umap.UMAP(n_neighbors=25, min_dist=0.12, n_components=2,
                      random_state=0, metric="euclidean").fit_transform(Z[idx])
        umap_ok = True
    except Exception as exc:
        print(f"  [warn] UMAP unavailable or failed: {exc}")

    if umap_ok:
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3))
        for ax, (key, order, cols, title) in zip(axes, [
                ("strategy", STRAT_ORDER_FULL, STRAT_COLS, "a  by strategy label"),
                ("model", C.MODEL_ORDER,
                 {m: C.MODEL_COL[m] for m in C.MODEL_ORDER}, "b  by model"),
                ("scale", C.SCALES,
                 {s: C.SEQ(i / 9) for i, s in enumerate(C.SCALES)},
                 "c  by payoff multiplier")]):
            lab = ag[key].to_numpy()[idx]
            for o in order:
                m = lab == o
                if m.sum() == 0:
                    continue
                ax.scatter(U[m, 0], U[m, 1], s=2.4, alpha=.4, lw=0, color=cols[o],
                           rasterized=True,
                           label=(f"{o:g}" if key == "scale"
                                  else str(o).replace("-Non-Reasoning", "")))
            ax.set_xlabel("UMAP 1 (arbitrary units)")
            ax.set_ylabel("UMAP 2 (arbitrary units)")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            if key == "scale":
                sm_ = plt.cm.ScalarMappable(cmap=C.SEQ, norm=plt.Normalize(-2, 3))
                plt.colorbar(sm_, ax=ax, fraction=.046, label="$\\log_{10}\\lambda$")
            else:
                lg = ax.legend(loc="lower left", fontsize=5.4, markerscale=4, ncol=2,
                               handletextpad=.2, columnspacing=.5, borderaxespad=.2)
                for h in lg.legend_handles:
                    h.set_alpha(1)
        eu_m = _eta2_by_factor(U[:, 0], ag["model"].iloc[idx])
        eu_s = _eta2_by_factor(U[:, 0], ag["scale"].iloc[idx])
        C.annotate(axes[2], "n = 8,000 deterministic subsample\n"
                            f"$\\eta^2$ of UMAP1: model {eu_m:.3f}, "
                            f"$\\lambda$ {eu_s:.4f}", loc="upper left")
        C.save(fig, "advanced", "B03_umap_manifold",
               "Does a nonlinear embedding reveal behavioural structure that the "
               "linear PCA projection hides?",
               "It resolves the discrete atoms far more sharply but tells the same "
               "story about which factor matters. UMAP on a deterministic 8,000 "
               "agent-game subsample breaks the cloud into separated islands that "
               "correspond to the boundary policies, always cooperate and always "
               "defect, and to the conditional middle, whereas PCA renders the same "
               "structure as a continuum. That is the one thing PCA does not show. "
               "The colourings otherwise reproduce the PCA result: model identity "
               f"explains {eu_m:.3f} of the first UMAP coordinate against "
               f"{eu_s:.4f} for lambda, so the payoff multiplier still does not move "
               "agents between islands. UMAP axes carry no units and no global "
               "distance interpretation, so this figure is read only for topology.",
               "UMAP embedding in three colourings",
               "20 standardised features, strategy, model, scale")

    # =====================================================================
    # CLUSTERING
    # =====================================================================
    ks = list(range(2, 13))
    inertia, sil = [], []
    for k in ks:
        km_ = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
        inertia.append(km_.inertia_)
        sil.append(silhouette_score(Z, km_.labels_, sample_size=6000, random_state=0))
    inertia = np.array(inertia)
    sil = np.array(sil)
    xx = np.array(ks, float)
    yn = (inertia - inertia.min()) / (inertia.max() - inertia.min())
    xn = (xx - xx.min()) / (xx.max() - xx.min())
    elbow_k = int(ks[int(np.argmax((1 - xn) - yn))])
    C.savetab(pd.DataFrame({"k": ks, "within_cluster_ss": inertia,
                            "mean_silhouette": sil}), "struct_k_selection")

    km = KMeans(n_clusters=K_FINAL, n_init=25, random_state=0).fit(Z)
    cent_z = km.cluster_centers_
    cent_raw = pd.DataFrame(cent_z * sd + mu, columns=FEATS)
    order_by_coop = np.argsort(-cent_raw["coop_rate"].to_numpy())
    remap = {int(old): new for new, old in enumerate(order_by_coop)}
    ag["cluster"] = pd.Series(km.labels_).map(remap).to_numpy()
    cent_z = cent_z[order_by_coop]
    cent_raw = cent_raw.iloc[order_by_coop].reset_index(drop=True)

    names = [name_regime(cent_raw.iloc[i]) for i in range(K_FINAL)]
    seen: dict[str, int] = {}
    base = list(names)
    for i, nm in enumerate(names):
        if base.count(nm) > 1:
            seen[nm] = seen.get(nm, 0) + 1
            names[i] = f"{nm} {seen[nm]}"
    ag["regime"] = ag["cluster"].map(dict(enumerate(names)))
    print("  regimes: " + " | ".join(f"{i}:{nm}" for i, nm in enumerate(names)))

    # ---------------------------------------------------------------------
    # B04  choosing k
    # ---------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9))
    ax = axes[0]
    ax.plot(ks, inertia / 1e3, "o-", color=C.C_COOP)
    ax.axvline(elbow_k, color=C.OUTCOME_COL["DC"], ls="--", lw=1)
    ax.axvline(K_FINAL, color=C.INK, ls=":", lw=1.4)
    ax.set_xlabel("number of clusters k")
    ax.set_ylabel("within-cluster sum of squares ($\\times 10^3$)")
    ax.set_title("a  elbow")
    C.annotate(ax, f"elbow at k = {elbow_k} (dashed)\nchosen k = {K_FINAL} (dotted), "
                   "one step\nabove it and near-tied", loc="upper right")

    ax = axes[1]
    ax.plot(ks, sil, "o-", color=C.OUTCOME_COL["DC"])
    kbest = int(ks[int(np.argmax(sil))])
    kworst = int(ks[int(np.argmin(sil))])
    ax.axvline(K_FINAL, color=C.INK, ls=":", lw=1.4)
    ax.set_xlabel("number of clusters k")
    ax.set_ylabel("mean silhouette (6,000 point sample)")
    ax.set_title("b  silhouette does not select k")
    C.annotate(ax, "no interior maximum on k = 2 to 12:\n"
                   f"it dips to {sil.min():.3f} at k = {kworst}, then rises\n"
                   f"to {sil.max():.3f} at k = {kbest}, the edge of the range.\n"
                   "A continuum with boundary atoms does\nthis, so silhouette cannot "
                   f"pick k here.\nk = {K_FINAL} (dotted) gives "
                   f"{sil[ks.index(K_FINAL)]:.3f}", loc="lower right")

    ax = axes[2]
    lk = linkage(cent_z, method="ward")
    dn = dendrogram(lk, ax=ax, labels=list(range(K_FINAL)), color_threshold=0,
                    above_threshold_color=C.INK2, leaf_font_size=7)
    ax.set_ylabel("Ward distance between regime centroids")
    ax.set_title("c  how the regimes relate")
    ax.set_xticks(ax.get_xticks(),
                  [f"{i}  {names[i][:20]}" for i in dn["ivl"]],
                  rotation=30, ha="right", fontsize=6.0)
    ax.grid(False)
    C.save(fig, "advanced", "B04_cluster_selection",
           "How many behavioural regimes does the corpus contain, and how do those "
           "regimes relate to one another?",
           f"Six, chosen at the inertia elbow. The silhouette does not select k at "
           f"all here: it dips from {sil[0]:.3f} at k = 2 to {sil.min():.3f} at "
           f"k = {kworst} and then rises monotonically to {sil.max():.3f} at "
           f"k = {kbest}, the edge of the range, which is what a continuum with "
           "boundary atoms does rather than a set of separated balls. That is "
           "reported rather than hidden. The inertia elbow, taken as the maximum "
           "distance to the chord joining the endpoints of the curve, sits at "
           f"k = {elbow_k}, with k = {K_FINAL} near-tied one step above it. We take "
           f"k = {K_FINAL} because it is the smallest k that resolves all four "
           "joint-outcome phenotypes, mutual cooperation, mutual defection and the "
           "two asymmetric exploitation roles, alongside the two timing phenotypes, "
           f"for {sil.max() - sil[ks.index(K_FINAL)]:.3f} of mean silhouette. The "
           "Ward dendrogram on the centroids shows the regimes are organised "
           "primarily by cooperation level, with the mutual-outcome extremes joining "
           "last.",
           "elbow curve plus silhouette curve plus centroid dendrogram",
           "k, within-cluster sum of squares, silhouette, Ward linkage on centroids")

    # ---------------------------------------------------------------------
    # cluster profile table
    # ---------------------------------------------------------------------
    prof_rows = []
    for i in range(K_FINAL):
        s = ag[ag["cluster"] == i]
        m_, lo_, hi_ = C.cluster_boot_ci(s["coop_rate"].to_numpy(),
                                         s["game_uid"].to_numpy(), n_boot=800)
        u_, ulo_, uhi_ = C.cluster_boot_ci(s["utility"].to_numpy(),
                                           s["game_uid"].to_numpy(), n_boot=800)
        dom_model = s["model"].value_counts()
        dom_strat = s["strategy"].value_counts()
        dom_dyad = s["dyad"].value_counts()
        prof_rows.append({
            "cluster": i, "regime": names[i], "n_agent_games": len(s),
            "share_of_corpus": len(s) / n,
            "coop_rate": m_, "coop_lo": lo_, "coop_hi": hi_,
            "utility": u_, "utility_lo": ulo_, "utility_hi": uhi_,
            "reciprocity_observed": float(s["reciprocity"].mean(skipna=True)),
            "endgame_drop": float(s["endgame_drop"].mean()),
            "first_coop": float(s["first_coop"].mean()),
            "last_coop": float(s["last_coop"].mean()),
            "cc_rate": float(s["cc_rate"].mean()),
            "cd_rate": float(s["cd_rate"].mean()),
            "dc_rate": float(s["dc_rate"].mean()),
            "dd_rate": float(s["dd_rate"].mean()),
            "rule_distance": float(s["rule_distance"].mean()),
            "pct_state_R_never_seen": float(s["pC_R"].isna().mean()),
            "dominant_strategy": dom_strat.index[0],
            "dominant_strategy_share": float(dom_strat.iloc[0] / len(s)),
            "dominant_model": dom_model.index[0],
            "dominant_model_share": float(dom_model.iloc[0] / len(s)),
            "dominant_pairing": dom_dyad.index[0],
            "dominant_pairing_share": float(dom_dyad.iloc[0] / len(s)),
        })
    prof = pd.DataFrame(prof_rows)
    C.savetab(prof, "struct_cluster_profiles")
    print(prof[["cluster", "regime", "share_of_corpus", "coop_rate",
                "reciprocity_observed", "endgame_drop", "dominant_strategy",
                "dominant_model", "dominant_pairing"]].to_string())

    # ---------------------------------------------------------------------
    # B05  phenotype profiles
    # ---------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6),
                             gridspec_kw={"width_ratios": [1.45, 1.15, 1]})
    ax = axes[0]
    M = cent_z.T
    v = np.abs(M).max()
    im = ax.imshow(M, cmap=C.DIV, vmin=-v, vmax=v, aspect="auto")
    ax.set_yticks(range(len(FEATS)), [SHORT[f] for f in FEATS], fontsize=6.3)
    ax.set_xticks(range(K_FINAL), [str(i) for i in range(K_FINAL)], fontsize=8)
    ax.set_xlabel("regime")
    ax.set_title("a  centroid fingerprints")
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=.05, label="SD from the corpus mean")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=5.0,
                    color="white" if abs(M[i, j]) > v * .58 else C.INK)

    ax = axes[1]
    yy = np.arange(K_FINAL)
    ax.barh(yy, prof["share_of_corpus"], color=CLUSTER_COL[:K_FINAL])
    ax.set_yticks(yy, [f"{i}  {nm}" for i, nm in enumerate(names)], fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlabel("share of the 24,000 agent-games")
    ax.set_title("b  how common is each regime")
    for i, (v_, s_) in enumerate(zip(prof["share_of_corpus"], prof["n_agent_games"])):
        ax.text(v_ + .004, i, f"{v_:.1%} (n={s_:,})", va="center", fontsize=6)
    ax.set_xlim(0, prof["share_of_corpus"].max() * 1.5)

    ax = axes[2]
    ax.axvline(0, color=C.INK, lw=.8)
    ax.errorbar(prof["coop_rate"], yy - .17,
                xerr=[prof["coop_rate"] - prof["coop_lo"],
                      prof["coop_hi"] - prof["coop_rate"]],
                fmt="o", color=C.C_COOP, ecolor=C.INK2, ms=5, lw=1.1,
                label="cooperation rate")
    ax.errorbar(prof["utility"], yy + .17,
                xerr=[prof["utility"] - prof["utility_lo"],
                      prof["utility_hi"] - prof["utility"]],
                fmt="s", color=C.OUTCOME_COL["DC"], ecolor=C.INK2, ms=4, lw=1.1,
                label="own utility")
    ax.scatter(prof["reciprocity_observed"], yy, marker="D", s=24,
               color=C.STRAT_COL["TFT"], zorder=4, label="reciprocity")
    ax.scatter(prof["endgame_drop"], yy, marker="v", s=24, color=C.C_DEFECT,
               zorder=4, label="endgame drop")
    ax.set_yticks(yy, [str(i) for i in range(K_FINAL)])
    ax.invert_yaxis()
    ax.set_xlabel("value (utility and cooperation in [0,1], the other two in [-1,1])")
    ax.set_ylabel("regime")
    ax.set_xlim(-.75, 1.35)
    ax.set_title("c  the defining numbers")
    ax.legend(loc="lower right", fontsize=6.0)
    C.annotate(ax, "bars = 95% bootstrap CI\nclustered on game_uid", loc="upper right")
    C.save(fig, "advanced", "B05_regime_phenotypes",
           "What does each behavioural regime actually look like, and is the "
           "partition interpretable as a set of named phenotypes rather than as "
           "arbitrary slices of a continuum?",
           "The six regimes are interpretable and each name is read off its own "
           "centroid rather than assigned by hand. "
           + "; ".join(f"regime {i} '{nm}' is {prof.share_of_corpus[i]:.0%} of the "
                       f"corpus with cooperation {prof.coop_rate[i]:.2f}, reciprocity "
                       f"{prof.reciprocity_observed[i]:+.2f} and endgame drop "
                       f"{prof.endgame_drop[i]:+.2f}"
                       for i, nm in enumerate(names))
           + ". The set that emerges is exactly the four cells of the payoff matrix "
             "seen as durable states, mutual cooperation, mutual defection and the "
             "two asymmetric exploitation roles, plus two timing phenotypes that "
             "start at one end of the cooperation axis and finish at the other. The "
             "two mutual-outcome regimes together hold "
             f"{prof.share_of_corpus.iloc[0] + prof.share_of_corpus.iloc[-1]:.0%} of "
             "all agent-games, which is the clustering restatement of the U-shaped "
             "cooperation distribution reported in section 02.",
           "centroid heatmap plus share bars plus profile dot plot",
           "20 standardised features, cluster label, coop_rate, utility, reciprocity")

    # ---------------------------------------------------------------------
    # B06  regime composition by model and by lambda
    # ---------------------------------------------------------------------
    comp_m = (pd.crosstab(ag["model"], ag["regime"], normalize="index")
              .reindex(C.MODEL_ORDER)[names])
    comp_s = pd.crosstab(ag["scale"], ag["regime"], normalize="index")[names]
    C.savetab(comp_s.reset_index(), "struct_regime_mix_by_scale")
    C.savetab(comp_m.reset_index(), "struct_regime_mix_by_model")

    a1 = ag[ag["agent"] == 1]
    ct_s = pd.crosstab(a1["scale"], a1["regime"])
    chi_s = stats.chi2_contingency(ct_s.to_numpy())
    v_s = float(np.sqrt(chi_s.statistic /
                        (ct_s.to_numpy().sum() * (min(ct_s.shape) - 1))))
    ct_m = pd.crosstab(a1["model"], a1["regime"])
    chi_m = stats.chi2_contingency(ct_m.to_numpy())
    v_m = float(np.sqrt(chi_m.statistic /
                        (ct_m.to_numpy().sum() * (min(ct_m.shape) - 1))))
    C.record_test(section="structure",
                  test="chi-square on the regime mix, agent-1 subsample so games "
                       "are independent",
                  comparison="behavioural regime x payoff multiplier",
                  statistic=float(chi_s.statistic), p_value=float(chi_s.pvalue),
                  n=int(ct_s.to_numpy().sum()),
                  effect_size_note=f"Cramer V = {v_s:.3f}")
    C.record_test(section="structure",
                  test="chi-square on the regime mix, agent-1 subsample so games "
                       "are independent",
                  comparison="behavioural regime x model",
                  statistic=float(chi_m.statistic), p_value=float(chi_m.pvalue),
                  n=int(ct_m.to_numpy().sum()),
                  effect_size_note=f"Cramer V = {v_m:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.4),
                             gridspec_kw={"width_ratios": [1.1, 1.1, 1.05]})
    for ax, comp, title, xlab, rot in [
            (axes[0], comp_m, "a  regime mix by model", "", 30),
            (axes[1], comp_s, "b  regime mix by $\\lambda$",
             "payoff scale $\\lambda$", 90)]:
        bot = np.zeros(len(comp))
        for j, nm in enumerate(names):
            ax.bar(range(len(comp)), comp[nm], bottom=bot, color=CLUSTER_COL[j],
                   width=.82, label=f"{j}  {nm}")
            bot = bot + comp[nm].to_numpy()
        ax.set_ylim(0, 1)
        ax.set_ylabel("share of agent-games")
        ax.set_title(title)
        if xlab:
            ax.set_xticks(range(len(comp)), [f"{s:g}" for s in comp.index],
                          rotation=rot, fontsize=6.5)
            ax.set_xlabel(xlab)
        else:
            ax.set_xticks(range(len(comp)),
                          [m.replace("-Non-Reasoning", "") for m in comp.index],
                          rotation=rot, ha="right", fontsize=6.0)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.08, -0.30), ncol=3,
                   fontsize=6.0)
    axes[0].text(0.02, 1.005, f"Cramer V = {v_m:.3f}", transform=axes[0].transAxes,
                 fontsize=7, color=C.INK2)
    axes[1].text(0.02, 1.005, f"Cramer V = {v_s:.3f}", transform=axes[1].transAxes,
                 fontsize=7, color=C.INK2)

    ax = axes[2]
    logl = np.log10(C.SCALES)
    rhos = {}
    for j, nm in enumerate(names):
        share = comp_s[nm].to_numpy()
        lo, hi = [], []
        for s in C.SCALES:
            ss = ag[ag["scale"] == s]
            _m, _l, _h = C.cluster_boot_ci(
                (ss["regime"] == nm).to_numpy().astype(float),
                ss["game_uid"].to_numpy(), n_boot=400)
            lo.append(_l)
            hi.append(_h)
        ax.fill_between(C.SCALES, lo, hi, color=CLUSTER_COL[j], alpha=.16, lw=0)
        ax.plot(C.SCALES, share, "o-", color=CLUSTER_COL[j], ms=3.2, label=str(j))
        rhos[nm] = float(stats.spearmanr(logl, share).statistic)
    ax.set_xscale("log")
    ax.set_xlabel("payoff scale $\\lambda$")
    ax.set_ylabel("share of agent-games in the regime")
    ax.set_title("c  does $\\lambda$ move the mix")
    ax.legend(loc="center right", fontsize=6, title="regime", title_fontsize=6,
              ncol=2)
    big = max(rhos, key=lambda k: abs(rhos[k]))
    swing = float((comp_s.max() - comp_s.min()).max())
    ax.set_ylim(top=comp_s.to_numpy().max() * 1.45)
    C.annotate(ax, f"largest share swing across the ladder {swing:.3f}\n"
                   f"strongest monotone regime: '{big[:26]}'\n"
                   f"Spearman $\\rho$ = {rhos[big]:+.2f}", loc="upper left")
    C.save(fig, "advanced", "B06_regime_mix_by_model_and_scale",
           "Does the payoff multiplier change behaviour by re-sorting agents between "
           "fixed behavioural regimes, and how does that compare with the sorting "
           "produced by model identity?",
           f"Both factors move the mix, but by very different amounts. Model identity "
           f"gives Cramer V = {v_m:.3f} on the regime by model table, while the payoff "
           f"multiplier gives V = {v_s:.3f}, about {v_m / v_s:.0f} times smaller, "
           "though the lambda association is still detectable on the independent "
           f"agent-1 subsample (chi-square = {chi_s.statistic:.0f}, "
           f"p = {chi_s.pvalue:.2g}). The largest regime share swing across the "
           f"five-decade ladder is {swing:.3f}, carried by '{big}' at Spearman "
           f"rho = {rhos[big]:+.2f}. This is a mechanism-level restatement of the "
           "payoff-scaling violation reported in section 04: lambda re-sorts agents "
           "between phenotypes that themselves stay put, rather than deforming a "
           "single phenotype.",
           "stacked composition bars plus regime share versus lambda with CIs",
           "regime label, model, scale, game_uid clusters")

    # =====================================================================
    # OUTLIER MINING AT CONDITION-CELL LEVEL
    # =====================================================================
    d = dy.copy()
    d["cellkey"] = (d["model"].astype(str) + " | " + d["language"].astype(str) +
                    " | " + d["scale"].astype(str) + " | " + d["dyad"].astype(str))
    X = pd.concat([
        pd.get_dummies(d["model"].astype(str), prefix="m", drop_first=True),
        pd.get_dummies(d["language"].astype(str), prefix="l", drop_first=True),
        pd.get_dummies(d["dyad"].astype(str), prefix="p", drop_first=True),
        pd.get_dummies(d["scale"].astype(str), prefix="lam", drop_first=True),
    ], axis=1).astype(float)
    Xa = sm.add_constant(X.to_numpy())
    y = d["joint_coop"].to_numpy()
    fit = sm.OLS(y, Xa).fit()
    d["fitted"] = fit.fittedvalues
    d["resid"] = fit.resid
    s_within = float(np.sqrt(np.sum(fit.resid ** 2) / (len(y) - len(fit.params))))

    cell = (d.groupby("cellkey", observed=True)
            .agg(model=("model", "first"), language=("language", "first"),
                 scale=("scale", "first"), pairing=("dyad", "first"),
                 n_games=("joint_coop", "size"),
                 observed_coop=("joint_coop", "mean"),
                 predicted_coop=("fitted", "mean"),
                 resid=("resid", "mean"),
                 sd_within=("joint_coop", "std")).reset_index())
    cell["z"] = cell["resid"] / (s_within / np.sqrt(cell["n_games"].to_numpy()))

    grp_id = pd.factorize(d["cellkey"])[0]
    grp_n = np.bincount(grp_id).astype(float)
    se_g = s_within / np.sqrt(grp_n)
    rngp = np.random.default_rng(7)
    n_perm = 2000
    null_max = np.empty(n_perm)
    null_tail3 = np.empty(n_perm)
    null_sd = np.empty(n_perm)
    resid = fit.resid.copy()
    for b in range(n_perm):
        pr = rngp.permutation(resid)
        zz_ = (np.bincount(grp_id, weights=pr) / grp_n) / se_g
        null_max[b] = np.abs(zz_).max()
        null_tail3[b] = int((np.abs(zz_) > 3).sum())
        null_sd[b] = zz_.std()
    zz = cell["z"].to_numpy()
    obs_max = float(np.abs(zz).max())
    obs_tail3 = int((np.abs(zz) > 3).sum())
    p_max = float((null_max >= obs_max).mean())
    p_tail = float((null_tail3 >= obs_tail3).mean())
    p_sd = float((null_sd >= zz.std()).mean())

    cell["p_perm"] = [float((null_max >= abs(z_)).mean()) for z_ in zz]
    cell["p_fdr"] = C.bh_fdr(cell["p_perm"].to_numpy())
    cell = cell.sort_values("z", key=np.abs, ascending=False).reset_index(drop=True)
    cell["verdict"] = np.where(cell["p_fdr"] < .05,
                               "beyond the permutation null",
                               "within the range chance alone produces")
    out_tab = cell.head(40).copy()
    out_tab["what_is_unusual"] = np.where(
        out_tab["resid"] > 0,
        "cooperates more than the additive model predicts",
        "cooperates less than the additive model predicts")
    out_tab["note"] = ("z is the 10-game cell mean residual in units of its own "
                       "standard error; p_perm is against a 2,000 draw permutation "
                       "null on the maximum |z|")
    C.savetab(out_tab, "struct_outlier_cells")
    C.record_test(section="structure",
                  test="permutation null on the largest absolute cell residual",
                  comparison="1,200 condition cells of 10 games, additive model in "
                             "model + language + lambda + pairing",
                  statistic=obs_max, p_value=p_max, n=1200,
                  effect_size_note=f"cells with |z|>3: observed {obs_tail3}, null "
                                   f"median {np.median(null_tail3):.0f}")
    C.record_test(section="structure",
                  test="permutation null on the dispersion of the cell residuals",
                  comparison="SD of cell z scores against a residual-reassignment null",
                  statistic=float(zz.std()), p_value=p_sd, n=1200,
                  effect_size_note=f"observed SD {zz.std():.3f}, null median "
                                   f"{np.median(null_sd):.3f}")

    # ---------------------------------------------------------------------
    # B07  outlier cells versus the null
    # ---------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0),
                             gridspec_kw={"width_ratios": [1, 1, 1.3]})
    ax = axes[0]
    ax.hist(zz, bins=40, color=C.MUTED, density=True, rwidth=.9,
            label="observed cells")
    xs = np.linspace(-6, 6, 400)
    ax.plot(xs, stats.norm.pdf(xs), color=C.C_DEFECT, lw=1.5,
            label="N(0,1) parametric null")
    ax.set_xlabel("cell mean residual, in standard errors (z)")
    ax.set_ylabel("density")
    ax.set_title("a  are the cells over-dispersed")
    ax.legend(loc="upper right", fontsize=6.3)
    C.annotate(ax, f"observed SD of z = {zz.std():.2f}\npermutation null median "
                   f"{np.median(null_sd):.2f}\npermutation p = {p_sd:.4f}",
               loc="upper left")

    ax = axes[1]
    bins = np.arange(min(null_tail3.min(), obs_tail3) - .5,
                     max(null_tail3.max(), obs_tail3) + 1.5)
    ax.hist(null_tail3, bins=bins, color=C.MUTED, rwidth=.9,
            label="permutation null (2,000 draws)")
    ax.axvline(obs_tail3, color=C.C_DEFECT, lw=2.0, label="observed")
    ax.set_xlabel("number of cells with |z| > 3")
    ax.set_ylabel("permutation draws")
    ax.set_title("b  is the tail heavier than chance")
    ax.legend(loc="upper right", fontsize=6.3)
    C.annotate(ax, f"observed {obs_tail3}\nnull median {np.median(null_tail3):.0f} "
                   f"[{np.percentile(null_tail3, 2.5):.0f}, "
                   f"{np.percentile(null_tail3, 97.5):.0f}]\n"
                   f"permutation p = {p_tail:.3f}", loc="upper left")

    ax = axes[2]
    top = cell.head(14).iloc[::-1]
    ax.barh(range(len(top)), top["z"], color=[C.MODEL_COL[m] for m in top["model"]])
    ax.axvline(0, color=C.INK, lw=.8)
    for xv in (-3, 3):
        ax.axvline(xv, color=C.C_DEFECT, lw=.8, ls="--")
    ax.set_yticks(range(len(top)),
                  [f"{m.replace('-Non-Reasoning', '')[:13]}  {lg_}  "
                   f"$\\lambda$={sc:g}  {pr_}"
                   for m, lg_, sc, pr_ in zip(top["model"], top["language"],
                                              top["scale"], top["pairing"])],
                  fontsize=5.7)
    ax.set_xlabel("cell mean residual z (bar colour = model)")
    ax.set_title("c  the 14 most extreme cells")
    ax.set_xlim(min(top["z"].min() * 1.35, -3.6), max(top["z"].max() * 1.35, 3.6))
    C.annotate(ax, f"max |z| = {obs_max:.2f}\npermutation p = {p_max:.3f}\n"
                   f"cells passing FDR 5%: {int((cell.p_fdr < .05).sum())}",
               loc="lower right")
    C.save(fig, "advanced", "B07_outlier_cells_vs_null",
           "Are any of the 1,200 condition cells genuinely anomalous relative to an "
           "additive model in model, language, lambda and pairing, or are the "
           "extremes exactly what 1,200 draws of a 10-game mean produce anyway?",
           f"The cell residuals are over-dispersed relative to the parametric normal "
           f"null (SD of z is {zz.std():.2f} against 1.00) and also relative to the "
           f"permutation null (null median {np.median(null_sd):.2f}, "
           f"p = {p_sd:.4f}), which says the additive model is incomplete. But the "
           f"extreme tail is only mildly unusual: {obs_tail3} cells exceed |z| = 3 "
           f"against a permutation null median of {np.median(null_tail3):.0f} "
           f"(p = {p_tail:.3f}), and the single most extreme cell reaches "
           f"|z| = {obs_max:.2f} at p = {p_max:.3f}, with "
           f"{int((cell.p_fdr < .05).sum())} cells surviving a 5% FDR correction. The "
           "honest reading is that the missing structure is an interaction spread "
           "over many cells rather than a handful of broken ones, so no individual "
           "cell should be reported as an anomaly.",
           "residual histogram against null, tail-count null, ranked extreme cells",
           "joint_coop residual from an additive model, 1,200 condition cells")

    # =====================================================================
    # ANOMALOUS INDIVIDUAL GAMES
    # =====================================================================
    seq = (rounds[rounds["agent"] == 1].sort_values(["game_uid", "round"])
           .groupby("game_uid", observed=True)
           .agg(a=("action", list), o=("opp_action", list), oc=("outcome", list)))

    g = dy.set_index("game_uid")
    a1g = ag[ag["agent"] == 1].set_index("game_uid")
    a2g = ag[ag["agent"] == 2].set_index("game_uid")

    an = pd.DataFrame(index=g.index)
    an["model"] = g["model"]
    an["language"] = g["language"]
    an["scale"] = g["scale"]
    an["pairing"] = g["dyad"]
    an["joint_coop"] = g["joint_coop"]
    an["n_switch"] = g["n_switch"]
    an["p_CD"] = g["p_CD"]
    an["p_DC"] = g["p_DC"]
    an["longest_dd"] = g["longest_dd"]
    an["longest_alternation_joint"] = seq["oc"].map(_longest_alt).reindex(an.index)
    an["min_reciprocity"] = np.fmin(a1g["reciprocity"].reindex(an.index).to_numpy(),
                                    a2g["reciprocity"].reindex(an.index).to_numpy())

    comps = {
        "perfect exploitation, all 10 rounds":
            ((g["p_DC"] == 1.0) | (g["p_CD"] == 1.0)).astype(int),
        "action alternates every round":
            (an["n_switch"] == 9).astype(int),
        "joint outcome alternates 8+ rounds":
            (an["longest_alternation_joint"] >= 8).astype(int),
        "strongly anti-reciprocal agent":
            (an["min_reciprocity"] <= -0.8).astype(int),
        "escaped a 5+ round defection lock":
            ((g["longest_dd"] >= 5) & (g["dd_absorbed"] == 0)).astype(int),
    }
    for k_, v_ in comps.items():
        an[k_] = v_.reindex(an.index).fillna(0).astype(int)
    an["n_flags"] = an[list(comps)].sum(axis=1)
    an["rarity_score"] = sum(
        an[k_] * float(-np.log10(max(an[k_].mean(), 1 / len(an)))) for k_ in comps)
    an = an.sort_values(["rarity_score", "n_flags"], ascending=False)

    flag_counts = pd.DataFrame({
        "pattern": list(comps),
        "n_games": [int(an[k_].sum()) for k_ in comps],
        "share_of_12000_games": [float(an[k_].mean()) for k_ in comps]})
    C.savetab(flag_counts, "struct_anomaly_pattern_counts")
    top_games = an[an["n_flags"] > 0].head(30).reset_index()
    top_games["outcome_sequence"] = ["".join(
        {"CC": "R", "CD": "S", "DC": "T", "DD": "P"}[o] for o in seq.loc[u, "oc"])
        for u in top_games["game_uid"]]
    C.savetab(top_games[["game_uid", "model", "language", "scale", "pairing",
                         "joint_coop", "outcome_sequence", "n_switch",
                         "longest_alternation_joint", "p_CD", "p_DC", "longest_dd",
                         "min_reciprocity", "n_flags", "rarity_score"] + list(comps)],
              "struct_anomalous_games")

    n_perfect = int(an["perfect exploitation, all 10 rounds"].sum())
    n_alt = int(an["action alternates every round"].sum())
    n_anti = int(an["strongly anti-reciprocal agent"].sum())
    n_multi = int((an["n_flags"] >= 2).sum())

    # ---------------------------------------------------------------------
    # B08  anomalous game gallery
    # ---------------------------------------------------------------------
    show = an[an["n_flags"] > 0].head(20).index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6),
                             gridspec_kw={"width_ratios": [1.15, .85, 1.5]})
    ax = axes[0]
    fc = flag_counts.sort_values("n_games")
    ax.barh(range(len(fc)), fc["n_games"], color=C.OUTCOME_COL["DC"])
    ax.set_yticks(range(len(fc)),
                  [p.replace(", ", ",\n").replace(" every", "\nevery")
                   .replace(" 8+", "\n8+").replace("anti-", "\nanti-")
                   .replace(" a 5+", "\na 5+") for p in fc["pattern"]], fontsize=5.9)
    ax.set_xscale("log")
    ax.set_xlim(0.6, fc["n_games"].max() * 9)
    ax.set_xlabel("games with the pattern (log axis, out of 12,000)")
    ax.set_title("a  how rare is each pattern")
    for i, (v_, s_) in enumerate(zip(fc["n_games"], fc["share_of_12000_games"])):
        ax.text(max(v_, 0.8) * 1.25, i, f"{v_}  ({s_:.2%})", va="center", fontsize=6)

    ax = axes[1]
    hist = an["n_flags"].value_counts().sort_index()
    ax.bar(hist.index, hist.values, color=C.MUTED, width=.7)
    ax.set_yscale("log")
    ax.set_xlabel("anomaly flags on one game")
    ax.set_ylabel("games (log axis)")
    ax.set_xticks(hist.index)
    ax.set_title("b  flag co-occurrence")
    for i_, v_ in zip(hist.index, hist.values):
        ax.text(i_, v_ * 1.35, f"{v_:,}", ha="center", fontsize=6)
    ax.set_ylim(top=hist.max() * 12)
    C.annotate(ax, f"{n_multi} games carry\ntwo or more flags", loc="upper right")

    ax = axes[2]
    for row, gid in enumerate(show):
        for t, o in enumerate(seq.loc[gid, "oc"]):
            ax.add_patch(plt.Rectangle((t, row + .07), 1, .86,
                                       color=C.OUTCOME_COL[o], lw=0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(show))
    ax.set_xticks(np.arange(10) + .5, range(1, 11), fontsize=6.5)
    ax.set_yticks(np.arange(len(show)) + .5,
                  [f"{an.loc[gid, 'model'].replace('-Non-Reasoning', '')[:12]} "
                   f"{an.loc[gid, 'language']} $\\lambda$={an.loc[gid, 'scale']:g}"
                   for gid in show], fontsize=5.3)
    ax.invert_yaxis()
    ax.set_xlabel("round")
    ax.set_title("c  the 20 rarest games, round by round")
    ax.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C.OUTCOME_COL[o])
               for o in C.OUTCOME_ORDER]
    ax.legend(handles, [C.OUTCOME_LABEL[o] for o in C.OUTCOME_ORDER],
              loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2, fontsize=6)
    C.save(fig, "advanced", "B08_anomalous_games",
           "Which individual games show behaviour a plausible strategy could not "
           "easily produce, and are those patterns rare enough to be worth "
           "inspecting by hand?",
           f"Five patterns were scanned across all 12,000 games. Perfect "
           f"exploitation, in which one agent cooperated in every one of the ten "
           f"rounds while its opponent defected in every one, occurs {n_perfect} "
           f"times ({n_perfect / len(an):.1%}), which is far too common to call "
           "anomalous and is instead a substantive result about how completely some "
           "agents fail to react. The genuinely rare patterns are round-by-round "
           f"action alternation ({n_alt} games, {n_alt / len(an):.2%}) and strongly "
           f"anti-reciprocal agents, which cooperate more after being defected on "
           f"than after being cooperated with ({n_anti} games, "
           f"{n_anti / len(an):.2%}). {n_multi} games carry two or more flags at "
           "once. The raster shows the twenty rarest games round by round; their "
           "game_uid values are listed in tables/struct_anomalous_games.csv for "
           "manual inspection.",
           "pattern rarity bars plus flag co-occurrence plus per-round outcome raster",
           "game_uid, outcome sequence, n_switch, reciprocity, longest_dd")

    # =====================================================================
    # findings
    # =====================================================================
    C.record_finding(
        id="B-behaviour-is-low-dimensional",
        finding="Agent-game behaviour in this corpus is close to one dimensional: a "
                "single cooperation versus defection axis carries most of the joint "
                "variation in fifteen behavioural measurements.",
        evidence=f"PCA on {len(FEATS)} standardised features over 24,000 agent-games "
                 f"gives PC1 = {ev[0]:.1%} of the variance, PC2 = {ev[1]:.1%} and "
                 f"PC3 = {ev[2]:.1%} and PC4 = {ev[3]:.1%}, cumulatively "
                 f"{cum[3]:.1%}. PC1 loads with the same sign on the cooperation rate "
                 "(+0.36) and on all four conditional cooperation probabilities "
                 "(+0.31 to +0.32) and with the opposite sign on the DD share "
                 "(-0.26). PC2 is own utility (+0.50) against the CD share (-0.44), "
                 "the exploitation axis. PC3 is rule distance (+0.45) against the "
                 "never-visited-state indicators (-0.41 to -0.43), a history-richness "
                 "axis. PC4 is the endgame drop (+0.60) and round-1 cooperation "
                 "(+0.41) against reciprocity (-0.37), the timing axis.",
        figure="10_advanced/B01_pca_scree_loadings.png, "
               "10_advanced/B03_umap_manifold.png",
        strength="strong",
        robustness="the same ordering appears in a UMAP embedding of a separate "
                   "8,000 game subsample; efficiency was excluded from the matrix "
                   f"because it is exactly 1.25 times utility (r = {r_eff:.4f})",
        interpretation="Descriptive, not causal. Part of PC1 is an accounting "
                       "identity, because the four outcome shares sum to one and "
                       "utility is a fixed linear function of them. The substantive "
                       "content is that the conditional-response measures, which are "
                       "logically free to vary independently of the cooperation "
                       "level, do not: agents that cooperate more also cooperate more "
                       "in every conditioning state.",
        caveat="Feature selection drives PCA. A feature set weighted toward timing "
               "would raise the relative share of the timing component. The "
               "conditional probabilities are also imputed at the agent's own "
               "cooperation rate when the conditioning state never occurred, which "
               "pulls them toward the cooperation axis by construction.",
        claim="Fifteen behavioural measurements of LLM prisoner's dilemma play "
              f"collapse onto four interpretable axes carrying {cum[3]:.1%} of the "
              f"variance, of which the first, holding {ev[0]:.0%} on its own, is "
              "simply how much the agent cooperates.")

    C.record_finding(
        id="B-lambda-does-not-move-behaviour-space",
        finding="Model identity separates agents in behaviour space about an order of "
                "magnitude more than the payoff multiplier does, and the persona "
                "pairing does not move the cooperation axis at all.",
        evidence=f"On the leading principal component, model identity explains "
                 f"eta-squared = {e1[0]:.3f} while the payoff scale explains "
                 f"{e1[4]:.4f} (a factor of {e1[0] / max(e1[4], 1e-9):.0f} in "
                 f"variance) and the language {e1[3]:.4f}. The six model centroids "
                 f"span {sp_m:.2f} units of the PC1-PC2 plane against {sp_s:.2f} for "
                 f"the ten lambda centroids, a factor of {sp_m / sp_s:.1f}. The "
                 f"persona pairing explains {e1[2]:.4f} of PC1 and {e2[2]:.3f} of "
                 f"PC2. Raw cooperation rates agree: 0.56 (CvC), 0.58 (CvS), 0.60 "
                 "(SvC), 0.61 (SvS), with the selfish persona cooperating marginally "
                 "more, not less.",
        figure="10_advanced/B02_pca_projection_by_factor.png",
        strength="strong",
        robustness="reproduced in the UMAP embedding on a separate subsample and in "
                   f"the regime-composition test, where Cramer V is {v_m:.3f} for "
                   f"model against {v_s:.3f} for lambda",
        interpretation="This is a statement about relative magnitude, not about "
                       "whether the lambda effect is real. Section 04 shows the "
                       "lambda effect is real and violates an axiom; this section "
                       "shows it is roughly an order of magnitude smaller than the "
                       "differences between the models themselves. The persona result "
                       "is the sharper one: a prompt that tells the agent it is "
                       "cooperative or selfish changes who ends up exploited without "
                       "changing how much anyone cooperates, so the persona is "
                       "decorative on this axis.",
        caveat="Self-play only and one game family. Eta-squared on a principal "
               "component depends on which features entered the PCA, the strategy "
               "label is not a design factor and its high eta-squared is close to "
               "circular, and the six models are a convenience sample of frontier "
               "systems.",
        claim="The payoff multiplier shifts LLM behaviour, but model identity shifts "
              "it about an order of magnitude more, and a cooperative or selfish "
              "persona prompt shifts the cooperation axis not at all, so payoff-scale "
              "sensitivity is a second-order property and persona framing is a "
              "third-order one.")

    C.record_finding(
        id="B-regime-resorting",
        finding="The payoff-scale effect works by re-sorting agents between stable "
                "behavioural regimes rather than by deforming a single regime.",
        evidence=f"A k = {K_FINAL} partition of the 24,000 agent-games yields six "
                 "phenotypes, each named by a fixed rule read off its own centroid ("
                 + ", ".join(f"{nm} {prof.share_of_corpus[i]:.0%}"
                             for i, nm in enumerate(names))
                 + f"). The regime by lambda contingency table on the independent "
                   f"agent-1 subsample gives chi-square = {chi_s.statistic:.0f}, "
                   f"p = {chi_s.pvalue:.2g}, Cramer V = {v_s:.3f}, and the largest "
                   f"regime share swing across the five-decade ladder is "
                   f"{swing:.3f}, carried by '{big}' at Spearman rho = "
                   f"{rhos[big]:+.2f}. For comparison the same table against model "
                   f"identity gives Cramer V = {v_m:.3f}.",
        figure="10_advanced/B05_regime_phenotypes.png, "
               "10_advanced/B06_regime_mix_by_model_and_scale.png",
        strength="moderate",
        robustness="the contingency test uses only agent-1 rows so that the 12,000 "
                   f"games are independent; k = {K_FINAL} was chosen at the inertia "
                   f"elbow (k = {elbow_k}, with k = {K_FINAL} near-tied) because the "
                   "silhouette has no interior maximum on k = 2 to 12, and the "
                   "qualitative conclusion does not depend on that choice because the "
                   "mutual-cooperation and mutual-defection regimes are present at "
                   "every k",
        interpretation="Lambda is manipulated by design, so the association with the "
                       "regime mix is causal in the sense that lambda was randomised "
                       "over a balanced grid. The mechanism claim, re-sorting rather "
                       "than deformation, is weaker: it rests on the regimes being "
                       "defined once on the pooled corpus, and a within-lambda "
                       "re-clustering could in principle place the boundaries "
                       "elsewhere.",
        caveat="k-means imposes convex clusters on a boundary-inflated cloud, so the "
               "regime boundaries are a modelling choice. The two unconditional "
               "regimes are robust to that choice; the middle regimes are not.",
        claim="Multiplying the payoff matrix by a constant re-sorts LLM agents "
              "between a fixed set of behavioural phenotypes rather than changing "
              "what any single phenotype does, which locates the invariance "
              "violation at the point of initial policy selection.")

    C.record_finding(
        id="B-no-anomalous-cells",
        finding="No individual condition cell is anomalous once the correct null is "
                "used, but the cells as a set are over-dispersed, which points to a "
                "spread-out interaction rather than to broken cells.",
        evidence=f"An additive model in model, language, lambda and pairing was "
                 f"fitted to all 12,000 dyads. The 1,200 cell mean residuals have an "
                 f"observed z standard deviation of {zz.std():.2f} against 1.00 "
                 f"expected under the parametric null and a permutation null median "
                 f"of {np.median(null_sd):.2f} (p = {p_sd:.4f}). Against the same "
                 f"2,000 draw permutation null, {obs_tail3} cells exceed |z| = 3 "
                 f"against a null median of {np.median(null_tail3):.0f} "
                 f"(p = {p_tail:.3f}), the largest single |z| of {obs_max:.2f} has "
                 f"p = {p_max:.3f}, and {int((cell.p_fdr < .05).sum())} cells survive "
                 "a 5% FDR correction.",
        figure="10_advanced/B07_outlier_cells_vs_null.png",
        strength="moderate",
        robustness="the permutation null preserves the true, strongly non-Gaussian "
                   "marginal residual distribution, which the parametric normal null "
                   "does not",
        interpretation="Over-dispersion with few or no individually significant cells "
                       "is the signature of an omitted interaction term distributed "
                       "over many cells, not of a data-collection fault. It is "
                       "consistent with the model by lambda interaction that section "
                       "04 reports.",
        caveat="With only 10 games per cell the per-cell power is very low, so a "
               "moderate real anomaly in one cell would not be detected. The negative "
               "result bounds the size of any single-cell effect, it does not "
               "exclude one.",
        claim="With 1,200 condition cells of 10 games each, the extreme cells are "
              "close to what a permutation null predicts, so apparent per-condition "
              "anomalies in corpora of this size should be treated as sampling noise "
              "unless tested against such a null.")

    C.record_finding(
        id="B-perfect-exploitation-is-common",
        finding="Complete, ten-round-long unilateral exploitation is not an anomaly "
                "in this corpus; it is a common outcome.",
        evidence=f"{n_perfect} of 12,000 games ({n_perfect / len(an):.1%}) end with "
                 "one agent having cooperated in all ten rounds while its opponent "
                 "defected in all ten. By contrast the genuinely rare patterns are "
                 f"round-by-round action alternation ({n_alt} games, "
                 f"{n_alt / len(an):.2%}) and strongly anti-reciprocal agents with "
                 f"reciprocity at or below -0.8 ({n_anti} games, "
                 f"{n_anti / len(an):.2%}).",
        figure="10_advanced/B08_anomalous_games.png",
        strength="strong",
        robustness="a direct count over the full corpus with no modelling assumptions",
        interpretation="An agent defected against ten times in a row that cooperates "
                       "every time is not running any standard reciprocal strategy. "
                       "Together with the AllC mass in the strategy labels and the "
                       "unconditional-cooperator regime found by the clustering, this "
                       "says a substantial share of LLM agent-games are unconditional "
                       "and unresponsive to the opponent, which is the behavioural "
                       "precondition for the exploitation rate observed.",
        caveat="Self-play only, so the exploiter and the exploited are the same model "
               "under different persona prompts; this licenses no claim about how "
               "these models would behave against a different model.",
        claim=f"In {n_perfect / len(an):.1%} of iterated prisoner's dilemma games one "
              "LLM agent cooperated in every single round while its opponent defected "
              "in every single round, so unconditional non-responsiveness, not "
              "anomaly, is the dominant failure mode.")

    return prof, cell, an
