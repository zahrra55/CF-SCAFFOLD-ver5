"""
CF-SCAFFOLD — interactive companion to the thesis.

A three-page application:

    1. The problem   — why federated training fails when clients lack classes
    2. The results   — what each correction achieves, with the measured data
    3. The descriptor — compute C = Ft/Fb on a slice and inspect every step

Design rule followed throughout: the application displays measurements that
were produced on Kaggle and stored as CSV files. It does not train anything.
A single federated run takes about 130 seconds on a GPU and the full
comparison takes roughly 45 minutes, so live training in a browser is not
possible. The one computation performed live is the fractal descriptor, which
takes a few seconds and needs no GPU.

Run locally:   streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

from lib.descriptor import tumor_to_brain_ratio, DESCRIPTOR_AUC, C_RANGE

# ---------------------------------------------------------------------------
# Page configuration and a restrained visual style
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CF-SCAFFOLD", layout="wide")

NAVY, GOLD, STEEL, GREY, RED = "#0a2540", "#c5a572", "#2e6b8f", "#9aa0a6", "#c0392b"

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1150px;}
  h1, h2, h3 {color: #0a2540;}
  .stMetric {background: #f7f8fa; padding: 0.7rem; border-radius: 4px;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading. Cached so the files are read once per session.
# ---------------------------------------------------------------------------
@st.cache_data
def load(name):
    return pd.read_csv(f"data/{name}.csv")


METHOD_LABELS = {
    "FedAvg": "FedAvg",
    "FedRS_alpha0.5": "FedRS (α = 0.5)",
    "FedRS_alpha0.25": "FedRS (α = 0.25)",
    "FedLC": "Logit calibration",
    "Masking_alpha0": "Absent-class masking (α = 0)",
}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("CF-SCAFFOLD")
st.sidebar.caption(
    "Brain tumor MRI classification and federated learning under label skew"
)
page = st.sidebar.radio(
    "Section",
    ["The problem", "The results", "The descriptor"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Zahraa Ibrahim Khaleel**  \n"
    "Supervisor: Prof. Dr. Mohammed Sahib Mehdi  \n"
    "Department of Computer Science  \n"
    "College of Science, Al-Nahrain University"
)
st.sidebar.markdown("---")
st.sidebar.info(
    "Results shown here were measured on Kaggle and stored. The application "
    "displays them; it does not retrain. The fractal descriptor on page 3 is "
    "computed live."
)


# ===========================================================================
# PAGE 1 — the problem
# ===========================================================================
if page == "The problem":
    st.title("Why federated training fails here")

    st.markdown("""
Several hospitals want to train one diagnostic model without exchanging patient
images. Each hospital keeps its data and shares only model updates. The
difficulty is that hospitals do not see the same tumor types: a centre
specialising in pituitary surgery may accumulate hundreds of those cases and
none of another type.

Use the control below to see what that does to a federation of 18 simulated
sites.
    """)

    st.subheader("Simulating the partition")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        alpha_dir = st.select_slider(
            "Partition severity (Dirichlet concentration)",
            options=[0.1, 0.3, 0.5, 1.0, 3.0],
            value=0.3,
            help="A small value concentrates each tumor type on a few sites. "
                 "A large value spreads them evenly.",
        )
        seed = st.number_input("Partition draw (seed)", 0, 99, 7, step=1)

    # Draw a Dirichlet partition exactly as the experiments do
    rng = np.random.default_rng(int(seed))
    n_clients, n_classes = 18, 3
    composition = np.zeros((n_clients, n_classes))
    for c in range(n_classes):
        composition[:, c] = rng.dirichlet([alpha_dir] * n_clients)
    # a site holds a class if it received a non-negligible share of it
    holds = composition > (0.5 / n_clients)
    n_absent = n_classes - holds.sum(axis=1)
    active = holds.sum(axis=1) > 0
    n_absent = n_absent[active]
    mean_absence = float(np.mean(n_absent) / n_classes)

    with col_b:
        m1, m2, m3 = st.columns(3)
        m1.metric("Active sites", int(active.sum()))
        m2.metric("Sites missing a tumor type",
                  f"{int((n_absent > 0).sum())} of {int(active.sum())}")
        m3.metric("Mean local absence", f"{mean_absence:.2f}",
                  help="Average fraction of the three tumor types that a site "
                       "never sees during training.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.1))
    counts = [int((n_absent == k).sum()) for k in range(n_classes)]
    ax1.bar(["0", "1", "2"], counts, color=["#27ae60", GOLD, RED], width=0.6)
    for i, v in enumerate(counts):
        ax1.text(i, v + 0.15, str(v), ha="center", fontweight="bold", fontsize=9)
    ax1.set_xlabel("tumor types absent from the site")
    ax1.set_ylabel("number of sites")
    ax1.set_title("How many types each site is missing", fontsize=10)

    shown = composition[active] / composition[active].sum(axis=1, keepdims=True)
    im = ax2.imshow(shown.T, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["Meningioma", "Glioma", "Pituitary"], fontsize=8)
    ax2.set_xlabel("site")
    ax2.set_title("What each site holds (pale = absent)", fontsize=10)
    fig.colorbar(im, ax=ax2, fraction=0.04)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("What goes wrong")
    st.markdown(r"""
When a site holds no examples of a class, that class is never the correct
answer in its data. In the cross-entropy loss the derivative with respect to
the score of class $c$ is
""")
    st.latex(r"\frac{\partial \ell}{\partial z_c} = p_c - \mathbb{1}[c = y]")
    st.markdown(r"""
For an absent class the second term is always zero, so the derivative is
$p_c > 0$ at **every** training step. The parameters for that class are pushed
in one direction throughout local training, with nothing pulling back.

The server then averages models from sites that have each pushed a *different*
set of classes. The combined classifier is distorted, and how badly depends on
which sites happened to be missing which classes, that is, on the partition
draw. That is the source of both the loss of accuracy and the instability
between runs shown on the next page.
    """)

    st.info(
        "The correction studied in this work removes that term from the local "
        "objective, so the parameters of an absent class are returned to the "
        "server exactly as they were received."
    )


# ===========================================================================
# PAGE 2 — the results
# ===========================================================================
elif page == "The results":
    st.title("What each correction achieves")

    main = load("figshare_main_comparison")
    brisc = load("brisc_external_validation")
    central = load("centralized_bound")
    sweep = load("alpha_sweep")
    resnet = load("resnet_backbone")

    st.markdown("""
All figures below were measured over 20 independent partitions of the Figshare
dataset, evaluated at the level of the patient rather than the slice. The
metric is balanced accuracy, the average of the recall of the three tumor
types; chance is 0.333.
    """)

    # --- main comparison -----------------------------------------------
    st.subheader("Primary comparison")

    methods = ["FedAvg", "FedRS_alpha0.5", "FedLC", "Masking_alpha0"]
    summary = pd.DataFrame({
        "Method": [METHOD_LABELS[m] for m in methods],
        "Balanced accuracy": [f"{main[m].mean():.3f} ± {main[m].std():.3f}"
                              for m in methods],
        "Worst partition": [f"{main[m].min():.3f}" for m in methods],
    })
    summary.loc[len(summary)] = [
        "Centralised (pooled data)",
        f"{central['balanced_acc'].mean():.3f} ± {central['balanced_acc'].std():.3f}",
        f"{central['balanced_acc'].min():.3f}",
    ]
    st.dataframe(summary, hide_index=True, use_container_width=True)

    gap = central["balanced_acc"].mean() - main["FedAvg"].mean()
    closed = (main["Masking_alpha0"].mean() - main["FedAvg"].mean()) / gap * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Gain over FedAvg",
              f"+{(main['Masking_alpha0'].mean() - main['FedAvg'].mean())*100:.1f} pts")
    c2.metric("Variability across partitions",
              f"{main['Masking_alpha0'].std():.3f}",
              delta=f"{main['Masking_alpha0'].std() - main['FedAvg'].std():.3f}",
              delta_color="inverse",
              help="Standard deviation across the 20 partitions; lower is better.")
    c3.metric("Gap to centralised training closed", f"{closed:.0f}%")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    means = [main[m].mean() for m in methods]
    stds = [main[m].std() for m in methods]
    ax.bar(range(4), means, yerr=stds, capsize=4,
           color=[GREY, STEEL, STEEL, GOLD], edgecolor="white")
    ax.axhline(central["balanced_acc"].mean(), ls="--", color=NAVY, lw=1.2)
    ax.text(3.45, central["balanced_acc"].mean() + 0.006, "centralised",
            fontsize=8, color=NAVY, ha="right")
    ax.set_xticks(range(4))
    ax.set_xticklabels([METHOD_LABELS[m].replace(" (", "\n(") for m in methods],
                       fontsize=8)
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0.45, 0.76)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.008, f"{m:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("How these differences were tested"):
        st.markdown("""
Each method is run on the same 20 partitions, so the comparison is paired and a
paired test applies. Differences are assessed with the Wilcoxon signed-rank
test, which makes no assumption that the differences are normally distributed,
and the resulting p-values are adjusted with the Holm procedure because three
comparisons are made against the same reference.
        """)
        rows = []
        base = main["Masking_alpha0"]
        for m in ["FedAvg", "FedRS_alpha0.5", "FedLC"]:
            d = base - main[m]
            p = stats.wilcoxon(base, main[m]).pvalue
            rows.append({
                "Compared with": METHOD_LABELS[m],
                "Difference": f"{d.mean():+.3f}",
                "Wilcoxon p (unadjusted)": f"{p:.4f}",
                "Better on": f"{int((d > 0).sum())} of 20 partitions",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)
        st.caption(
            "Holm-adjusted values reported in the manuscript are 0.002, 0.002 "
            "and 0.123. The two significant results remain significant and the "
            "third remains non-significant: masking and logit calibration "
            "cannot be told apart."
        )

    st.markdown("---")

    # --- honest reading -------------------------------------------------
    st.subheader("Reading the table honestly")
    st.markdown("""
Two of the four objectives leave the absent-class term in place and two remove
it. The two that remove it gain six to seven points; the two that leave it gain
almost nothing. **Masking and logit calibration cannot be distinguished
statistically.** That is the finding: what matters is that the term is removed,
not the particular way of removing it. Masking is preferred here only because it
introduces no parameter that would have to be tuned.
    """)

    st.markdown("---")

    # --- the mechanism --------------------------------------------------
    st.subheader("The benefit depends on how much is missing")
    st.markdown("""
If the explanation on the previous page is right, the correction should help
most where sites are missing most, and do nothing where sites hold everything.
That is a prediction, and it can be checked.
    """)

    show = sweep.rename(columns={
        "alpha_dir": "Dirichlet concentration",
        "mean_local_absence": "Mean local absence",
        "alpha_mask_0": "Masking (α = 0)",
        "alpha_mask_0.5": "α = 0.5",
        "alpha_mask_1.0": "No correction",
        "margin_pp": "Margin (points)",
        "wilcoxon_p": "p",
    })
    st.dataframe(show, hide_index=True, use_container_width=True)

    fig, ax = plt.subplots(figsize=(6, 2.9))
    ax.plot(sweep["mean_local_absence"], sweep["margin_pp"],
            marker="s", ms=8, lw=2.2, color=NAVY)
    ax.axhline(0, ls="--", color=GREY, lw=0.9)
    ax.set_xlabel("mean fraction of tumor types absent per site")
    ax.set_ylabel("advantage of masking (points)")
    for x, y in zip(sweep["mean_local_absence"], sweep["margin_pp"]):
        ax.annotate(f"+{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=NAVY)
    ax.set_ylim(-2, 21)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.success(
        "The advantage falls from 17.5 points where sites are missing half the "
        "label space to nothing where they hold almost all of it. The "
        "correction switches off exactly where there is nothing to correct, "
        "which is what the explanation predicts and what a general "
        "regularisation effect would not do."
    )

    st.markdown("---")

    # --- robustness -----------------------------------------------------
    st.subheader("Does it hold elsewhere?")
    t1, t2 = st.tabs(["A second dataset", "A larger network"])

    with t1:
        bm = ["FedAvg", "FedRS_alpha0.5", "FedRS_alpha0.25", "Masking_alpha0"]
        st.dataframe(pd.DataFrame({
            "Method": [METHOD_LABELS[m] for m in bm],
            "Balanced accuracy": [f"{brisc[m].mean():.3f} ± {brisc[m].std():.3f}"
                                  for m in bm],
        }), hide_index=True, use_container_width=True)
        st.markdown(f"""
On BRISC the advantage over FedRS is
**{(brisc['Masking_alpha0'].mean() - brisc['FedRS_alpha0.5'].mean())*100:.1f} points**,
close to the {(main['Masking_alpha0'].mean() - main['FedRS_alpha0.5'].mean())*100:.1f}
measured on Figshare.

BRISC is drawn in part from the same sources as the primary dataset, so this is
a check that the result is not an artefact of one particular collection rather
than validation on an unseen population. It also lacks patient identifiers, so
its sites are formed from slices; its numbers compare methods against each
other but are not comparable in absolute terms to the patient-level figures
above.
        """)

    with t2:
        rm = ["FedAvg", "FedRS_alpha0.5", "FedLC", "Masking_alpha0"]
        st.dataframe(pd.DataFrame({
            "Method": [METHOD_LABELS[m] for m in rm],
            "Small network": [f"{main[m].mean():.3f}" for m in rm],
            "Residual network": [f"{resnet[m].mean():.3f}" for m in rm],
        }), hide_index=True, use_container_width=True)
        st.markdown("""
The correction acts on the final layer, so it has less to act on when a larger
feature extractor carries more of the classification. The gain halves, from 6.5
points to 2.9. The larger network is also worse in absolute terms: with about
2400 training slices spread over 17 sites there is too little data per site to
support it. This bounds where the result should be expected to apply, and it is
reported rather than omitted.
        """)


# ===========================================================================
# PAGE 3 — the descriptor
# ===========================================================================
elif page == "The descriptor":
    st.title("The tumor-to-brain fractal ratio")

    st.markdown(r"""
The descriptor is a ratio between two fractal dimensions measured on the same
slice: one for the tumor, one for the brain tissue with the tumor removed.
""")
    st.latex(r"C \;=\; \frac{F_t}{F_b}")
    st.markdown(r"""
A fractal dimension on its own is not comparable between patients, because it
depends on the scanner, the sequence and the noise. The surrounding brain in
the same slice was acquired under exactly those conditions, so dividing by it
cancels what they have in common. What remains expresses how irregular the
tumor is **relative to the brain that hosts it**.

Supply a slice and its tumor mask below, or use the generated example, and each
step of the computation is shown.
    """)

    source = st.radio("Input", ["Generated example", "Upload my own"],
                      horizontal=True)

    gray = mask = None

    if source == "Generated example":
        c1, c2 = st.columns([1, 2])
        with c1:
            roughness = st.slider("Tumor irregularity", 5, 60, 35, step=5,
                                  help="Higher values give a more "
                                       "heterogeneous tumor interior.")
            radius = st.slider("Tumor radius (pixels)", 8, 30, 18)
            ex_seed = st.number_input("Example seed", 0, 99, 3, step=1)

        rng = np.random.default_rng(int(ex_seed))
        yy, xx = np.mgrid[0:128, 0:128]
        gray = np.zeros((128, 128))
        brain = ((xx - 64) ** 2 + (yy - 64) ** 2) < 55 ** 2
        gray[brain] = 120 + rng.normal(0, 15, brain.sum())
        tum = ((xx - 70) ** 2 + (yy - 60) ** 2) < radius ** 2
        gray[tum] = 200 + rng.normal(0, roughness, tum.sum())
        gray = np.clip(gray, 0, 255)
        mask = tum.astype(np.uint8)

        with c2:
            st.caption(
                "This is a synthetic slice used to demonstrate the computation. "
                "It is not medical data and no diagnostic meaning attaches to "
                "the value obtained from it."
            )
    else:
        c1, c2 = st.columns(2)
        f_img = c1.file_uploader("Slice image", type=["png", "jpg", "jpeg"])
        f_msk = c2.file_uploader("Tumor mask", type=["png", "jpg", "jpeg"])
        if f_img and f_msk:
            from PIL import Image
            img = Image.open(f_img).convert("L").resize((128, 128))
            msk = Image.open(f_msk).convert("L").resize((128, 128),
                                                        Image.NEAREST)
            gray = np.asarray(img).astype(float)
            mask = (np.asarray(msk) > 127).astype(np.uint8)
        else:
            st.info("Upload both a slice and its tumor mask to continue. "
                    "The mask must mark the tumor in white on black.")

    if gray is not None:
        result = tumor_to_brain_ratio(gray, mask)

        if not result["valid"]:
            st.warning(result["message"])
        else:
            st.markdown("---")
            st.subheader("Step 1 — separate the regions")
            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            for ax, (data, title) in zip(axes, [
                (gray, "Slice"),
                (result["tumor"], "Tumor"),
                (result["brain"], "Brain without tumor"),
            ]):
                ax.imshow(data, cmap="gray")
                ax.set_title(title, fontsize=9)
                ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                f"Tumor: {int(result['tumor'].sum())} pixels. "
                f"Tumor-free brain: {int(result['brain'].sum())} pixels. "
                "The background is excluded first, because empty boxes would "
                "otherwise bias the count."
            )

            st.subheader("Step 2 — measure each region")
            fig, ax = plt.subplots(figsize=(5.5, 3))
            for key, colour, label in [("tumor", GOLD, "Tumor"),
                                       ("brain", STEEL, "Brain")]:
                sizes, counts = result["curves"][key]
                ax.plot(np.log(1.0 / sizes), np.log(counts), "o-",
                        color=colour, label=label, ms=5)
            ax.set_xlabel("log (1 / box size)")
            ax.set_ylabel("log (box count)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "The dimension is the slope of each line. A steeper slope means "
                "detail continues to appear as the boxes get smaller, which is "
                "what structural irregularity looks like when measured."
            )

            st.subheader("Step 3 — the ratio")
            m1, m2, m3 = st.columns(3)
            m1.metric("Ft (tumor)", f"{result['Ft']:.4f}")
            m2.metric("Fb (brain)", f"{result['Fb']:.4f}")
            m3.metric("C = Ft / Fb", f"{result['C']:.4f}")

            lo, hi = C_RANGE
            if lo <= result["C"] <= hi:
                st.success(
                    f"This value lies inside the range {lo}–{hi} observed "
                    "across the 2740 slices used in the thesis."
                )
            else:
                st.warning(
                    f"This value lies outside the range {lo}–{hi} observed in "
                    "the thesis data, which usually means the regions are "
                    "unusual in size or texture."
                )

            st.markdown("---")
            st.subheader("How much can one number tell you?")
            st.markdown("""
Not as much for every tumor type. Measured on the thesis data, C used on its
own separates the three types very unevenly:
            """)
            fig, ax = plt.subplots(figsize=(5.5, 2))
            names = list(DESCRIPTOR_AUC)
            vals = [DESCRIPTOR_AUC[n] for n in names]
            ax.barh(range(3), vals,
                    color=["#27ae60", GOLD, RED], height=0.55)
            ax.axvline(0.5, ls=":", color=GREY)
            ax.set_yticks(range(3))
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlim(0.4, 1.0)
            ax.set_xlabel("one-vs-rest AUC of C alone")
            ax.invert_yaxis()
            for i, v in enumerate(vals):
                ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.warning(
                "C identifies pituitary adenomas well and meningiomas barely "
                "better than chance. Meningiomas vary widely among themselves, "
                "some smooth and well circumscribed, others lobulated, so a "
                "single measure of irregularity cannot separate them. In the "
                "thesis C is one of sixteen features for exactly this reason, "
                "and it is not used alone."
            )
