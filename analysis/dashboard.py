"""
Filter Bubble Simulation — Streamlit Dashboard
===============================================

Visualises the results of the research simulation run and lets you
run small exploratory mini-simulations directly in the browser.

HOW TO RUN
----------
From the project root (filter_bubble_sim/):

    streamlit run analysis/dashboard.py

The dashboard will open at http://localhost:8501
Make sure experiments/run_001_summary.json exists (run experiments/full_run.py first).
"""

import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import json
import numpy as np
import streamlit as st
matplotlib.use("Agg")

# ── path bootstrap so imports work from any CWD ─────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.simulation_config import SimulationConfig
from experiments.run_simulation import run_simulation, initialize_simulation


# ── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Filter Bubble Simulation",
    page_icon="🫧",
    layout="wide"
)


# ── GLOBAL STYLES ────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { padding-top: 1rem; }
[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── DATA LOADERS ─────────────────────────────────────────────────────────────

@st.cache_data
def load_summary() -> dict | None:
    """Load the pre-computed run_001_summary.json from disk."""
    path = "experiments/run_001_summary.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_full_results(run_id: str) -> dict | None:
    path = f"experiments/{run_id}_full_results.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_all_summaries() -> dict:
    """
    Load all available run summary JSONs.

    Returns a dict keyed by human-readable run label, containing only
    those runs whose JSON file already exists on disk.
    """
    runs = {
        "Run 001 — Diversity Injection (t=800)":
            "experiments/run_001_summary.json",
        "Run 002 — Early Intervention (t=400)":
            "experiments/run_002_summary.json",
        "Run 003 — Algorithm Friction":
            "experiments/run_003_summary.json",
        "Run 004 — Social Bridge":
            "experiments/run_004_summary.json",
    }
    loaded = {}
    for label, path in runs.items():
        if os.path.exists(path):
            with open(path) as f:
                loaded[label] = json.load(f)
    return loaded


@st.cache_data
def run_mini_simulation(
    n_agents: int,
    n_timesteps: int,
    intervention_type: str,
    seed: int,
) -> dict:
    """
    Run a lightweight simulation and return serialisable metric arrays.

    Returns a plain dict (arrays as lists) so Streamlit can cache it.
    """
    config = SimulationConfig(
        random_seed=seed,
        total_timesteps=n_timesteps,
        burnin_end=max(1, int(n_timesteps * 0.1)),
        intervention_start=int(n_timesteps * 0.8),
        checkpoints=[
            int(n_timesteps * 0.1),
            int(n_timesteps * 0.4),
            int(n_timesteps * 0.8),
            n_timesteps,
        ],
        sequential_llm_checkpoints=[n_timesteps],
        use_llm=False,
        n_agents=n_agents,
        agent_distribution={
            "heavy_imdb":    n_agents // 5,
            "heavy_reddit":  n_agents // 5,
            "heavy_youtube": n_agents // 5,
            "balanced":      n_agents // 5,
            "cross_platform": n_agents - 4 * (n_agents // 5),
        },
        intervention_type=intervention_type,
        new_agent_entry_rate=0,
    )
    state = run_simulation(config)
    store = state.measurement_store

    return {
        "belief_positions": store.belief_positions[:n_agents, :n_timesteps].tolist(),
        "final_positions":  [a.belief_position for a in state.agents[:n_agents]],
        "n_skips":          len(store.skip_log),
        "n_agents":         n_agents,
        "n_timesteps":      n_timesteps,
        "burnin":           max(1, int(n_timesteps * 0.1)),
        "intervention_start": int(n_timesteps * 0.8),
    }


# ── SIDEBAR ──────────────────────────────────────────────────────────────────

st.sidebar.title("🫧 Filter Bubble Sim")
st.sidebar.markdown(
    "Research dashboard for studying algorithmic filter-bubble formation "
    "across IMDB, Reddit and YouTube."
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Overview",
        "📈 Belief Trajectories",
        "🤖 Platform Analysis",
        "🛡️ Interventions",
        "⚠️ Anomalies",
        "📊 Multi-Run Comparison",
        "🔬 Deep Analysis",
    ],
)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":

    st.title("Filter Bubble Simulation — Overview")
    st.markdown(
        "Studying filter bubble formation across **IMDB**, **Reddit**, and **YouTube** "
        "using a 300-agent, 1 000-timestep agent-based model."
    )

    summary = load_summary()

    if summary:
        # ── Top KPI row ──────────────────────────────────────────────────────
        st.subheader("Research Run 001 — Key Results")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Agent Count", summary["final_n_agents"])
        col2.metric(
            "Final Mean Belief",
            f"{summary['final_mean_belief']:.3f}",
            delta=f"{summary['final_mean_belief'] - 5.0:+.3f}",
            help="Started at ~5.0 (neutral)",
        )
        col3.metric("Total Anomalies", summary["total_anomalies"])
        col4.metric(
            "Adapted to Intervention",
            f"{summary['adapted_fraction']:.1%}",
            help="Fraction of agents that learned to skip intervention content",
        )

        st.markdown("---")

        # ── Belief trajectory chart ──────────────────────────────────────────
        st.subheader("Population Mean Belief at Checkpoints")

        checkpoints = summary["checkpoint_means"]
        timesteps = [int(k) for k in checkpoints.keys()]
        means = [checkpoints[k] for k in checkpoints.keys()]

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.fill_between(timesteps, means, alpha=0.12, color="#4C84FF")
        ax.plot(timesteps, means, color="#4C84FF", linewidth=2.5, marker="o",
                markersize=8, markerfacecolor="white", markeredgewidth=2,
                markeredgecolor="#4C84FF", label="Mean belief position")
        ax.axhline(y=5.0, color="#888", linestyle="--", alpha=0.6,
                   linewidth=1.2, label="Neutral start (5.0)")
        ax.axvline(x=800, color="#E84040", linestyle="--", alpha=0.8,
                   linewidth=1.5, label="Intervention start (t=800)")
        ax.set_xlabel("Timestep", fontsize=11)
        ax.set_ylabel("Mean Belief Position", fontsize=11)
        ax.set_title("Population Mean Belief Position Over Time",
                     fontsize=13, fontweight="bold")
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_ylim(0, 10)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")

        # ── Bottom two-column section ────────────────────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("🫧 Bubble Formation")
            st.metric("Early Exposure Bias (t=200)",
                      f"{summary['early_exposure_bias']:.3f}",
                      help="Higher = more homogeneous content consumption")
            st.metric("Late Exposure Bias (t=800)",
                      f"{summary['late_exposure_bias']:.3f}")
            st.metric("Post-Intervention (t=1000)",
                      f"{summary['post_intervention_bias']:.3f}")

            bias_change = summary["late_exposure_bias"] - summary["early_exposure_bias"]
            if bias_change > 0.01:
                st.success(f"Bubbles formed: +{bias_change:.3f} increase in exposure bias")
            elif bias_change < -0.01:
                st.info(f"Exposure bias fell by {abs(bias_change):.3f} — bubbles softened")
            else:
                st.info("No significant bubble formation detected")

        with col_b:
            st.subheader("📺 YouTube RL Learning")
            wv_start = summary["youtube_wv_start"]
            wv_end   = summary["youtube_wv_end"]
            st.metric("W_V Start (valence weight)", f"{wv_start:.3f}")
            st.metric(
                "W_V End",
                f"{wv_end:.3f}",
                delta=f"{wv_end - wv_start:+.3f}",
                help="Positive delta = algorithm learned to amplify emotional content",
            )
            if wv_end > wv_start:
                st.warning("Algorithm learned to amplify emotional content")
            else:
                st.info("Algorithm did not amplify emotional content in this run")

            st.metric("Total Skip Events",
                      f"{summary.get('total_skips', 'N/A'):,}")
            st.metric("Reddit Communities",
                      summary.get("n_communities", "N/A"))

        st.markdown("---")

        # ── Anomaly breakdown ────────────────────────────────────────────────
        st.subheader("Anomaly Breakdown")
        if summary.get("anomaly_types"):
            acols = st.columns(len(summary["anomaly_types"]))
            for i, (atype, count) in enumerate(summary["anomaly_types"].items()):
                acols[i].metric(atype.replace("_", " ").title(), count)
        else:
            st.info("No anomalies detected")

    else:
        st.warning(
            "⚠️ No summary data found. "
            "Run `python -m experiments.full_run` first, then refresh."
        )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — BELIEF TRAJECTORIES
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📈 Belief Trajectories":

    st.title("Belief Position Trajectories")
    st.markdown(
        "Run a mini simulation to visualise how agent belief positions drift over time. "
        "Adjust settings in the sidebar and click **▶ Run Simulation**."
    )

    # Sidebar controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Settings")
    n_agents    = st.sidebar.slider("Number of Agents", 20, 100, 50, step=10)
    n_timesteps = st.sidebar.slider("Timesteps", 50, 300, 100, step=50)
    seed        = st.sidebar.number_input("Random Seed", value=42, step=1)
    intervention = st.sidebar.selectbox(
        "Intervention Type",
        ["diversity_injection", "algorithm_friction", "social_bridge", "null"],
    )
    run_button = st.sidebar.button("▶ Run Simulation", use_container_width=True)

    if run_button:
        with st.spinner(
            f"Running {n_timesteps} timesteps with {n_agents} agents…"
        ):
            result = run_mini_simulation(
                n_agents, n_timesteps, intervention, int(seed)
            )

        bp           = np.array(result["belief_positions"])
        burnin       = result["burnin"]
        intv_start   = result["intervention_start"]
        final_positions = result["final_positions"]

        # ── Two-panel chart ──────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left — individual trajectories
        ax = axes[0]
        n_sample = min(20, n_agents)
        for i in range(n_sample):
            ax.plot(bp[i], alpha=0.35, linewidth=0.9)
        mean_traj = np.mean(bp, axis=0)
        ax.plot(mean_traj, color="black", linewidth=2.5, label="Population mean")
        ax.axhline(y=5.0, color="#888", linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(x=intv_start, color="#E84040", linestyle="--", alpha=0.8,
                   linewidth=1.5, label=f"Intervention (t={intv_start})")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Belief Position")
        ax.set_title("Individual Agent Trajectories")
        ax.set_ylim(0, 10)
        ax.legend(framealpha=0.85)
        ax.grid(True, alpha=0.2, linestyle="--")

        # Right — distribution shift
        ax = axes[1]
        early = bp[:, burnin]
        final = bp[:, n_timesteps - 1]
        ax.hist(early, bins=20, alpha=0.55, color="#4C84FF",
                label=f"Early (t={burnin})", edgecolor="white", linewidth=0.4)
        ax.hist(final, bins=20, alpha=0.55, color="#E84040",
                label=f"Final (t={n_timesteps})", edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Belief Position")
        ax.set_ylabel("Number of Agents")
        ax.set_title("Belief Distribution: Early vs Final")
        ax.legend(framealpha=0.85)
        ax.grid(True, alpha=0.2, linestyle="--")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Summary metrics ──────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Mean",  f"{np.mean(final_positions):.3f}")
        c2.metric("Final Std",   f"{np.std(final_positions):.3f}")
        c3.metric("Mean Drift",  f"{np.mean(final_positions) - 5.0:+.3f}")
        c4.metric("Total Skips", f"{result['n_skips']:,}")

    else:
        st.info(
            "👈 Adjust settings in the sidebar and click **▶ Run Simulation** to begin."
        )

        # Placeholder chart from summary data
        summary = load_summary()
        if summary:
            st.markdown("---")
            st.subheader("Preview: Research Run 001 Checkpoint Means")
            checkpoints = summary["checkpoint_means"]
            timesteps = [int(k) for k in checkpoints.keys()]
            means = [checkpoints[k] for k in checkpoints.keys()]
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(timesteps, means, "o--", color="#4C84FF", linewidth=2)
            ax.axvline(x=800, color="#E84040", linestyle="--", alpha=0.7)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Mean Belief")
            ax.set_title("Research Run 001 — Checkpoint Means (reference)")
            ax.grid(True, alpha=0.2, linestyle="--")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — PLATFORM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🤖 Platform Analysis":

    st.title("Platform Algorithm Analysis")
    st.markdown(
        "Each platform uses a different algorithmic mechanism. "
        "The table below compares their bubble-formation dynamics."
    )

    summary = load_summary()

    if summary:
        # ── YouTube RL weight evolution ──────────────────────────────────────
        st.subheader("YouTube RL Weight Evolution")
        st.markdown(
            "The YouTube contextual-bandit learns which content features maximise "
            "engagement. **W_V rising** means the algorithm learned to push emotionally "
            "charged content — a core driver of rabbit-hole dynamics."
        )

        wv_start = summary["youtube_wv_start"]
        wv_end   = summary["youtube_wv_end"]
        delta    = wv_end - wv_start

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("W_V Start (valence)", f"{wv_start:.4f}")
        c2.metric("W_V End",             f"{wv_end:.4f}", delta=f"{delta:+.4f}")
        c3.metric("Equal-weight baseline", "0.2500")
        c4.metric("W_V Change",            f"{delta:+.4f}")

        # Weight bar chart (values from run output)
        labels      = ["W_V\n(valence)", "W_E\n(engagement)",
                       "W_N\n(novelty)", "W_D\n(distance)"]
        start_vals  = [wv_start, 0.448, 0.0, 0.174]
        end_vals    = [wv_end,   0.493, 0.0, 0.124]

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x - width/2, start_vals, width, label="Start",
               color="#4C84FF", alpha=0.85, edgecolor="white")
        ax.bar(x + width/2, end_vals, width, label="End",
               color="#E84040", alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Weight value")
        ax.set_title("YouTube RL Weights: Start vs End of Run 001",
                     fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.2, axis="y", linestyle="--")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")

    # ── Platform comparison cards ────────────────────────────────────────────
    st.subheader("Platform Bubble Mechanisms")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎬 IMDB")
        st.markdown("""
| Property | Value |
|---|---|
| Mechanism | Taste similarity |
| Network | Dynamic k-NN |
| Bubble type | Soft aesthetic isolation |
| Speed | Slow, gradual |
| Algorithm | Matrix Factorisation (SGD) |
| Algo weight | 0.60 |
        """)

    with col2:
        st.markdown("### 💬 Reddit")
        st.markdown("""
| Property | Value |
|---|---|
| Mechanism | Community absorption |
| Network | Stochastic block model |
| Bubble type | Community-level echo chamber |
| Speed | Medium, event-driven |
| Algorithm | Hot sort + Wilson score |
| Algo weight | 0.50 |
        """)

    with col3:
        st.markdown("### 📺 YouTube")
        st.markdown("""
| Property | Value |
|---|---|
| Mechanism | Algorithmic drift |
| Network | Sparse subscription graph |
| Bubble type | Rabbit-hole radicalisation |
| Speed | Fast, RL-driven |
| Algorithm | Contextual Bandit RL |
| Algo weight | 0.90 |
        """)

    if not summary:
        st.markdown("---")
        st.warning(
            "Run `python -m experiments.full_run` to see live weight data."
        )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — INTERVENTIONS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🛡️ Interventions":

    st.title("Intervention Analysis")
    st.markdown(
        "Four interventions can be applied at `t=800` to test whether exposure bias "
        "can be reduced. Research Run 001 used **diversity_injection**."
    )

    summary = load_summary()

    if summary:
        st.subheader("Research Run 001 — Diversity Injection Results")

        pre    = summary["late_exposure_bias"]
        post   = summary["post_intervention_bias"]
        effect = post - pre

        c1, c2, c3 = st.columns(3)
        c1.metric("Pre-Intervention Bias (t=800)",    f"{pre:.4f}")
        c2.metric(
            "Post-Intervention Bias (t=1000)", f"{post:.4f}",
            delta=f"{effect:+.4f}",
            delta_color="inverse",
        )
        c3.metric("Agents Who Adapted", f"{summary['adapted_fraction']:.1%}",
                  help="Learned to skip intervention content via RL")

        # Bias bar chart
        fig, ax = plt.subplots(figsize=(6, 2.5))
        bar_colors = ["#4C84FF", "#E84040" if effect > 0 else "#2CA060"]
        ax.barh(["Pre (t=800)", "Post (t=1000)"], [pre, post],
                color=bar_colors, edgecolor="white", height=0.45)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Exposure Bias")
        ax.set_title("Exposure Bias Before vs After Intervention", fontweight="bold")
        ax.grid(True, alpha=0.2, axis="x", linestyle="--")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        if effect < -0.02:
            st.success(
                "Intervention **reduced** exposure bias — diversity injection worked."
            )
        elif effect > 0.02:
            st.error(
                "Intervention **increased** exposure bias — backfire detected."
            )
        else:
            st.warning("Intervention had minimal effect on exposure bias.")

        st.markdown("---")

    # ── Four intervention type descriptions ──────────────────────────────────
    st.subheader("Four Intervention Types")

    interventions = {
        "diversity_injection": {
            "icon": "🌈",
            "desc": (
                "Forces ideologically distant content into recommendations. "
                "Tests whether direct exposure to diverse viewpoints breaks bubbles."
            ),
            "risk": "Agents may learn to skip the injected content (backfire).",
        },
        "algorithm_friction": {
            "icon": "🔧",
            "desc": (
                "Reduces the YouTube W_V (valence) weight, slowing emotional amplification. "
                "A regulatory-style intervention acting on the algorithm itself."
            ),
            "risk": "May reduce engagement without changing underlying belief positions.",
        },
        "social_bridge": {
            "icon": "🌉",
            "desc": (
                "Adds cross-community edges in the Reddit social graph. "
                "Tests whether social exposure across ideological boundaries helps."
            ),
            "risk": "Weak ties may not produce sufficient social influence.",
        },
        "null": {
            "icon": "⚪",
            "desc": "No intervention is applied. Serves as the control condition baseline.",
            "risk": "N/A — used to isolate natural bubble formation dynamics.",
        },
    }

    used = (
        summary.get("intervention_type", "diversity_injection")
        if summary else ""
    )

    for itype, meta in interventions.items():
        active = (itype == used)
        with st.expander(
            f"{meta['icon']} `{itype}`"
            + (" ← **used in Run 001**" if active else ""),
            expanded=active,
        ):
            st.markdown(f"**Description:** {meta['desc']}")
            st.markdown(f"**Risk:** {meta['risk']}")

    if not summary:
        st.markdown("---")
        st.warning(
            "Run `python -m experiments.full_run` to see live intervention results."
        )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — ANOMALIES
# ═══════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Anomalies":

    st.title("Anomaly Detection")
    st.markdown(
        "Unexpected agent behaviours are flagged during the simulation for "
        "**Sequential Layering** (LLM) analysis. Anomalies are the most "
        "interesting findings — they reveal what the aggregate statistics hide."
    )

    summary = load_summary()

    if summary:
        total = summary["total_anomalies"]
        st.metric("Total Anomalies Detected", total)

        anomaly_types = summary.get("anomaly_types", {})

        # ── Pie chart ────────────────────────────────────────────────────────
        if anomaly_types:
            fig, ax = plt.subplots(figsize=(5, 4))
            labels = [k.replace("_", " ").title() for k in anomaly_types.keys()]
            counts = list(anomaly_types.values())
            colors = ["#4C84FF", "#E84040", "#F59E0B", "#2CA060"]
            wedges, texts, autotexts = ax.pie(
                counts, labels=labels, autopct="%1.0f%%",
                colors=colors[: len(counts)], startangle=90,
                pctdistance=0.82,
                wedgeprops=dict(edgecolor="white", linewidth=1.5),
            )
            for at in autotexts:
                at.set_fontsize(11)
                at.set_fontweight("bold")
            ax.set_title("Anomaly Type Distribution", fontsize=13, fontweight="bold")
            fig.tight_layout()
            col_pie, _ = st.columns([1, 1])
            with col_pie:
                st.pyplot(fig)
            plt.close()

        st.markdown("---")

        # ── Per-type descriptions ─────────────────────────────────────────────
        st.subheader("Anomaly Type Details")

        descriptions = {
            "unexpected_resilience": {
                "icon": "🛡️",
                "desc": (
                    "Agent maintained content diversity despite high arousal levels. "
                    "Suggests a protective factor — e.g. high critical_thinking or "
                    "open_mind parameters — is counteracting the bubble mechanism."
                ),
                "research_value": "Identifies protective factors against bubble formation.",
            },
            "rapid_radicalization": {
                "icon": "🚀",
                "desc": (
                    "Agent drifted unusually fast. A possible cascade or compounding "
                    "effect: high susceptibility + emotionally-charged content + "
                    "strong social influence all peaked simultaneously."
                ),
                "research_value": "Identifies cascade mechanisms and vulnerability factors.",
            },
            "intervention_backfire": {
                "icon": "💥",
                "desc": (
                    "Exposure bias *increased* after diversity injection. "
                    "The YouTube RL bandit learned to assign high skip-probability "
                    "to intervention content, reinforcing the bubble instead of breaking it."
                ),
                "research_value": "Identifies when diversity injection strengthens bubbles.",
            },
        }

        for atype, meta in descriptions.items():
            count = anomaly_types.get(atype, 0)
            with st.expander(
                f"{meta['icon']} {atype.replace('_', ' ').title()} — **{count:,} events**",
                expanded=(count > 0),
            ):
                st.markdown(f"**What happened:** {meta['desc']}")
                st.markdown(f"**Research value:** *{meta['research_value']}*")

        st.markdown("---")

        # ── What anomalies mean ───────────────────────────────────────────────
        st.subheader("What Anomalies Mean for Research")
        st.markdown("""
When **Sequential Layering** (LLM analysis) runs at checkpoints, it focuses on anomalous
agents to generate explanations for *why* unexpected outcomes occurred.

| Anomaly Type | LLM Focus | Output |
|---|---|---|
| Unexpected Resilience | Protective factors | Which agent traits prevent bubble formation |
| Rapid Radicalization | Cascade mechanisms | Which content × social × trait combos drive fast drift |
| Intervention Backfire | Adaptation pathways | How the RL bandit learned to avoid intervention content |

Anomalies flagged here feed directly into the `llm/coordinator.py` pipeline
as high-priority cases for the next sequential analysis pass.
        """)

    else:
        st.warning(
            "⚠️ No summary data found. "
            "Run `python -m experiments.full_run` first, then refresh."
        )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — MULTI-RUN COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Multi-Run Comparison":

    st.title("Multi-Run Experiment Comparison")
    st.markdown(
        "Comparing four intervention strategies "
        "across identical simulation conditions (300 agents, 1 000 timesteps, seed=42)."
    )

    all_runs = load_all_summaries()

    if not all_runs:
        st.warning(
            "No run summaries found. Run the experiment files first:\n\n"
            "```\n"
            "python -m experiments.full_run\n"
            "python -m experiments.run_002_early_intervention\n"
            "python -m experiments.run_003_algorithm_friction\n"
            "python -m experiments.run_004_social_bridge\n"
            "```"
        )
    else:
        n_available = len(all_runs)
        if n_available == 4:
            st.success(f"All 4 runs available")
        else:
            st.info(f"{n_available} of 4 runs available — run remaining experiments to unlock full comparison")

        # ── COMPARISON TABLE ─────────────────────────────────────────────────
        st.subheader("Results Comparison Table")

        table_data = []
        for label, data in all_runs.items():
            intervention_effect = (
                data.get("post_intervention_bias", 0)
                - data.get("late_exposure_bias", 0)
            )
            table_data.append({
                "Run": label,
                "Final Mean Belief":
                    f"{data.get('final_mean_belief', 0):.3f}",
                "Pre-Intervention Bias":
                    f"{data.get('late_exposure_bias', 0):.3f}",
                "Post-Intervention Bias":
                    f"{data.get('post_intervention_bias', 0):.3f}",
                "Intervention Effect":
                    f"{intervention_effect:+.3f}",
                "Anomalies":
                    data.get("total_anomalies", 0),
                "Adapted %":
                    f"{data.get('adapted_fraction', 0):.1%}",
            })

        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

        # ── INTERVENTION EFFECT CHART ─────────────────────────────────────────
        st.subheader("Intervention Effect by Strategy")
        st.markdown(
            "**Negative** = intervention reduced bubbles (good). "
            "**Positive** = intervention worsened bubbles (backfire)."
        )

        labels = list(all_runs.keys())
        short_labels = [
            lbl.split("\u2014")[1].strip() if "\u2014" in lbl else lbl
            for lbl in labels
        ]
        effects = []
        colors  = []
        for data in all_runs.values():
            effect = (
                data.get("post_intervention_bias", 0)
                - data.get("late_exposure_bias", 0)
            )
            effects.append(effect)
            colors.append("#d32f2f" if effect > 0 else "#388e3c")

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(
            short_labels, effects,
            color=colors, alpha=0.82, edgecolor="black", linewidth=0.8
        )
        ax.axhline(y=0, color="black", linewidth=1.5)
        ax.set_ylabel("Intervention Effect on Exposure Bias", fontsize=11)
        ax.set_title(
            "Intervention Effect Comparison\n"
            "(Red = backfire, Green = effective)",
            fontweight="bold"
        )
        ax.grid(True, alpha=0.25, axis="y", linestyle="--")
        for bar, effect in zip(bars, effects):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.002 if effect >= 0 else -0.006),
                f"{effect:+.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )
        plt.xticks(rotation=15, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── BELIEF DRIFT COMPARISON ───────────────────────────────────────────
        st.subheader("Final Mean Belief by Run")
        st.markdown(
            "All runs start at ~5.0. "
            "Lower final mean = more leftward drift from the neutral starting point."
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        final_means = [
            data.get("final_mean_belief", 5.0)
            for data in all_runs.values()
        ]
        ax.bar(
            short_labels, final_means,
            color="#1565c0", alpha=0.75,
            edgecolor="black", linewidth=0.8
        )
        ax.axhline(
            y=5.0, color="#888",
            linestyle="--", alpha=0.7, linewidth=1.5,
            label="Starting mean (5.0)"
        )
        ax.set_ylabel("Final Mean Belief Position", fontsize=11)
        ax.set_title("Final Mean Belief Position by Run", fontweight="bold")
        ax.set_ylim(0, 10)
        ax.legend(framealpha=0.85)
        ax.grid(True, alpha=0.25, axis="y", linestyle="--")
        plt.xticks(rotation=15, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── ANOMALY COMPARISON ────────────────────────────────────────────────
        st.subheader("Anomaly Counts by Run")

        fig, ax = plt.subplots(figsize=(10, 4))
        anomaly_counts = [
            data.get("total_anomalies", 0)
            for data in all_runs.values()
        ]
        ax.bar(
            short_labels, anomaly_counts,
            color="#f57c00", alpha=0.78,
            edgecolor="black", linewidth=0.8
        )
        ax.set_ylabel("Total Anomalies Detected", fontsize=11)
        ax.set_title("Anomaly Count by Intervention Type", fontweight="bold")
        ax.grid(True, alpha=0.25, axis="y", linestyle="--")
        plt.xticks(rotation=15, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── KEY FINDING ───────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Cross-Run Finding")

        if len(effects) >= 2:
            best_idx  = effects.index(min(effects))
            worst_idx = effects.index(max(effects))
            best_label  = short_labels[best_idx]
            worst_label = short_labels[worst_idx]
            st.info(
                f"**Most effective intervention:** {best_label} "
                f"(effect: {min(effects):+.3f})\n\n"
                f"**Least effective intervention:** {worst_label} "
                f"(effect: {max(effects):+.3f})"
            )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — DEEP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔬 Deep Analysis":

    st.title("Deep Run Analysis")

    run_options = {
        "Run 001 — Diversity Injection (t=800)": 
            "research_run_001",
        "Run 002 — Early Intervention (t=400)": 
            "research_run_002",
        "Run 003 — Algorithm Friction": 
            "research_run_003",
        "Run 004 — Social Bridge": 
            "research_run_004",
    }

    selected = st.selectbox(
        "Select run to analyze", 
        list(run_options.keys())
    )
    run_id = run_options[selected]
    data = load_full_results(run_id)

    if data is None:
        st.warning(
            f"No full results found for {run_id}. "
            f"Run the experiment file first."
        )
    else:
        config = data["config"]
        intervention_start = config["intervention_start"]
        total_timesteps = config["total_timesteps"]
        timesteps = list(range(1, total_timesteps + 1))
        
        # ── ROW 1: KEY METRICS ──
        col1, col2, col3, col4 = st.columns(4)
        final_mean = data["mean_trajectory"][-1]
        col1.metric(
            "Final Mean Belief", 
            f"{final_mean:.3f}",
            delta=f"{final_mean - 5.0:+.3f}"
        )
        col2.metric(
            "Total Anomalies", 
            data["total_anomalies"]
        )
        col3.metric(
            "Skip Events", 
            f"{data['skip_count']:,}"
        )
        col4.metric(
            "Runtime", 
            f"{data['elapsed_seconds']/60:.1f} min"
        )
        
        st.markdown("---")
        
        # ── CHART 1: BELIEF TRAJECTORY ──
        st.subheader("Population Belief Position Over Time")
        
        fig, ax = plt.subplots(figsize=(12, 4))
        means = data["mean_trajectory"]
        stds = data["std_trajectory"]
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        
        ax.plot(timesteps, means, "b-", linewidth=2,
                label="Population mean")
        ax.fill_between(
            timesteps,
            means_arr - stds_arr,
            means_arr + stds_arr,
            alpha=0.2, color="blue",
            label="±1 std deviation"
        )
        ax.axhline(y=5.0, color="gray", linestyle="--",
                   alpha=0.5, label="Starting mean")
        ax.axvline(x=intervention_start, color="red",
                   linestyle="--", alpha=0.8,
                   label=f"Intervention (t={intervention_start})")
        ax.axvspan(0, 100, alpha=0.05, color="gray",
                   label="Burn-in period")
        ax.set_xlabel("Timestep (1 = 1 day)")
        ax.set_ylabel("Mean Belief Position")
        ax.set_title(
            f"Belief Trajectory — {config['intervention_type']} "
            f"intervention at t={intervention_start}"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 10)
        st.pyplot(fig)
        plt.close()
        
        # ── CHART 2: THREE METRICS OVER TIME ──
        st.subheader("Bubble Metrics Over Time")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        metrics = [
            ("diversity_trajectory", "Content Diversity Score",
             "green", "Higher = more diverse content consumed"),
            ("echo_trajectory", "Echo Chamber Index",
             "orange", "Higher = stronger echo chamber"),
            ("exposure_trajectory", "Exposure Bias",
             "red", "Higher = more ideologically narrow"),
        ]
        
        for ax, (key, title, color, desc) in zip(axes, metrics):
            values = data[key]
            ax.plot(timesteps, values, color=color, linewidth=1.5)
            ax.axvline(x=intervention_start, color="black",
                       linestyle="--", alpha=0.5)
            ax.axvspan(0, 100, alpha=0.05, color="gray")
            ax.set_xlabel("Timestep")
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            st.caption(desc)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # ── CHART 3: YOUTUBE WEIGHT EVOLUTION ──
        if data["weight_history"]:
            st.subheader("YouTube RL Weight Evolution")
            
            wh = data["weight_history"]
            wh_timesteps = [w["timestep"] for w in wh]
            wv = [w["W_V"] for w in wh]
            we = [w["W_E"] for w in wh]
            wn = [w["W_N"] for w in wh]
            wd = [w["W_D"] for w in wh]
            
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(wh_timesteps, wv, label="W_V (Valence)",
                    linewidth=2, color="red")
            ax.plot(wh_timesteps, we, label="W_E (Engagement)",
                    linewidth=2, color="blue")
            ax.plot(wh_timesteps, wn, label="W_N (Novelty)",
                    linewidth=2, color="green")
            ax.plot(wh_timesteps, wd, label="W_D (Distance penalty)",
                    linewidth=2, color="purple")
            ax.axhline(y=0.25, color="gray", linestyle="--",
                       alpha=0.5, label="Starting value (0.25)")
            ax.axvline(x=intervention_start, color="black",
                       linestyle="--", alpha=0.5)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Weight Value")
            ax.set_title(
                "YouTube RL Algorithm Weight Evolution\n"
                "W_V rising = algorithm learned to push "
                "emotional content"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # ── CHART 4: ARCHETYPE BREAKDOWN ──
        st.subheader("Final Belief Position by Archetype")
        
        archetype_means = data["archetype_means"]
        if archetype_means:
            fig, ax = plt.subplots(figsize=(10, 4))
            archetypes = list(archetype_means.keys())
            means_by_arch = list(archetype_means.values())
            colors_arch = [
                "#1565c0" if m < 4.5 
                else "#43a047" if m < 5.5 
                else "#e53935"
                for m in means_by_arch
            ]
            bars = ax.bar(
                archetypes, means_by_arch,
                color=colors_arch, alpha=0.8,
                edgecolor="black"
            )
            ax.axhline(y=5.0, color="gray",
                       linestyle="--", alpha=0.7,
                       label="Starting mean (5.0)")
            for bar, mean in zip(bars, means_by_arch):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{mean:.2f}",
                    ha="center", va="bottom", fontsize=10
                )
            ax.set_ylabel("Final Mean Belief Position")
            ax.set_title("Final Belief by Agent Archetype")
            ax.set_ylim(0, 10)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # ── ANOMALY TIMELINE ──
        st.subheader("Anomaly Timeline")
        
        anomalies = data["anomaly_log"]
        if anomalies:
            fig, ax = plt.subplots(figsize=(12, 3))
            
            colors_map = {
                "unexpected_resilience": "green",
                "rapid_radicalization": "red",
                "intervention_backfire": "orange",
            }
            
            for anomaly in anomalies:
                color = colors_map.get(
                    anomaly["type"], "gray"
                )
                ax.scatter(
                    anomaly["timestep"],
                    anomaly["belief_position"],
                    c=color, alpha=0.3, s=10
                )
            
            ax.axvline(
                x=intervention_start,
                color="black", linestyle="--",
                alpha=0.5,
                label=f"Intervention (t={intervention_start})"
            )
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Agent Belief Position")
            ax.set_title(
                "Anomaly Events Over Time\n"
                "Green=unexpected resilience, "
                "Red=rapid radicalization, "
                "Orange=backfire"
            )
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="green", 
                      label="Unexpected resilience"),
                Patch(facecolor="red",
                      label="Rapid radicalization"),
                Patch(facecolor="orange",
                      label="Intervention backfire"),
            ]
            ax.legend(handles=legend_elements)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            st.caption(
                f"Total anomalies: {len(anomalies)} | "
                f"Unexpected resilience: "
                f"{sum(1 for a in anomalies if a['type'] == 'unexpected_resilience')} | "
                f"Backfire: "
                f"{sum(1 for a in anomalies if a['type'] == 'intervention_backfire')}"
            )
        else:
            st.info("No anomalies detected in this run.")
