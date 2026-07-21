# 🫧 Filter Bubble Simulation Framework

An agent-based research framework for modeling and analyzing **filter bubble formation**, **ideological drift**, and **algorithmic echo chambers** across heterogeneous online platforms (**IMDB**, **Reddit**, and **YouTube**).

Includes an interactive **Streamlit Dashboard**, multi-experiment batch runner, and an optional **LLM Pipeline** powered by **Groq** (`llama-3.3-70b-versatile`) for static parameter grounding, dynamic anomaly analysis, and sequential checkpoint research synthesis.

---

## 🌟 Key Features

- **Multi-Platform Ecosystem**:
  - **🎬 IMDB**: Taste-similarity clustering via SGD Matrix Factorization & dynamic k-NN graphs.
  - **💬 Reddit**: Community absorption & echo chambers via Stochastic Block Models, Hot sort & Wilson score ranking.
  - **📺 YouTube**: Algorithmic drift & rabbit holes via Contextual Bandit Reinforcement Learning ($W_V, W_E, W_N, W_D$ weights).
- **Intervention Testing Engine**:
  - **Diversity Injection**: Forces ideologically distant content into recommendations.
  - **Algorithm Friction**: Regulatory-style intervention curbing emotional valence amplification.
  - **Social Bridge**: Introduces cross-community ties across ideological boundaries.
  - **Null Control**: Baseline natural bubble formation.
- **Interactive Streamlit Dashboard**:
  - 📊 Overview metrics & population trajectory charts.
  - 📈 Interactive mini-simulation runner.
  - 🤖 Platform RL weight evolution & mechanism breakdowns.
  - 🛡️ Pre/post intervention efficacy & adaptation tracking.
  - ⚠️ Anomaly timeline & detection breakdown.
  - 📊 Multi-run cross-experiment comparison.
  - 🔬 Deep Run Analysis with $\pm 1$ std deviation trajectory bands & archetype breakdowns.
- **LLM Research Coordinator (Groq powered)**:
  - Grounding agent psychology parameters against empirical research.
  - Dynamic real-time analysis of anomalous agent events (rapid radicalization, backfire).
  - Sequential checkpoint hypothesis generation for multi-batch simulation runs.

---

## 🏗️ Repository Architecture

```text
filter_bubble_sim/
├── analysis/
│   └── dashboard.py                  # Full 7-section Streamlit Dashboard
├── config/
│   ├── env_loader.py                 # Loads GROQ_API_KEY from root .env
│   └── simulation_config.py          # Central SimulationConfig & default hyperparams
├── core/
│   ├── agent.py                      # Agent class & psychological state mutation
│   ├── content.py                    # BaseContent & platform-specific content schemas
│   └── platform.py                   # Platform base class & PlatformFactory
├── data/
│   ├── loader.py                     # Load synthetic IMDB, Reddit & YouTube CSV pools
│   ├── imdb/
│   ├── reddit/
│   └── youtube/
├── experiments/
│   ├── run_simulation.py             # 9-step interaction loop & simulation state runner
│   ├── full_run.py                   # Research Run 001 (Diversity Injection t=800)
│   ├── run_002_early_intervention.py # Research Run 002 (Early Diversity Injection t=400)
│   ├── run_003_algorithm_friction.py # Research Run 003 (Algorithm Friction t=800)
│   └── run_004_social_bridge.py      # Research Run 004 (Social Bridge t=800)
├── interventions/
│   ├── manager.py                    # Intervention activation & adaptation tracking
│   ├── diversity_injection.py
│   ├── algorithm_friction.py
│   └── social_bridge.py
├── llm/
│   ├── coordinator.py                # LLMPipelineCoordinator
│   ├── static_llm.py                 # Psychology parameter grounding pass
│   ├── dynamic_llm.py                # Mid-sim anomaly trigger analysis
│   └── sequential_llm.py             # Checkpoint synthesis & hypothesis generator
├── measurement/
│   ├── store.py                      # MeasurementStore for all agent time-series
│   ├── metrics.py                    # Belief drift, exposure bias & diversity calculations
│   └── anomaly.py                    # Anomaly detectors (resilience, backfire, radicalization)
├── network/
│   └── multiplex.py                  # Multi-layer social graph & influence propagation
├── tests/                            # Comprehensive unit test suite (pytest)
├── .env                              # API key configuration (ignored by git)
├── .gitignore                        # Git exclusion rules
└── requirements.txt                  # Python package dependencies
```

---

## 🚀 Quick Start Guide

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/pavitravashishtha/Filter_Bubble.git
cd Filter_Bubble
pip install -r requirements.txt
```

*(Or manually: `pip install streamlit matplotlib pandas numpy pytest groq`)*

### 2. Configure API Keys (Optional for LLM features)

Create or update the `.env` file in the project root:

```env
GROQ_API_KEY=your-groq-api-key-here
```

*Note: Simulation runs work out-of-the-box with `use_llm=False` without requiring an API key.*

### 3. Run Experiments

Generate simulation runs and full results data:

```bash
# Research Run 001 — Diversity Injection (t=800)
python -m experiments.full_run

# Research Run 002 — Early Intervention (t=400)
python -m experiments.run_002_early_intervention

# Research Run 003 — Algorithm Friction (t=800)
python -m experiments.run_003_algorithm_friction

# Research Run 004 — Social Bridge (t=800)
python -m experiments.run_004_social_bridge
```

### 4. Launch the Streamlit Dashboard

Start the interactive dashboard web server:

```bash
python -m streamlit run analysis/dashboard.py
```

Open your browser at `http://localhost:8501`.

---

## 🧪 Running Unit Tests

Verify codebase integrity with `pytest`:

```bash
python -m pytest tests/ -v
```

---

## 📜 Research Simulation Summary

- **Population**: 300 initial agents + dynamic entry (archetypes: `heavy_imdb`, `heavy_reddit`, `heavy_youtube`, `balanced`, `cross_platform`).
- **Timesteps**: 1,000 timesteps (~1 timestep = 1 day equivalent).
- **Checkpoints**: Timesteps 100 (burn-in end), 200, 400, 600, 800 (pre-intervention), 900, 1000 (final).

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
