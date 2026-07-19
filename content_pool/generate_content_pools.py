"""
Content Pool Generator — Filter Bubble Simulation (Phase 0/1 data pipeline)

Produces three content pools matching the design doc's Section 1 schema:
  1. IMDB   — REAL data (data/raw/imdb_top_1000.csv, scraped IMDb top-1000 titles),
              cleaned + mapped to the universal + IMDB-specific properties.
              Optionally padded with clearly-flagged synthetic titles to reach
              a target pool size (design doc range: 5,000-10,000).
  2. Reddit — SYNTHETIC, generated per Section 1.2 / 1.6 spec (community_id,
              upvotes/downvotes, content_type, cross_post_origin).
  3. YouTube— SYNTHETIC, generated per Section 1.2 / 1.6 spec (topic_cluster,
              predicted_watch_time, thumbnail_appeal, creator_authority).

All three pools share the five universal properties (Section 1.1) and use the
Normal(mean=5.0, std=2.0) belief-space initialization from Section 1.3.

Usage:
    python generate_content_pools.py --config pool_config.yaml
    python generate_content_pools.py   # uses built-in defaults below
"""

import json
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------------------------------
# CONFIG — edit these or wire up to argparse/YAML later; kept flat & explicit
# on purpose so every magic number in the design doc is visible in one place.
# --------------------------------------------------------------------------
SEED = 42

IMDB_RAW_CSV = "imdb_top_1000.csv"          # real scraped data (1000 rows)
IMDB_TARGET_POOL_SIZE = 1000                 # set higher (e.g. 5000) to pad with
                                              # flagged synthetic titles; 1000 = real-only
IMDB_NOVELTY_DECAY_RATE = 0.002
IMDB_RELEASE_ERA_UNKNOWN = 1900

REDDIT_POOL_SIZE = 2500                      # design range: 2,000-3,000
REDDIT_N_COMMUNITIES = 20
REDDIT_COMMENT_RATIO = 0.3                   # fraction of items that are comments
REDDIT_CROSS_POST_RATE = 0.05
REDDIT_NOVELTY_DECAY_RATE = 0.05

YOUTUBE_POOL_SIZE = 4000                     # design range: 3,000-5,000
YOUTUBE_N_TOPIC_CLUSTERS = 15
YOUTUBE_NOVELTY_DECAY_RATE = 0.02

OUT_DIR = Path("content_pools")

rng = np.random.default_rng(SEED)
random.seed(SEED)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def normal_belief_position(n):
    """Section 1.3 — universal belief-space init for ALL platforms."""
    draws = rng.normal(loc=5.0, scale=2.0, size=n)
    return np.clip(draws, 0.0, 10.0)


def clip01(arr):
    return np.clip(arr, 0.0, 1.0)


def minmax_norm(series):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def log_norm(series):
    logged = np.log1p(series.astype(float))
    return minmax_norm(logged)


# --------------------------------------------------------------------------
# 1. IMDB — real data
# --------------------------------------------------------------------------
GENRE_VALENCE_WEIGHTS = {
    # heuristic prior used only to seed emotional_valence for real titles —
    # documented assumption, not derived from any external source.
    "Horror": 0.85, "Thriller": 0.75, "War": 0.75, "Crime": 0.65,
    "Action": 0.6, "Mystery": 0.55, "Sci-Fi": 0.5, "Drama": 0.45,
    "Adventure": 0.45, "Fantasy": 0.4, "Sport": 0.4, "Biography": 0.35,
    "History": 0.35, "Romance": 0.35, "Comedy": 0.3, "Animation": 0.3,
    "Family": 0.25, "Music": 0.3, "Musical": 0.3, "Film-Noir": 0.5,
    "Western": 0.5,
}


def build_imdb_pool():
    raw = pd.read_csv(IMDB_RAW_CSV)
    n = len(raw)

    genres_split = raw["Genre"].str.split(", ")
    all_genres = sorted({g for lst in genres_split for g in lst})
    genre_index = {g: i for i, g in enumerate(all_genres)}

    def to_genre_vector(genre_list):
        vec = [0] * len(all_genres)
        for g in genre_list:
            vec[genre_index[g]] = 1
        return vec

    def to_valence(genre_list):
        weights = [GENRE_VALENCE_WEIGHTS.get(g, 0.4) for g in genre_list]
        base = float(np.mean(weights))
        return base

    runtime = raw["Runtime"].str.replace(" min", "", regex=False).astype(float)
    year = pd.to_numeric(raw["Released_Year"], errors="coerce").fillna(IMDB_RELEASE_ERA_UNKNOWN).astype(int)
    era = (year // 10) * 10
    gross = raw["Gross"].astype(str).str.replace(",", "", regex=False)
    gross = pd.to_numeric(gross, errors="coerce").fillna(0)

    meta_score = raw["Meta_score"].fillna(raw["Meta_score"].median())
    imdb_rating = raw["IMDB_Rating"]

    critical_consensus = clip01(meta_score / 100.0)
    imdb_rating_norm = clip01((imdb_rating - 1) / 9.0)          # design doc formula, Sec 3.1
    mainstream_appeal = clip01(log_norm(raw["No_of_Votes"]))
    quality_signal = clip01(0.5 * critical_consensus + 0.5 * imdb_rating_norm)

    valence_base = np.array([to_valence(g) for g in genres_split])
    emotional_valence = clip01(valence_base + rng.normal(0, 0.08, n))

    engagement_potential = clip01(
        0.6 * mainstream_appeal + 0.4 * clip01(log_norm(gross.replace(0, np.nan)).fillna(0.3))
        + rng.normal(0, 0.05, n)
    )

    df = pd.DataFrame({
        "content_id": [f"imdb_{i:05d}" for i in range(n)],
        "source": "real",
        "title": raw["Series_Title"],
        "released_year": year,
        "release_era": era,
        "genre_list": genres_split.apply(lambda g: json.dumps(g)),
        "genre_vector": [json.dumps(to_genre_vector(g)) for g in genres_split],
        "runtime_minutes": runtime,
        "imdb_rating_raw": imdb_rating,
        "imdb_rating_normalized": imdb_rating_norm,
        "meta_score": meta_score,
        "critical_consensus": critical_consensus,
        "no_of_votes": raw["No_of_Votes"],
        "mainstream_appeal": mainstream_appeal,
        "gross_usd": gross,
        # Universal properties (Section 1.1)
        "position_in_belief_space": normal_belief_position(n),
        "emotional_valence": emotional_valence,
        "engagement_potential": engagement_potential,
        "quality_signal": quality_signal,
        "novelty": 1.0,
        "birth_timestep": 0,
        # Left for simulation runtime to initialize (not derivable from raw data):
        # latent_features (20-30 dim, random init), ideological_loading_vector,
        # item_bias (init 0.0)
    })

    n_pad = IMDB_TARGET_POOL_SIZE - n
    if n_pad > 0:
        df_pad = synthesize_additional_movies(df, all_genres, genre_index, n_pad)
        df = pd.concat([df, df_pad], ignore_index=True)
    elif n_pad < 0:
        df = df.sample(n=IMDB_TARGET_POOL_SIZE, random_state=SEED).reset_index(drop=True)

    return df, all_genres


def synthesize_additional_movies(real_df, all_genres, genre_index, n_pad):
    """
    Pads the pool past the real 1000 titles by sampling from the EMPIRICAL
    distributions of the real data (genre frequency, rating distribution,
    era distribution, vote counts) rather than inventing values from
    scratch. These rows are clearly flagged source='synthetic_augmented' —
    do not present them as real IMDb titles in your writeup.
    """
    genre_freq = real_df["genre_list"].apply(json.loads).explode().value_counts(normalize=True)
    eras = real_df["release_era"].value_counts(normalize=True)
    rating_samples = real_df["imdb_rating_raw"].to_numpy()
    votes_samples = real_df["no_of_votes"].to_numpy()

    rows = []
    for i in range(n_pad):
        k = rng.integers(1, 4)
        genres = list(rng.choice(genre_freq.index, size=k, replace=False, p=genre_freq.values))
        era = int(rng.choice(eras.index, p=eras.values))
        rating = float(rng.choice(rating_samples)) + rng.normal(0, 0.2)
        rating = float(np.clip(rating, 1.0, 10.0))
        votes = int(max(100, rng.choice(votes_samples) * rng.uniform(0.5, 1.5)))
        meta = float(np.clip(rng.normal(60, 15), 1, 100))
        vec = [0] * len(all_genres)
        for g in genres:
            vec[genre_index[g]] = 1
        valence = float(np.clip(np.mean([GENRE_VALENCE_WEIGHTS.get(g, 0.4) for g in genres])
                                 + rng.normal(0, 0.08), 0, 1))
        rows.append({
            "content_id": f"imdb_synth_{i:05d}",
            "source": "synthetic_augmented",
            "title": f"[Synthetic Title {i}]",
            "released_year": era + rng.integers(0, 10),
            "release_era": era,
            "genre_list": json.dumps(genres),
            "genre_vector": json.dumps(vec),
            "runtime_minutes": float(np.clip(rng.normal(110, 20), 60, 220)),
            "imdb_rating_raw": rating,
            "imdb_rating_normalized": (rating - 1) / 9.0,
            "meta_score": meta,
            "critical_consensus": meta / 100.0,
            "no_of_votes": votes,
            "mainstream_appeal": None,  # filled after concat, needs full-pool log-norm
            "gross_usd": 0,
            "position_in_belief_space": float(normal_belief_position(1)[0]),
            "emotional_valence": valence,
            "engagement_potential": None,
            "quality_signal": (meta / 100.0 + (rating - 1) / 9.0) / 2,
            "novelty": 1.0,
            "birth_timestep": 0,
        })
    pad_df = pd.DataFrame(rows)
    pad_df["mainstream_appeal"] = clip01(log_norm(pad_df["no_of_votes"]))
    pad_df["engagement_potential"] = clip01(0.6 * pad_df["mainstream_appeal"] + 0.4 * 0.3)
    return pad_df


# --------------------------------------------------------------------------
# 2. Reddit — synthetic (Section 1.2, 1.6)
# --------------------------------------------------------------------------
def build_reddit_pool():
    n = REDDIT_POOL_SIZE
    community_id = rng.integers(0, REDDIT_N_COMMUNITIES, size=n)
    content_type = np.where(rng.random(n) < REDDIT_COMMENT_RATIO, "comment", "post")

    # Vote counts: lognormal to mimic real skewed engagement (few huge posts,
    # long tail of low-engagement ones) — "realistic vote baselines" per 1.2.
    net_karma = rng.lognormal(mean=2.2, sigma=1.4, size=n).astype(int)
    upvote_ratio = clip01(rng.normal(0.85, 0.1, n))
    upvotes = np.round(net_karma / np.clip(2 * upvote_ratio - 1, 0.05, None)).astype(int)
    upvotes = np.clip(upvotes, 1, None)
    downvotes = np.clip(upvotes - net_karma, 0, None).astype(int)

    timestamp = rng.integers(0, 50, size=n)  # birth timestep spread over early warm-up window

    cross_post_origin = np.where(
        rng.random(n) < REDDIT_CROSS_POST_RATE,
        rng.integers(0, REDDIT_N_COMMUNITIES, size=n),
        -1,  # -1 == None (no cross-post)
    )

    emotional_valence = clip01(rng.beta(2, 2, n))  # spread across range, mild center bias
    engagement_potential = clip01(0.5 * minmax_norm(pd.Series(net_karma)) + 0.5 * rng.random(n))
    quality_signal = clip01(rng.normal(0.5, 0.2, n))

    df = pd.DataFrame({
        "content_id": [f"reddit_{i:06d}" for i in range(n)],
        "community_id": community_id,
        "content_type": content_type,
        "upvotes": upvotes,
        "downvotes": downvotes,
        "timestamp": timestamp,
        "cross_post_origin": cross_post_origin,  # -1 = None
        "position_in_belief_space": normal_belief_position(n),
        "emotional_valence": emotional_valence,
        "engagement_potential": engagement_potential,
        "quality_signal": quality_signal,
        "novelty": 1.0,
        "birth_timestep": timestamp,
    })
    return df


# --------------------------------------------------------------------------
# 3. YouTube — synthetic (Section 1.2, 1.6)
# --------------------------------------------------------------------------
def build_youtube_pool():
    n = YOUTUBE_POOL_SIZE
    topic_cluster = rng.integers(0, YOUTUBE_N_TOPIC_CLUSTERS, size=n)

    # "varied valence profiles" per 1.2 — give each topic cluster its own
    # valence center so some clusters run hotter than others (rabbit-hole seed).
    cluster_valence_center = rng.uniform(0.2, 0.85, size=YOUTUBE_N_TOPIC_CLUSTERS)
    emotional_valence = clip01(
        cluster_valence_center[topic_cluster] + rng.normal(0, 0.12, n)
    )

    thumbnail_appeal = clip01(rng.beta(2, 2, n))
    creator_authority = clip01(rng.normal(0.5, 0.2, n))
    completion_rate_baseline = np.full(n, 0.5)  # design doc: "initialised at 0.5"
    predicted_watch_time = clip01(
        0.5 * completion_rate_baseline + 0.3 * thumbnail_appeal + 0.2 * rng.random(n)
    )
    engagement_potential = clip01(0.5 * predicted_watch_time + 0.5 * thumbnail_appeal)
    quality_signal = clip01(rng.normal(0.5, 0.2, n))

    df = pd.DataFrame({
        "content_id": [f"youtube_{i:06d}" for i in range(n)],
        "topic_cluster": topic_cluster,
        "predicted_watch_time": predicted_watch_time,
        "completion_rate_baseline": completion_rate_baseline,
        "thumbnail_appeal": thumbnail_appeal,
        "creator_authority": creator_authority,
        "position_in_belief_space": normal_belief_position(n),
        "emotional_valence": emotional_valence,
        "engagement_potential": engagement_potential,
        "quality_signal": quality_signal,
        "novelty": 1.0,
        "birth_timestep": 0,
    })
    return df


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)

    imdb_df, all_genres = build_imdb_pool()
    reddit_df = build_reddit_pool()
    youtube_df = build_youtube_pool()

    imdb_df.to_csv(OUT_DIR / "imdb_content_pool.csv", index=False)
    reddit_df.to_csv(OUT_DIR / "reddit_content_pool.csv", index=False)
    youtube_df.to_csv(OUT_DIR / "youtube_content_pool.csv", index=False)

    with open(OUT_DIR / "imdb_genre_index.json", "w") as f:
        json.dump({g: i for i, g in enumerate(all_genres)}, f, indent=2)

    summary = {
        "imdb": {
            "n_items": len(imdb_df),
            "n_real": int((imdb_df["source"] == "real").sum()),
            "n_synthetic_augmented": int((imdb_df["source"] == "synthetic_augmented").sum()),
            "n_genres": len(all_genres),
        },
        "reddit": {
            "n_items": len(reddit_df),
            "n_communities": REDDIT_N_COMMUNITIES,
            "pct_comments": float((reddit_df["content_type"] == "comment").mean()),
            "pct_cross_post": float((reddit_df["cross_post_origin"] != -1).mean()),
        },
        "youtube": {
            "n_items": len(youtube_df),
            "n_topic_clusters": YOUTUBE_N_TOPIC_CLUSTERS,
        },
        "seed": SEED,
    }
    with open(OUT_DIR / "generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
