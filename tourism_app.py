"""
Tourism Analytics — Streamlit Application
==========================================
Provides:
  1. Predicted Visit Mode  (classification)
  2. Predicted Attraction Rating  (regression)
  3. Personalized Recommendations — Collaborative + Content-Based
  4. Analytics Dashboard — popular attractions, regions, user segments
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import requests
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tourism Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# 👇 PASTE YOUR GOOGLE DRIVE SHARE LINKS BELOW (paste the full link)
# How: Right-click file in Drive → Share → "Anyone with the link" → Copy link
# ──────────────────────────────────────────────────────────────────────────────
GDRIVE_MODEL_LINK = "https://drive.google.com/file/d/1c8b2k_xhHOEhzR25KOYkKKfP59b1UivO/view?usp=sharing"
GDRIVE_DATA_LINK  = "https://drive.google.com/file/d/1wK50R_0RKFcZBX7AWkNzGK5_VTB4_Drf/view?usp=sharing"

# ──────────────────────────────────────────────────────────────────────────────
# Robust Google Drive downloader (handles large files & confirmation pages)
# ──────────────────────────────────────────────────────────────────────────────
def extract_file_id(link):
    match = re.search(r"/d/([a-zA-Z0-9_-]{20,})", link)
    return match.group(1) if match else link

def download_from_gdrive(link, dest_path, label):
    """Download from Google Drive, handling both old and new confirmation flows."""
    if os.path.exists(dest_path):
        os.remove(dest_path)  # remove any previously broken download
    file_id = extract_file_id(link)
    with st.spinner(f"Downloading {label} (first launch only - please wait)..."):
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = session.get(url, stream=True, timeout=60)
        # Old-style cookie token
        token = next((v for k, v in response.cookies.items() if k.startswith("download_warning")), None)
        if token:
            response = session.get(url, params={"confirm": token}, stream=True, timeout=120)
        # New-style HTML confirmation page - use usercontent endpoint instead
        if "text/html" in response.headers.get("Content-Type", ""):
            url2 = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
            response = session.get(url2, stream=True, timeout=120)
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
    # Verify the file is not an HTML error page
    with open(dest_path, "rb") as f:
        header = f.read(10)
    if header[:4] in (b"<htm", b"<!DO"):
        os.remove(dest_path)
        st.error(
            f"**Failed to download `{label}` from Google Drive.**\n\n"
            "Google is blocking the download. Please:\n"
            "1. Open the file in Google Drive\n"
            "2. Click Share → set to **Anyone with the link**\n"
            "3. Copy the new link and update it in the app"
        )
        st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Load models / data
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    download_from_gdrive(GDRIVE_MODEL_LINK, "tourism_models.pkl", "tourism_models.pkl")
    with open("tourism_models.pkl", "rb") as f:
        pkg = pickle.load(f)
    return pkg

@st.cache_data
def load_data():
    download_from_gdrive(GDRIVE_DATA_LINK, "preprocessed_tourism_data.csv", "preprocessed_tourism_data.csv")
    return pd.read_csv("preprocessed_tourism_data.csv")

# ──────────────────────────────────────────────────────────────────────────────
# Recommendation helpers
# ──────────────────────────────────────────────────────────────────────────────
def collab_recommend(user_id, pkg, df, n=5):
    uim = pkg["user_item_matrix"]
    knn = pkg["recommendation_knn"]
    if user_id not in uim.index:
        popular = df["AttractionId"].value_counts().head(n)
        return popular.index.tolist()
    u_idx = uim.index.get_loc(user_id)
    distances, indices = knn.kneighbors(
        uim.iloc[u_idx].values.reshape(1, -1),
        n_neighbors=min(15, len(uim)),
    )
    scores = {}
    for sim_idx in indices.flatten()[1:]:
        for col_idx, rating in enumerate(uim.iloc[sim_idx]):
            if rating > 0:
                aid = uim.columns[col_idx]
                if uim.iloc[u_idx, col_idx] == 0:
                    scores.setdefault(aid, []).append(rating)
    avg = {aid: np.mean(r) for aid, r in scores.items()}
    return [aid for aid, _ in sorted(avg.items(), key=lambda x: x[1], reverse=True)[:n]]


def content_recommend(user_id, pkg, df, n=5):
    visited = df[df["UserId"] == user_id]["AttractionId"].unique().tolist()
    sim_df  = pkg["item_similarity"]
    if not visited:
        popular = df["AttractionId"].value_counts().head(n)
        return popular.index.tolist()
    scores = {}
    for aid in visited:
        if aid not in sim_df.index:
            continue
        for cand, score in sim_df[aid].items():
            if cand not in visited:
                scores[cand] = scores.get(cand, 0) + score
    if not scores:
        popular = df["AttractionId"].value_counts().head(n)
        return popular.index.tolist()
    return [aid for aid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]]


def enrich_recs(rec_ids, df):
    rows = []
    attr_df = df[["AttractionId", "Attraction", "AttractionType",
                  "AttractionCityName", "attraction_avg_rating",
                  "attraction_visit_count"]].drop_duplicates("AttractionId")
    for aid in rec_ids:
        r = attr_df[attr_df["AttractionId"] == aid]
        if not r.empty:
            rows.append(r.iloc[0].to_dict())
    return pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR ──────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/globe.png", width=72)
st.sidebar.title("🌍 Tourism Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔮 Predictions", "🎯 Recommendations", "📊 Analytics Dashboard"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Try loading — show friendly error if files missing
# ──────────────────────────────────────────────────────────────────────────────
try:
    pkg = load_models()
    df  = load_data()
    models_loaded = True
except FileNotFoundError:
    models_loaded = False

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🌍 Tourism Analytics Platform")
    st.markdown(
        """
        Welcome! This platform uses **machine learning** to help travel agencies and tourists
        make smarter decisions.

        ### What can you do here?
        | Page | Description |
        |------|-------------|
        | 🔮 **Predictions** | Predict a user's visit mode and expected attraction rating |
        | 🎯 **Recommendations** | Get personalised attraction suggestions |
        | 📊 **Analytics Dashboard** | Explore tourism trends, top attractions, and user segments |
        """
    )

    if models_loaded:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records",    f"{len(df):,}")
        c2.metric("Unique Users",     f"{df['UserId'].nunique():,}")
        c3.metric("Attractions",      f"{df['AttractionId'].nunique()}")
        c4.metric("Avg Rating",       f"{df['Rating'].mean():.2f} / 5")

        st.markdown("---")
        st.subheader("Model Performance Summary")
        col1, col2 = st.columns(2)
        with col1:
            reg_p = pkg["regression_performance"]
            st.info(
                f"**Regression Model** ({pkg['regression_model_name']})\n\n"
                f"R² = `{reg_p['r2']:.4f}` | RMSE = `{reg_p['rmse']:.4f}` | MAE = `{reg_p['mae']:.4f}`"
            )
        with col2:
            cls_p = pkg["classification_performance"]
            st.success(
                f"**Classification Model** ({pkg['classification_model_name']})\n\n"
                f"Accuracy = `{cls_p['accuracy']:.4f}` | F1 = `{cls_p['f1']:.4f}`"
            )
    else:
        st.warning(
            "⚠️ Model files not found. Please run the notebook first to generate "
            "`tourism_models.pkl` and `preprocessed_tourism_data.csv`, then place them "
            "in the same folder as this app."
        )

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predictions":
    st.title("🔮 Predictions")
    if not models_loaded:
        st.error("Model files not loaded. Run the notebook first.")
        st.stop()

    st.markdown("Fill in the details below to get predictions.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("User Details")
        continent_opts = sorted(df["UserContinent"].dropna().unique())
        continent      = st.selectbox("Continent", continent_opts)
        country_opts   = sorted(df[df["UserContinent"] == continent]["UserCountry"].dropna().unique())
        country        = st.selectbox("Country", country_opts)
        region_opts    = sorted(df[df["UserCountry"] == country]["UserRegion"].dropna().unique())
        region         = st.selectbox("Region", region_opts if region_opts else ["Unknown"])

    with col2:
        st.subheader("Visit & Attraction Details")
        year  = st.slider("Visit Year",  2010, 2025, 2023)
        month = st.slider("Visit Month", 1, 12, 6)
        attraction_type_opts = sorted(df["AttractionType"].dropna().unique())
        attr_type = st.selectbox("Attraction Type", attraction_type_opts)

    # Build feature vector
    def get_feature_vector(feature_list):
        enc = pkg["encoders"]
        season_map = {**{m: "Winter" for m in [12,1,2]},
                      **{m: "Spring" for m in [3,4,5]},
                      **{m: "Summer" for m in [6,7,8]},
                      **{m: "Fall"   for m in [9,10,11]}}
        season = season_map[month]

        def safe_encode(encoder_key, value):
            le = enc.get(encoder_key)
            if le is None: return 0
            classes = list(le.classes_)
            return classes.index(value) if value in classes else 0

        feat_map = {
            "VisitYear":                  year,
            "VisitMonth":                 month,
            "ContinentId":                safe_encode("UserContinent", continent),
            "RegionId":                   safe_encode("UserRegion",    region),
            "CountryId":                  safe_encode("UserCountry",   country),
            "AttractionTypeId":           safe_encode("AttractionType", attr_type),
            "user_avg_rating":            df["user_avg_rating"].mean(),
            "user_rating_std":            df["user_rating_std"].mean(),
            "user_visit_count":           df["user_visit_count"].mean(),
            "attraction_avg_rating":      df[df["AttractionType"] == attr_type]["attraction_avg_rating"].mean()
                                          if attr_type in df["AttractionType"].values
                                          else df["attraction_avg_rating"].mean(),
            "attraction_rating_std":      df["attraction_rating_std"].mean(),
            "attraction_visit_count":     df["attraction_visit_count"].mean(),
            "Season_encoded":             safe_encode("Season", season),
            "Quarter":                    (month - 1) // 3 + 1,
            "user_visit_frequency":       df["user_visit_frequency"].mean(),
            "user_attraction_diversity":  df["user_attraction_diversity"].mean(),
            "attraction_popularity_score": df["attraction_popularity_score"].mean(),
        }
        return np.array([feat_map.get(f, 0) for f in feature_list]).reshape(1, -1)

    if st.button("🔮 Generate Predictions", use_container_width=True):
        with st.spinner("Running models..."):
            # Rating prediction
            reg_feats = pkg["regression_features"]
            X_reg     = get_feature_vector(reg_feats)
            pred_rating = pkg["regression_model"].predict(X_reg)[0]
            pred_rating = float(np.clip(pred_rating, 1, 5))

            # Visit mode prediction
            cls_feats = pkg["classification_features"]
            X_cls     = get_feature_vector(cls_feats)
            pred_mode_enc = pkg["classification_model"].predict(X_cls)[0]
            le_mode   = pkg["classification_encoder"]
            pred_mode = le_mode.inverse_transform([pred_mode_enc])[0]

            # Probabilities
            if hasattr(pkg["classification_model"], "predict_proba"):
                proba = pkg["classification_model"].predict_proba(X_cls)[0]
                mode_classes = le_mode.classes_
            else:
                proba, mode_classes = None, []

        st.markdown("---")
        r1, r2 = st.columns(2)
        with r1:
            stars = "⭐" * round(pred_rating)
            st.success(f"### Predicted Rating\n## {pred_rating:.2f} / 5.0\n{stars}")
        with r2:
            mode_emoji = {"Business": "💼", "Family": "👨‍👩‍👧", "Couples": "💑",
                          "Friends": "👫", "Solo": "🧍"}.get(pred_mode, "✈️")
            st.info(f"### Predicted Visit Mode\n## {mode_emoji} {pred_mode}")

        if proba is not None and len(mode_classes):
            st.subheader("Visit Mode Probability Distribution")
            prob_df = pd.DataFrame({"Mode": mode_classes, "Probability": proba * 100})\
                        .sort_values("Probability", ascending=True)
            fig, ax = plt.subplots(figsize=(8, 3))
            colors = ["#4CAF50" if m == pred_mode else "#90CAF9" for m in prob_df["Mode"]]
            ax.barh(prob_df["Mode"], prob_df["Probability"], color=colors)
            ax.set_xlabel("Probability (%)")
            ax.set_title("Visit Mode Probabilities")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Recommendations":
    st.title("🎯 Personalised Recommendations")
    if not models_loaded:
        st.error("Model files not loaded. Run the notebook first.")
        st.stop()

    st.markdown(
        "Enter a **User ID** to get personalised attraction recommendations using both "
        "collaborative filtering and content-based filtering."
    )

    sample_users = df["UserId"].value_counts().head(20).index.tolist()
    user_id_input = st.selectbox(
        "Select or enter a User ID",
        options=sample_users,
        help="These are the most active users in the dataset.",
    )

    n_recs = st.slider("Number of recommendations", 3, 10, 5)
    method = st.radio(
        "Recommendation Method",
        ["Both", "Collaborative Filtering", "Content-Based Filtering"],
        horizontal=True,
    )

    if st.button("🎯 Get Recommendations", use_container_width=True):
        with st.spinner("Finding best attractions for you..."):
            user_history = df[df["UserId"] == user_id_input][["Attraction", "AttractionType", "Rating"]]\
                             .drop_duplicates("Attraction")

            collab_ids  = collab_recommend(user_id_input,  pkg, df, n=n_recs)
            content_ids = content_recommend(user_id_input, pkg, df, n=n_recs)

        # User profile
        st.markdown("---")
        st.subheader(f"👤 User {user_id_input} — Visit History")
        if not user_history.empty:
            urow = df[df["UserId"] == user_id_input].iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Visits",     len(user_history))
            c2.metric("Avg Rating Given", f"{df[df['UserId']==user_id_input]['Rating'].mean():.2f}")
            c3.metric("From",             urow.get("UserCountry", "Unknown"))
            st.dataframe(
                user_history.rename(columns={"Attraction": "Attraction Visited",
                                             "AttractionType": "Type",
                                             "Rating": "Rating Given"}).head(8),
                use_container_width=True,
            )
        else:
            st.info("No history found for this user (cold start).")

        st.markdown("---")

        def show_recs(recs_df, title, color):
            st.subheader(title)
            if recs_df.empty:
                st.warning("No recommendations found.")
                return
            for i, row in recs_df.iterrows():
                with st.container():
                    c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
                    c1.markdown(f"**{row.get('Attraction', 'N/A')}**")
                    c2.markdown(f"🏛️ {row.get('AttractionType','')}")
                    c3.markdown(f"📍 {row.get('AttractionCityName','')}")
                    avg_r = row.get("attraction_avg_rating", 0)
                    c4.markdown(f"⭐ {avg_r:.2f}")

        if method in ["Both", "Collaborative Filtering"]:
            collab_df = enrich_recs(collab_ids, df)
            show_recs(collab_df, "🤝 Collaborative Filtering Recommendations", "blue")

        if method in ["Both", "Content-Based Filtering"]:
            content_df = enrich_recs(content_ids, df)
            show_recs(content_df, "🔍 Content-Based Filtering Recommendations", "green")

        if method == "Both" and not collab_df.empty and not content_df.empty:
            st.markdown("---")
            st.subheader("🔀 Hybrid Recommendations (Union)")
            combined_ids = list(dict.fromkeys(collab_ids + content_ids))[:n_recs]
            hybrid_df    = enrich_recs(combined_ids, df)
            st.dataframe(
                hybrid_df[["Attraction", "AttractionType", "AttractionCityName",
                            "attraction_avg_rating", "attraction_visit_count"]]
                  .rename(columns={"attraction_avg_rating": "Avg Rating",
                                   "attraction_visit_count": "Total Visits",
                                   "AttractionCityName": "City",
                                   "AttractionType": "Type"}),
                use_container_width=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    if not models_loaded:
        st.error("Model files not loaded. Run the notebook first.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🌍 Geographic", "🏛️ Attractions", "👥 User Segments", "📈 Trends"]
    )

    # ── Geographic ──────────────────────────────────────────────────────────
    with tab1:
        st.subheader("User Distribution by Geography")
        col1, col2 = st.columns(2)

        with col1:
            cont_dist = df["UserContinent"].value_counts()
            fig, ax   = plt.subplots(figsize=(7, 4))
            ax.barh(cont_dist.index, cont_dist.values, color="teal")
            ax.set_title("Visits by Continent", fontweight="bold")
            ax.set_xlabel("Number of Visits")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with col2:
            top_country = df["UserCountry"].value_counts().head(10)
            fig, ax     = plt.subplots(figsize=(7, 4))
            ax.barh(top_country.index[::-1], top_country.values[::-1], color="steelblue")
            ax.set_title("Top 10 Countries", fontweight="bold")
            ax.set_xlabel("Number of Visits")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # Visit Mode by Continent
        st.subheader("Visit Mode Distribution by Continent")
        mc = pd.crosstab(df["UserContinent"], df["VisitModeName"], normalize="index") * 100
        fig, ax = plt.subplots(figsize=(12, 4))
        mc.plot(kind="bar", ax=ax, colormap="Set3", width=0.7)
        ax.set_title("Visit Mode % by Continent", fontweight="bold")
        ax.set_ylabel("Percentage (%)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(title="Visit Mode", bbox_to_anchor=(1.01, 1))
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Attractions ─────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Attraction Insights")

        col1, col2 = st.columns(2)
        with col1:
            top10 = df["Attraction"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(top10.index[::-1], top10.values[::-1], color="coral")
            ax.set_title("Top 10 Most Visited Attractions", fontweight="bold")
            ax.set_xlabel("Number of Visits")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with col2:
            type_rating = df.groupby("AttractionType")["Rating"].mean().sort_values(ascending=True)
            fig, ax     = plt.subplots(figsize=(7, 4))
            ax.barh(type_rating.index, type_rating.values, color="gold")
            ax.set_title("Avg Rating by Attraction Type", fontweight="bold")
            ax.set_xlabel("Average Rating")
            ax.set_xlim([type_rating.min() * 0.97, 5])
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # Rating distribution
        st.subheader("Overall Rating Distribution")
        fig, ax = plt.subplots(figsize=(8, 3))
        df["Rating"].value_counts().sort_index().plot(kind="bar", ax=ax, color="skyblue")
        ax.set_xlabel("Rating"); ax.set_ylabel("Count")
        ax.set_title("Rating Distribution", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── User Segments ───────────────────────────────────────────────────────
    with tab3:
        st.subheader("User Segments by Visit Mode")

        mode_dist = df.drop_duplicates("TransactionId")["VisitModeName"].value_counts()
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 6))
            wedges, texts, autotexts = ax.pie(
                mode_dist.values, labels=mode_dist.index,
                autopct="%1.1f%%", startangle=90,
                colors=plt.cm.Set3.colors[:len(mode_dist)]
            )
            ax.set_title("Visit Mode Share", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with col2:
            mode_avg = df.groupby("VisitModeName")["Rating"].agg(["mean", "count"]).reset_index()
            mode_avg.columns = ["Visit Mode", "Avg Rating", "Total Visits"]
            mode_avg = mode_avg.sort_values("Avg Rating", ascending=False)
            st.dataframe(
                mode_avg.style.format({"Avg Rating": "{:.3f}", "Total Visits": "{:,.0f}"}),
                use_container_width=True
            )

        # Season vs mode heatmap
        st.subheader("Visits by Season and Mode")
        season_mode = pd.crosstab(df["Season"], df["VisitModeName"])
        fig, ax = plt.subplots(figsize=(10, 4))
        import seaborn as sns
        sns.heatmap(season_mode, annot=True, fmt="d", cmap="Blues",
                    linewidths=0.5, ax=ax)
        ax.set_title("Visits: Season × Visit Mode", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Trends ──────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Tourism Trends Over Time")

        year_visits = df.groupby("VisitYear").size().reset_index(name="Visits")
        fig, ax     = plt.subplots(figsize=(10, 4))
        ax.plot(year_visits["VisitYear"], year_visits["Visits"],
                marker="o", color="steelblue", linewidth=2)
        ax.fill_between(year_visits["VisitYear"], year_visits["Visits"],
                         alpha=0.15, color="steelblue")
        ax.set_title("Total Visits by Year", fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel("Number of Visits")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        col1, col2 = st.columns(2)
        with col1:
            year_rating = df.groupby("VisitYear")["Rating"].mean().reset_index()
            fig, ax     = plt.subplots(figsize=(6, 3))
            ax.plot(year_rating["VisitYear"], year_rating["Rating"],
                    marker="s", color="green", linewidth=2)
            ax.set_title("Avg Rating by Year", fontweight="bold")
            ax.set_xlabel("Year"); ax.set_ylabel("Avg Rating")
            ax.set_ylim([3, 5])
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with col2:
            month_visits = df.groupby("VisitMonth").size().reset_index(name="Visits")
            month_names  = ["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"]
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(month_visits["VisitMonth"], month_visits["Visits"], color="mediumpurple")
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(month_names, fontsize=9)
            ax.set_title("Visits by Month (Seasonality)", fontweight="bold")
            ax.set_ylabel("Number of Visits")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("Tourism Analytics Platform · Built with Streamlit & scikit-learn")
