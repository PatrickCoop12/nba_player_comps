import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import pull_player_ids, get_player_comps

# Page Configuration
st.set_page_config(page_title="NBA Player Similarity Engine", layout="wide")

st.title("🏀 Active NBA Player Comparison Engine")
st.caption("🔥 **Who is today's star playing like?** Pick any active NBA player to instantly stack their career stats against retired legends and discover their ultimate historical comp.")

# Interactive Usage Guide & Overview
with st.expander("ℹ️ How It Works & Key Features", expanded=False):
    st.markdown("""
    **How It Works:**
    * **Career Averages Comparison:** The algorithm takes the full per-game career averages of an active NBA player and compares them across historical career datasets of retired NBA players.
    * **Cosine Similarity Scoring:** Vectors of normalized box score metrics are analyzed to identify retired players who produced the most statistically identical career profiles.

    **Key Features:**
    * 🎯 **Match Confidence Scores:** Visual ranking of the Top 10 closest historical comparisons.
    * ⚔️ **Head-to-Head Visual Comparison:** Overlaid per-game volume bar charts alongside a normalized skill radar profile.
    * 📊 **Comprehensive Data View:** View complete statistical breakdowns across traditional per-game statistics.
    """)

# Load active players dictionary
N_PLAYERS = pull_player_ids()

# Sidebar Configuration
st.sidebar.title("Player Selection")
selected_player = st.sidebar.selectbox("Select Active NBA Player", options=list(N_PLAYERS.keys()))

if selected_player:
    # Fetch reference stats and comparisons
    ref_stats_df, comps_df = get_player_comps(selected_player)

    # Isolate top 10 historical comparisons
    top_10 = comps_df.head(10).copy()

    # Column Mapper to align active and retired schemas
    column_map = {
        'player': 'Player', 'g': 'GP', 'gs': 'GS', 'mp': 'MIN', 'fg': 'FGM',
        'fga': 'FGA', 'fg_percent': 'FG%', 'x3p': '3PM', 'x3pa': '3PA',
        'x3p_percent': '3P%', 'ft': 'FTM', 'fta': 'FTA', 'ft_percent': 'FT%',
        'orb': 'OREB', 'drb': 'DREB', 'trb': 'REB', 'ast': 'AST',
        'stl': 'STL', 'blk': 'BLK', 'tov': 'TOV', 'pf': 'PF', 'pts': 'PTS'
    }

    # Format Top 10 data
    top_10_clean = top_10.drop(columns=[c for c in ['status'] if c in top_10.columns]).rename(columns=column_map)
    top_10_clean['Sim_Pct'] = (top_10_clean['similarity'] * 100).round(1)

    # Standardize Reference Player Data
    ref_clean = ref_stats_df.rename(columns={
        'name': 'Player', 'FG_PCT': 'FG%', 'FG3M': '3PM', 'FG3A': '3PA',
        'FG3_PCT': '3P%', 'FT_PCT': 'FT%'
    })

    # Metric Header Cards (Career Averages)
    st.subheader(f"Reference Player Career Averages: **{selected_player.title()}**")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Points", f"{ref_clean['PTS'].values[0]:.1f}")
    m2.metric("Rebounds", f"{ref_clean['REB'].values[0]:.1f}")
    m3.metric("Assists", f"{ref_clean['AST'].values[0]:.1f}")
    m4.metric("FG%", f"{ref_clean['FG%'].values[0] * 100:.1f}%" if ref_clean['FG%'].values[
                                                                       0] <= 1 else f"{ref_clean['FG%'].values[0]:.1f}%")
    m5.metric("3P%", f"{ref_clean['3P%'].values[0] * 100:.1f}%" if ref_clean['3P%'].values[
                                                                       0] <= 1 else f"{ref_clean['3P%'].values[0]:.1f}%")
    m6.metric("Games Played", f"{ref_clean['GP'].values[0]:.0f}")

    st.markdown("---")

    # SECTION 1: Plotly Horizontal Bar Chart for Similarity Scores
    st.subheader("🎯 Top 10 Match Confidence Scores")

    fig_sim = px.bar(
        top_10_clean.sort_values(by="similarity", ascending=True),
        x="Sim_Pct",
        y="Player",
        orientation="h",
        text="Sim_Pct",
        color="Sim_Pct",
        color_continuous_scale="Blues",
        labels={"Sim_Pct": "Match Similarity (%)", "Player": ""},
    )
    fig_sim.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_sim.update_layout(
        showlegend=False,
        xaxis_range=[0, 105],
        height=380,
        margin=dict(l=20, r=20, t=10, b=20),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    st.markdown("---")

    # SECTION 2: Head-to-Head Visual Matchup (Radar & Per-Game Overlay)
    st.subheader("⚔️ Head-to-Head Visual Comparison")

    selected_comp = st.selectbox("Select a Retired Player to Overlay:", options=top_10_clean['Player'].tolist())
    comp_data = top_10_clean[top_10_clean['Player'] == selected_comp].iloc[0]

    col_radar, col_bar = st.columns([1, 1])

    # Core volume stats to graph
    stats_to_plot = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
    ref_vals = [ref_clean[s].values[0] for s in stats_to_plot]
    comp_vals = [comp_data[s] for s in stats_to_plot]

    # Plot 1: Overlay Bar Chart
    with col_bar:
        st.markdown(f"##### **Career Stat Overlay: {selected_player.title()} vs {selected_comp}**")
        fig_grouped = go.Figure()
        fig_grouped.add_trace(go.Bar(
            x=stats_to_plot, y=ref_vals, name=selected_player.title(), marker_color='#1f77b4'
        ))
        fig_grouped.add_trace(go.Bar(
            x=stats_to_plot, y=comp_vals, name=f"{selected_comp} (Retired)", marker_color='#ff7f0e'
        ))
        fig_grouped.update_layout(barmode='group', height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_grouped, use_container_width=True)

    # Plot 2: Radar / Spider Chart Profile
    with col_radar:
        st.markdown("##### **Skill Profile Shape**")

        # Max-scaling metrics to prevent scale distortion
        max_vals = [max(r, c, 1) for r, c in zip(ref_vals, comp_vals)]
        ref_norm = [r / m for r, m in zip(ref_vals, max_vals)]
        comp_norm = [c / m for c, m in zip(comp_vals, max_vals)]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=ref_norm + [ref_norm[0]],
            theta=stats_to_plot + [stats_to_plot[0]],
            fill='toself',
            name=selected_player.title()
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=comp_norm + [comp_norm[0]],
            theta=stats_to_plot + [stats_to_plot[0]],
            fill='toself',
            name=f"{selected_comp} (Retired)"
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
            showlegend=True,
            height=350,
            margin=dict(l=30, r=30, t=30, b=10)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # SECTION 3: Raw Full Breakdown Table
    with st.expander("📊 View Complete Raw Data Table"):
        st.dataframe(top_10_clean.reset_index(drop=True), use_container_width=True)