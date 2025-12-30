"""
Pommelien Thijs State Space Analysis
=====================================
Modeling the synchronicities in Nouri's listening history.
Focus: 2025, especially October's 16-hour wrapped phenomenon.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Data paths
DATA_DIR = Path(r"C:\Users\Nouri\Desktop\easiest game of my life\mabrouk_spotify_dataset\Spotify Extended Streaming History")

def load_streaming_data():
    """Load all streaming history into a single DataFrame."""
    files = [
        "Streaming_History_Audio_2024-2025_3.json",
        "Streaming_History_Audio_2025_4.json"
    ]

    all_data = []
    for f in files:
        filepath = DATA_DIR / f
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
                all_data.extend(data)
                print(f"Loaded {len(data):,} records from {f}")

    df = pd.DataFrame(all_data)
    df['ts'] = pd.to_datetime(df['ts'])
    df['date'] = df['ts'].dt.date
    df['hour'] = df['ts'].dt.hour
    df['day_of_week'] = df['ts'].dt.day_name()
    df['week'] = df['ts'].dt.isocalendar().week
    df['month'] = df['ts'].dt.month
    df['year'] = df['ts'].dt.year
    df['minutes_played'] = df['ms_played'] / 60000
    df['hours_played'] = df['minutes_played'] / 60

    return df

def filter_pommelien(df):
    """Filter for Pommelien Thijs tracks."""
    mask = df['master_metadata_album_artist_name'].str.contains(
        'Pommelien', case=False, na=False
    )
    return df[mask].copy()

def analyze_state_space(pommelien_df):
    """Analyze the state space of Pommelien listening patterns."""
    # Group by date for daily patterns
    daily = pommelien_df.groupby('date').agg({
        'minutes_played': 'sum',
        'master_metadata_track_name': 'count',
        'hour': 'mean',
        'skipped': 'sum'
    }).reset_index()
    daily.columns = ['date', 'total_minutes', 'play_count', 'avg_hour', 'skips']
    daily['date'] = pd.to_datetime(daily['date'])

    # Calculate listening intensity (state variable)
    daily['intensity'] = daily['total_minutes'] / daily['total_minutes'].max()

    # Calculate momentum (rate of change)
    daily['momentum'] = daily['total_minutes'].diff().fillna(0)

    # Session clustering - time between plays
    pommelien_df_sorted = pommelien_df.sort_values('ts')
    pommelien_df_sorted['time_since_last'] = pommelien_df_sorted['ts'].diff().dt.total_seconds() / 3600

    return daily, pommelien_df_sorted

def create_visualizations(df, pommelien_df, daily, pommelien_sorted):
    """Create comprehensive visualizations."""

    # Filter for 2025
    df_2025 = df[df['year'] == 2025]
    pommelien_2025 = pommelien_df[pommelien_df['year'] == 2025]
    daily_2025 = daily[daily['date'].dt.year == 2025]

    print(f"\n=== 2025 POMMELIEN THIJS STATISTICS ===")
    total_hours_2025 = pommelien_2025['hours_played'].sum()
    print(f"Total listening time in 2025: {total_hours_2025:.1f} hours")

    # October stats
    october_df = pommelien_2025[pommelien_2025['month'] == 10]
    october_hours = october_df['hours_played'].sum()
    print(f"October 2025 listening time: {october_hours:.1f} hours")

    # Track breakdown
    track_counts = pommelien_2025.groupby('master_metadata_track_name').agg({
        'minutes_played': 'sum',
        'ts': 'count'
    }).sort_values('minutes_played', ascending=False)
    track_counts.columns = ['total_minutes', 'play_count']
    print(f"\n=== TOP POMMELIEN TRACKS 2025 ===")
    print(track_counts.head(10))

    # Create the mega visualization
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            '🎵 Daily Listening Intensity (State Space)',
            '⏰ Listening Hours Heatmap',
            '📈 Cumulative Hours Over Time',
            '🔄 Session Momentum (Phase Space)',
            '🎯 Track Distribution',
            '📊 Weekly Patterns',
            '🌊 Listening Flow (October Deep Dive)',
            '✨ Synchronicity Detection'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "heatmap"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "pie"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )

    # 1. Daily Listening Intensity
    if len(daily_2025) > 0:
        fig.add_trace(
            go.Scatter(
                x=daily_2025['date'],
                y=daily_2025['total_minutes'],
                mode='lines+markers',
                name='Daily Minutes',
                line=dict(color='#1DB954', width=2),
                marker=dict(size=8, color=daily_2025['intensity'], colorscale='Viridis'),
                hovertemplate='%{x}<br>%{y:.1f} minutes<extra></extra>'
            ),
            row=1, col=1
        )

    # 2. Hour-of-day heatmap
    hour_day = pommelien_2025.groupby(['day_of_week', 'hour'])['minutes_played'].sum().unstack(fill_value=0)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hour_day = hour_day.reindex(day_order)

    fig.add_trace(
        go.Heatmap(
            z=hour_day.values,
            x=list(range(24)),
            y=day_order,
            colorscale='Greens',
            showscale=True,
            hovertemplate='Hour: %{x}<br>Day: %{y}<br>Minutes: %{z:.1f}<extra></extra>'
        ),
        row=1, col=2
    )

    # 3. Cumulative listening over time
    pommelien_2025_sorted = pommelien_2025.sort_values('ts')
    pommelien_2025_sorted['cumulative_hours'] = pommelien_2025_sorted['hours_played'].cumsum()

    fig.add_trace(
        go.Scatter(
            x=pommelien_2025_sorted['ts'],
            y=pommelien_2025_sorted['cumulative_hours'],
            mode='lines',
            name='Cumulative Hours',
            fill='tozeroy',
            line=dict(color='#1DB954', width=2),
            hovertemplate='%{x}<br>%{y:.2f} total hours<extra></extra>'
        ),
        row=2, col=1
    )

    # 4. Phase Space - Intensity vs Momentum
    if len(daily_2025) > 1:
        fig.add_trace(
            go.Scatter(
                x=daily_2025['intensity'],
                y=daily_2025['momentum'],
                mode='markers+lines',
                name='Phase Space',
                marker=dict(
                    size=10,
                    color=np.arange(len(daily_2025)),
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title='Day #', x=1.02)
                ),
                line=dict(color='rgba(150,150,150,0.3)', width=1),
                hovertemplate='Intensity: %{x:.2f}<br>Momentum: %{y:.1f}<extra></extra>'
            ),
            row=2, col=2
        )

    # 5. Track Distribution (Pie)
    top_tracks = track_counts.head(8)
    fig.add_trace(
        go.Pie(
            labels=top_tracks.index,
            values=top_tracks['total_minutes'],
            hole=0.4,
            marker=dict(colors=px.colors.sequential.Greens_r),
            textinfo='percent+label',
            textposition='outside'
        ),
        row=3, col=1
    )

    # 6. Weekly patterns
    weekly = pommelien_2025.groupby('week')['hours_played'].sum().reset_index()
    fig.add_trace(
        go.Bar(
            x=weekly['week'],
            y=weekly['hours_played'],
            name='Weekly Hours',
            marker_color='#1DB954',
            hovertemplate='Week %{x}<br>%{y:.2f} hours<extra></extra>'
        ),
        row=3, col=2
    )

    # 7. October Deep Dive - Flow visualization
    if len(october_df) > 0:
        october_sorted = october_df.sort_values('ts')
        fig.add_trace(
            go.Scatter(
                x=october_sorted['ts'],
                y=october_sorted['minutes_played'],
                mode='markers',
                name='October Sessions',
                marker=dict(
                    size=october_sorted['minutes_played'] / october_sorted['minutes_played'].max() * 20 + 5,
                    color=october_sorted['hour'],
                    colorscale='Sunset',
                    showscale=True,
                    colorbar=dict(title='Hour', x=0.45)
                ),
                hovertemplate='%{x}<br>%{y:.1f} minutes<extra></extra>'
            ),
            row=4, col=1
        )

    # 8. Synchronicity Detection - Listening bursts
    # Detect sessions with unusually high engagement
    if len(daily_2025) > 0:
        mean_minutes = daily_2025['total_minutes'].mean()
        std_minutes = daily_2025['total_minutes'].std()
        daily_2025['z_score'] = (daily_2025['total_minutes'] - mean_minutes) / (std_minutes + 0.001)
        synchronicities = daily_2025[daily_2025['z_score'] > 1.5]

        fig.add_trace(
            go.Scatter(
                x=daily_2025['date'],
                y=daily_2025['z_score'],
                mode='lines',
                name='Baseline',
                line=dict(color='gray', width=1),
                hovertemplate='%{x}<br>Z-score: %{y:.2f}<extra></extra>'
            ),
            row=4, col=2
        )

        if len(synchronicities) > 0:
            fig.add_trace(
                go.Scatter(
                    x=synchronicities['date'],
                    y=synchronicities['z_score'],
                    mode='markers',
                    name='Synchronicities',
                    marker=dict(size=15, color='gold', symbol='star'),
                    hovertemplate='SYNC: %{x}<br>Z-score: %{y:.2f}<extra></extra>'
                ),
                row=4, col=2
            )

        # Add threshold line as a trace instead
        fig.add_trace(
            go.Scatter(
                x=[daily_2025['date'].min(), daily_2025['date'].max()],
                y=[1.5, 1.5],
                mode='lines',
                name='Sync Threshold',
                line=dict(color='red', dash='dash', width=2),
                hoverinfo='skip'
            ),
            row=4, col=2
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text='🎵 POMMELIEN THIJS STATE SPACE ANALYSIS 🎵<br><sup>Nouri\'s 2025 Listening Synchronicities</sup>',
            font=dict(size=24, color='#1DB954')
        ),
        showlegend=False,
        height=1400,
        width=1400,
        paper_bgcolor='#121212',
        plot_bgcolor='#1a1a1a',
        font=dict(color='white'),
        hoverlabel=dict(bgcolor='#1DB954', font_size=12)
    )

    # Update axes colors
    fig.update_xaxes(gridcolor='#333', zerolinecolor='#333')
    fig.update_yaxes(gridcolor='#333', zerolinecolor='#333')

    return fig, track_counts, daily_2025

def main():
    print("🎵 Loading Spotify streaming history...")
    df = load_streaming_data()
    print(f"\nTotal records: {len(df):,}")

    print("\n🔍 Filtering for Pommelien Thijs...")
    pommelien_df = filter_pommelien(df)
    print(f"Pommelien tracks found: {len(pommelien_df):,}")

    if len(pommelien_df) == 0:
        print("No Pommelien Thijs tracks found! Let me check artist names...")
        artists = df['master_metadata_album_artist_name'].dropna().unique()
        similar = [a for a in artists if 'pomm' in a.lower() or 'thijs' in a.lower()]
        print(f"Similar artists: {similar}")
        return

    print("\n📊 Analyzing state space...")
    daily, pommelien_sorted = analyze_state_space(pommelien_df)

    print("\n🎨 Creating visualizations...")
    fig, track_counts, daily_2025 = create_visualizations(df, pommelien_df, daily, pommelien_sorted)

    # Save and open
    output_path = Path(r"C:\Users\Nouri\Documents\GitHub\Nouri-Mabrouk\experiments\pommelien_state_space.html")
    fig.write_html(str(output_path), include_plotlyjs=True)
    print(f"\n✅ Visualization saved to: {output_path}")

    # Open in browser
    webbrowser.open(f'file:///{output_path}')
    print("🚀 Opening in browser...")

    # Print summary statistics
    if len(daily_2025) > 0:
        print("\n" + "="*50)
        print("📈 STATE SPACE SUMMARY")
        print("="*50)
        total_2025_hours = pommelien_df[pommelien_df['year'] == 2025]['hours_played'].sum()
        print(f"Total 2025 Pommelien hours: {total_2025_hours:.1f}")
        print(f"Days with Pommelien: {len(daily_2025)}")
        print(f"Average daily intensity: {daily_2025['total_minutes'].mean():.1f} minutes")
        print(f"Peak listening day: {daily_2025.loc[daily_2025['total_minutes'].idxmax(), 'date']}")
        print(f"Peak minutes: {daily_2025['total_minutes'].max():.1f}")

        # Synchronicity count
        mean_m = daily_2025['total_minutes'].mean()
        std_m = daily_2025['total_minutes'].std()
        sync_count = len(daily_2025[(daily_2025['total_minutes'] - mean_m) / (std_m + 0.001) > 1.5])
        print(f"\n✨ Synchronicity events detected: {sync_count}")

if __name__ == "__main__":
    main()
