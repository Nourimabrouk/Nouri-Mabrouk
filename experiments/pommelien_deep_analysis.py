"""
POMMELIEN THIJS - DEEP STATE SPACE ANALYSIS
============================================
Real data science. No fluff.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DATA_DIR = Path(r"C:\Users\Nouri\Desktop\easiest game of my life\mabrouk_spotify_dataset\Spotify Extended Streaming History")

def load_data():
    files = ["Streaming_History_Audio_2024-2025_3.json", "Streaming_History_Audio_2025_4.json"]
    all_data = []
    for f in files:
        with open(DATA_DIR / f, 'r', encoding='utf-8') as file:
            all_data.extend(json.load(file))

    df = pd.DataFrame(all_data)
    df['ts'] = pd.to_datetime(df['ts'])
    df['minutes'] = df['ms_played'] / 60000
    df['hours'] = df['minutes'] / 60
    df['date'] = df['ts'].dt.date
    df['hour'] = df['ts'].dt.hour
    df['dow'] = df['ts'].dt.dayofweek
    df['week'] = df['ts'].dt.isocalendar().week
    df['month'] = df['ts'].dt.month
    df['year'] = df['ts'].dt.year
    return df

def get_pommelien(df):
    mask = df['master_metadata_album_artist_name'].str.contains('Pommelien', case=False, na=False)
    return df[mask].copy()

def detect_sessions(pdf):
    """Detect listening sessions based on time gaps."""
    pdf = pdf.sort_values('ts').copy()
    pdf['time_gap'] = pdf['ts'].diff().dt.total_seconds() / 60  # minutes
    pdf['new_session'] = (pdf['time_gap'] > 30) | (pdf['time_gap'].isna())  # 30 min gap = new session
    pdf['session_id'] = pdf['new_session'].cumsum()
    return pdf

def analyze_sessions(pdf):
    """Analyze session characteristics."""
    sessions = pdf.groupby('session_id').agg({
        'ts': ['min', 'max'],
        'minutes': 'sum',
        'master_metadata_track_name': 'count',
        'hour': 'first',
        'dow': 'first',
        'date': 'first'
    })
    sessions.columns = ['start', 'end', 'duration_min', 'track_count', 'start_hour', 'dow', 'date']
    sessions['duration_hours'] = sessions['duration_min'] / 60
    return sessions

def find_synchronicities(daily_df):
    """Find statistically anomalous days."""
    mean = daily_df['minutes'].mean()
    std = daily_df['minutes'].std()
    daily_df['z_score'] = (daily_df['minutes'] - mean) / std

    # Find peaks in the time series
    peaks, properties = find_peaks(daily_df['minutes'].values, height=mean + std, distance=3)

    # Days > 2 std from mean
    sync_days = daily_df[daily_df['z_score'] > 2].copy()

    return sync_days, peaks, daily_df

def cluster_listening_patterns(sessions):
    """Cluster sessions by characteristics."""
    features = sessions[['duration_min', 'track_count', 'start_hour']].copy()
    features = features.dropna()

    if len(features) < 10:
        return None, features

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    clustering = DBSCAN(eps=0.8, min_samples=3).fit(X)
    features['cluster'] = clustering.labels_

    return clustering, features

def build_state_space(daily_df):
    """Build proper state space representation."""
    # State variables: intensity (normalized minutes), velocity (diff), acceleration (diff of diff)
    daily_df = daily_df.copy()
    daily_df['intensity'] = daily_df['minutes'] / daily_df['minutes'].max()
    daily_df['velocity'] = daily_df['intensity'].diff().fillna(0)
    daily_df['acceleration'] = daily_df['velocity'].diff().fillna(0)

    # Rolling statistics for regime detection
    daily_df['rolling_mean'] = daily_df['minutes'].rolling(7, min_periods=1).mean()
    daily_df['rolling_std'] = daily_df['minutes'].rolling(7, min_periods=1).std().fillna(0)

    return daily_df

def create_visualization(pdf, sessions, daily_state, sync_days, cluster_features, october_df):
    """Create the real visualization."""

    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'State Space Trajectory (Intensity vs Velocity)',
            'Phase Portrait (Velocity vs Acceleration)',
            'Daily Listening Time Series',
            'Session Duration Distribution',
            'Listening Hour Patterns',
            'October 2025 Deep Dive',
            'Track Play Frequency',
            'Synchronicity Events',
            'Cumulative Listening Trajectory'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.08
    )

    # 1. State Space: Intensity vs Velocity
    colors = np.arange(len(daily_state))
    fig.add_trace(
        go.Scatter(
            x=daily_state['intensity'],
            y=daily_state['velocity'],
            mode='lines+markers',
            marker=dict(size=6, color=colors, colorscale='Viridis', showscale=False),
            line=dict(color='rgba(100,100,100,0.3)', width=1),
            text=daily_state['date'].astype(str),
            hovertemplate='Date: %{text}<br>Intensity: %{x:.3f}<br>Velocity: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )

    # 2. Phase Portrait: Velocity vs Acceleration
    fig.add_trace(
        go.Scatter(
            x=daily_state['velocity'],
            y=daily_state['acceleration'],
            mode='lines+markers',
            marker=dict(size=6, color=colors, colorscale='Plasma', showscale=False),
            line=dict(color='rgba(100,100,100,0.3)', width=1),
            hovertemplate='Velocity: %{x:.3f}<br>Acceleration: %{y:.3f}<extra></extra>'
        ),
        row=1, col=2
    )

    # 3. Time Series with regime bands
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(daily_state['date']),
            y=daily_state['minutes'],
            mode='lines',
            name='Daily Minutes',
            line=dict(color='#1DB954', width=2)
        ),
        row=1, col=3
    )
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(daily_state['date']),
            y=daily_state['rolling_mean'],
            mode='lines',
            name='7-day MA',
            line=dict(color='orange', width=2, dash='dash')
        ),
        row=1, col=3
    )

    # Mark synchronicities on time series
    if len(sync_days) > 0:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(sync_days['date']),
                y=sync_days['minutes'],
                mode='markers',
                marker=dict(size=15, color='gold', symbol='star', line=dict(color='black', width=1)),
                name='Sync Events',
                hovertemplate='SYNC: %{x}<br>%{y:.0f} min<extra></extra>'
            ),
            row=1, col=3
        )

    # 4. Session Duration Histogram
    fig.add_trace(
        go.Histogram(
            x=sessions['duration_min'],
            nbinsx=50,
            marker_color='#1DB954',
            name='Session Duration'
        ),
        row=2, col=1
    )

    # 5. Hour of Day Pattern
    hour_counts = pdf.groupby('hour')['minutes'].sum()
    fig.add_trace(
        go.Bar(
            x=list(range(24)),
            y=[hour_counts.get(h, 0) for h in range(24)],
            marker_color=['#1DB954' if 6 <= h <= 22 else '#666' for h in range(24)],
            name='By Hour'
        ),
        row=2, col=2
    )

    # 6. October Deep Dive
    if len(october_df) > 0:
        oct_daily = october_df.groupby('date')['minutes'].sum().reset_index()
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(oct_daily['date']),
                y=oct_daily['minutes'],
                mode='lines+markers',
                fill='tozeroy',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=8),
                hovertemplate='%{x}<br>%{y:.0f} minutes<extra></extra>'
            ),
            row=2, col=3
        )

    # 7. Track Frequency
    track_counts = pdf['master_metadata_track_name'].value_counts().head(15)
    fig.add_trace(
        go.Bar(
            x=track_counts.values,
            y=track_counts.index,
            orientation='h',
            marker_color='#1DB954'
        ),
        row=3, col=1
    )

    # 8. Synchronicity scatter - showing the exceptional days
    if len(sync_days) > 0:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(sync_days['date']),
                y=sync_days['z_score'],
                mode='markers+text',
                marker=dict(size=sync_days['minutes']/20, color='gold', symbol='star'),
                text=sync_days['minutes'].round(0).astype(int).astype(str) + ' min',
                textposition='top center',
                hovertemplate='%{x}<br>Z-score: %{y:.2f}<extra></extra>'
            ),
            row=3, col=2
        )

    # 9. Cumulative trajectory
    pdf_sorted = pdf.sort_values('ts')
    pdf_sorted['cumulative'] = pdf_sorted['hours'].cumsum()
    fig.add_trace(
        go.Scatter(
            x=pdf_sorted['ts'],
            y=pdf_sorted['cumulative'],
            mode='lines',
            fill='tozeroy',
            line=dict(color='#1DB954', width=2),
            hovertemplate='%{x}<br>%{y:.1f} total hours<extra></extra>'
        ),
        row=3, col=3
    )

    fig.update_layout(
        title=dict(
            text='POMMELIEN THIJS - STATE SPACE DYNAMICS',
            font=dict(size=20, color='white')
        ),
        showlegend=False,
        height=1200,
        width=1600,
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#1a1a1a',
        font=dict(color='white', size=10)
    )

    fig.update_xaxes(gridcolor='#333', zerolinecolor='#444')
    fig.update_yaxes(gridcolor='#333', zerolinecolor='#444')

    return fig

def main():
    print("Loading data...")
    df = load_data()

    print("Filtering Pommelien...")
    pdf = get_pommelien(df)
    pdf_2025 = pdf[pdf['year'] == 2025].copy()

    print(f"Total Pommelien plays 2025: {len(pdf_2025)}")
    print(f"Total hours 2025: {pdf_2025['hours'].sum():.1f}")

    print("\nDetecting sessions...")
    pdf_2025 = detect_sessions(pdf_2025)
    sessions = analyze_sessions(pdf_2025)
    print(f"Sessions detected: {len(sessions)}")
    print(f"Avg session duration: {sessions['duration_min'].mean():.1f} min")
    print(f"Max session duration: {sessions['duration_min'].max():.1f} min ({sessions['duration_min'].max()/60:.1f} hours)")

    print("\nBuilding daily state space...")
    daily = pdf_2025.groupby('date')['minutes'].sum().reset_index()
    daily_state = build_state_space(daily)

    print("\nFinding synchronicities...")
    sync_days, peaks, daily_state = find_synchronicities(daily_state)
    print(f"Synchronicity events (z > 2): {len(sync_days)}")

    if len(sync_days) > 0:
        print("\n=== SYNCHRONICITY DAYS ===")
        for _, row in sync_days.iterrows():
            print(f"  {row['date']}: {row['minutes']:.0f} min ({row['minutes']/60:.1f} hrs) | z={row['z_score']:.2f}")

    print("\nClustering sessions...")
    clustering, cluster_features = cluster_listening_patterns(sessions)
    if clustering is not None:
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        print(f"Session clusters found: {n_clusters}")

    # October analysis
    october_df = pdf_2025[pdf_2025['month'] == 10]
    oct_hours = october_df['hours'].sum()
    print(f"\n=== OCTOBER 2025 ===")
    print(f"Total hours: {oct_hours:.1f}")
    print(f"Days active: {october_df['date'].nunique()}")
    print(f"Avg per active day: {oct_hours / october_df['date'].nunique():.1f} hours")

    # Peak day analysis
    peak_day = daily_state.loc[daily_state['minutes'].idxmax()]
    print(f"\n=== PEAK DAY ===")
    print(f"Date: {peak_day['date']}")
    print(f"Minutes: {peak_day['minutes']:.0f} ({peak_day['minutes']/60:.1f} hours)")

    # What was playing on peak day?
    peak_tracks = pdf_2025[pdf_2025['date'] == peak_day['date']]['master_metadata_track_name'].value_counts()
    print("Top tracks that day:")
    for track, count in peak_tracks.head(5).items():
        print(f"  {track}: {count} plays")

    print("\nGenerating visualization...")
    fig = create_visualization(pdf_2025, sessions, daily_state, sync_days, cluster_features, october_df)

    output = Path(r"C:\Users\Nouri\Documents\GitHub\Nouri-Mabrouk\experiments\pommelien_deep.html")
    fig.write_html(str(output))
    print(f"\nSaved: {output}")
    webbrowser.open(f'file:///{output}')

    # Return key findings
    return {
        'total_hours_2025': pdf_2025['hours'].sum(),
        'october_hours': oct_hours,
        'sync_days': sync_days,
        'peak_day': peak_day,
        'sessions': len(sessions),
        'avg_session_min': sessions['duration_min'].mean()
    }

if __name__ == "__main__":
    findings = main()
