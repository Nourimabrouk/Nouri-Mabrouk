"""
Artist Context Analysis: Pommelien Thijs Ecosystem
===================================================
Analyzes listening patterns around Pommelien Thijs in 2025:
- Before/after artist transitions
- Warm-up and cool-down patterns
- Comparison with other top artists
- Pommelien-only days vs mixed listening
"""

import json
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Data path
DATA_PATH = r"C:\Users\Nouri\Desktop\easiest game of my life\mabrouk_spotify_dataset\Spotify Extended Streaming History"

def load_2025_data():
    """Load all 2025 streaming data"""
    files = [
        "Streaming_History_Audio_2024-2025_3.json",
        "Streaming_History_Audio_2025_4.json"
    ]

    all_streams = []
    for f in files:
        path = os.path.join(DATA_PATH, f)
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            all_streams.extend(data)

    # Filter for 2025 only and sort by timestamp
    streams_2025 = []
    for stream in all_streams:
        ts = stream.get('ts', '')
        if ts.startswith('2025'):
            artist = stream.get('master_metadata_album_artist_name')
            if artist:  # Only include streams with artist info
                streams_2025.append({
                    'ts': datetime.fromisoformat(ts.replace('Z', '+00:00')),
                    'artist': artist,
                    'track': stream.get('master_metadata_track_name', 'Unknown'),
                    'ms_played': stream.get('ms_played', 0),
                    'skipped': stream.get('skipped', False),
                    'reason_start': stream.get('reason_start', ''),
                    'reason_end': stream.get('reason_end', '')
                })

    # Sort by timestamp
    streams_2025.sort(key=lambda x: x['ts'])
    return streams_2025

def analyze_context(streams, target_artist="Pommelien Thijs"):
    """Analyze what artists are played before/after the target artist"""

    before_artists = Counter()
    after_artists = Counter()
    pommelien_sessions = []

    current_session = None
    session_threshold = timedelta(minutes=30)  # Gap to define new session

    for i, stream in enumerate(streams):
        artist = stream['artist']

        if artist == target_artist:
            # Track artists before Pommelien
            if i > 0:
                prev = streams[i-1]
                time_gap = stream['ts'] - prev['ts']
                if time_gap < session_threshold and prev['artist'] != target_artist:
                    before_artists[prev['artist']] += 1

            # Start or continue session
            if current_session is None:
                current_session = {
                    'start': stream['ts'],
                    'end': stream['ts'],
                    'tracks': [stream['track']],
                    'ms_played': stream['ms_played'],
                    'before_artist': streams[i-1]['artist'] if i > 0 else None,
                    'after_artist': None
                }
            else:
                current_session['end'] = stream['ts']
                current_session['tracks'].append(stream['track'])
                current_session['ms_played'] += stream['ms_played']
        else:
            # Track artists after Pommelien
            if i > 0 and streams[i-1]['artist'] == target_artist:
                time_gap = stream['ts'] - streams[i-1]['ts']
                if time_gap < session_threshold:
                    after_artists[artist] += 1
                    if current_session:
                        current_session['after_artist'] = artist

            # End session
            if current_session:
                pommelien_sessions.append(current_session)
                current_session = None

    # Don't forget the last session
    if current_session:
        pommelien_sessions.append(current_session)

    return before_artists, after_artists, pommelien_sessions

def analyze_daily_patterns(streams, target_artist="Pommelien Thijs"):
    """Analyze daily listening patterns"""

    daily_stats = defaultdict(lambda: {
        'total_artists': set(),
        'pommelien_ms': 0,
        'total_ms': 0,
        'pommelien_tracks': 0,
        'total_tracks': 0
    })

    for stream in streams:
        date = stream['ts'].date()
        daily_stats[date]['total_artists'].add(stream['artist'])
        daily_stats[date]['total_ms'] += stream['ms_played']
        daily_stats[date]['total_tracks'] += 1

        if stream['artist'] == target_artist:
            daily_stats[date]['pommelien_ms'] += stream['ms_played']
            daily_stats[date]['pommelien_tracks'] += 1

    # Categorize days
    pommelien_only_days = []
    mixed_days = []
    no_pommelien_days = []

    for date, stats in sorted(daily_stats.items()):
        stats['artist_count'] = len(stats['total_artists'])
        if stats['pommelien_tracks'] > 0:
            if stats['artist_count'] == 1:
                pommelien_only_days.append((date, stats))
            else:
                mixed_days.append((date, stats))
        else:
            no_pommelien_days.append((date, stats))

    return pommelien_only_days, mixed_days, no_pommelien_days, daily_stats

def get_top_artists(streams, n=15):
    """Get top N artists by listening time"""
    artist_time = Counter()
    for stream in streams:
        artist_time[stream['artist']] += stream['ms_played']

    return artist_time.most_common(n)

def create_visualization(before_artists, after_artists, sessions,
                         pommelien_only_days, mixed_days, daily_stats,
                         top_artists, streams):
    """Create comprehensive visualization"""

    # Create figure with subplots
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Artists Played BEFORE Pommelien Thijs',
            'Artists Played AFTER Pommelien Thijs',
            'Top Artists by Listening Time (2025)',
            'Pommelien Share of Daily Listening',
            'Daily Listening Pattern: Pommelien vs Others',
            'Artist Transition Network',
            'Session Length Distribution',
            'Time of Day Analysis'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "domain"}],
            [{"type": "histogram"}, {"type": "bar"}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )

    # Color scheme
    pommelien_color = '#FF6B9D'  # Pink
    other_color = '#4ECDC4'  # Teal
    accent_color = '#FFE66D'  # Yellow

    # 1. Before Artists
    top_before = before_artists.most_common(10)
    if top_before:
        artists, counts = zip(*top_before)
        colors = [pommelien_color if a == 'Pommelien Thijs' else other_color for a in artists]
        fig.add_trace(
            go.Bar(x=list(artists), y=list(counts), marker_color=colors, name='Before'),
            row=1, col=1
        )

    # 2. After Artists
    top_after = after_artists.most_common(10)
    if top_after:
        artists, counts = zip(*top_after)
        colors = [pommelien_color if a == 'Pommelien Thijs' else other_color for a in artists]
        fig.add_trace(
            go.Bar(x=list(artists), y=list(counts), marker_color=colors, name='After'),
            row=1, col=2
        )

    # 3. Top Artists Comparison
    if top_artists:
        artists, times = zip(*top_artists)
        hours = [t / 3600000 for t in times]  # Convert to hours
        colors = [pommelien_color if a == 'Pommelien Thijs' else other_color for a in artists]
        fig.add_trace(
            go.Bar(x=list(artists), y=hours, marker_color=colors, name='Hours'),
            row=2, col=1
        )

    # 4. Pommelien Share of Daily Listening
    dates = sorted(daily_stats.keys())
    pommelien_pct = []
    for d in dates:
        stats = daily_stats[d]
        pct = (stats['pommelien_ms'] / stats['total_ms'] * 100) if stats['total_ms'] > 0 else 0
        pommelien_pct.append(pct)

    fig.add_trace(
        go.Scatter(
            x=dates, y=pommelien_pct,
            mode='lines+markers',
            line=dict(color=pommelien_color, width=2),
            marker=dict(size=4),
            name='Pommelien %',
            fill='tozeroy',
            fillcolor='rgba(255, 107, 157, 0.3)'
        ),
        row=2, col=2
    )

    # 5. Daily Listening Pattern
    pommelien_daily = [daily_stats[d]['pommelien_ms'] / 3600000 for d in dates]
    other_daily = [(daily_stats[d]['total_ms'] - daily_stats[d]['pommelien_ms']) / 3600000 for d in dates]

    fig.add_trace(
        go.Scatter(
            x=dates, y=pommelien_daily,
            mode='lines',
            name='Pommelien',
            line=dict(color=pommelien_color, width=2),
            stackgroup='one'
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=other_daily,
            mode='lines',
            name='Other Artists',
            line=dict(color=other_color, width=2),
            stackgroup='one'
        ),
        row=3, col=1
    )

    # 6. Transition Network as pie chart (showing before/after distribution)
    transition_data = {
        'Before Pommelien': sum(before_artists.values()),
        'After Pommelien': sum(after_artists.values()),
    }
    fig.add_trace(
        go.Pie(
            labels=list(transition_data.keys()),
            values=list(transition_data.values()),
            marker=dict(colors=[other_color, accent_color]),
            hole=0.4
        ),
        row=3, col=2
    )

    # 7. Session Length Distribution
    session_lengths = [(s['ms_played'] / 60000) for s in sessions]  # In minutes
    fig.add_trace(
        go.Histogram(
            x=session_lengths,
            nbinsx=20,
            marker_color=pommelien_color,
            name='Session Minutes'
        ),
        row=4, col=1
    )

    # 8. Time of Day Analysis
    hour_counts = Counter()
    for stream in streams:
        if stream['artist'] == 'Pommelien Thijs':
            hour_counts[stream['ts'].hour] += 1

    hours = list(range(24))
    counts = [hour_counts.get(h, 0) for h in hours]
    hour_labels = [f'{h:02d}:00' for h in hours]

    fig.add_trace(
        go.Bar(
            x=hour_labels,
            y=counts,
            marker_color=pommelien_color,
            name='Plays by Hour'
        ),
        row=4, col=2
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>Pommelien Thijs: Artist Ecosystem Analysis (2025)</b>',
            font=dict(size=24, color='white'),
            x=0.5
        ),
        showlegend=False,
        height=1600,
        width=1400,
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e',
        font=dict(color='white', size=11)
    )

    # Update all axes
    fig.update_xaxes(
        tickangle=45,
        gridcolor='rgba(255,255,255,0.1)',
        tickfont=dict(size=9)
    )
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')

    # Add annotations for key insights
    pommelien_hours = sum(t for a, t in top_artists if a == 'Pommelien Thijs') / 3600000
    pommelien_rank = next((i+1 for i, (a, _) in enumerate(top_artists) if a == 'Pommelien Thijs'), 'N/A')

    # Count special days
    pommelien_only_count = len(pommelien_only_days)
    mixed_count = len(mixed_days)

    insights_text = f"""
    <b>Key Insights:</b><br>
    - Pommelien Thijs Rank: #{pommelien_rank} with {pommelien_hours:.1f} hours<br>
    - Pommelien-only days: {pommelien_only_count}<br>
    - Mixed listening days: {mixed_count}<br>
    - Total Pommelien sessions: {len(sessions)}<br>
    - Most common warm-up: {before_artists.most_common(1)[0][0] if before_artists else 'N/A'}<br>
    - Most common cool-down: {after_artists.most_common(1)[0][0] if after_artists else 'N/A'}
    """

    fig.add_annotation(
        text=insights_text,
        xref="paper", yref="paper",
        x=1.02, y=0.98,
        showarrow=False,
        font=dict(size=12, color='white'),
        align='left',
        bgcolor='rgba(0,0,0,0.5)',
        bordercolor=pommelien_color,
        borderwidth=2,
        borderpad=10
    )

    return fig

def create_network_visualization(before_artists, after_artists, top_artists):
    """Create a separate network/sankey visualization"""

    # Build transition data for Sankey
    sources = []
    targets = []
    values = []
    labels = []
    colors = []

    pommelien_color = '#FF6B9D'
    other_color = '#4ECDC4'

    # Get unique artists
    all_artists = set(before_artists.keys()) | set(after_artists.keys()) | {'Pommelien Thijs'}
    artist_list = list(all_artists)
    artist_idx = {a: i for i, a in enumerate(artist_list)}

    # Create labels and colors
    for artist in artist_list:
        labels.append(artist)
        colors.append(pommelien_color if artist == 'Pommelien Thijs' else other_color)

    # Add before -> Pommelien flows
    for artist, count in before_artists.most_common(15):
        sources.append(artist_idx[artist])
        targets.append(artist_idx['Pommelien Thijs'])
        values.append(count)

    # Add Pommelien -> after flows
    for artist, count in after_artists.most_common(15):
        sources.append(artist_idx['Pommelien Thijs'])
        targets.append(artist_idx[artist])
        values.append(count)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='white', width=0.5),
            label=labels,
            color=colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color='rgba(255, 107, 157, 0.4)'
        )
    )])

    fig.update_layout(
        title=dict(
            text='<b>Artist Flow Around Pommelien Thijs</b><br><sub>Artists before (left) and after (right) Pommelien sessions</sub>',
            font=dict(size=20, color='white'),
            x=0.5
        ),
        height=800,
        width=1200,
        paper_bgcolor='#1a1a2e',
        font=dict(color='white', size=12)
    )

    return fig

def main():
    print("Loading 2025 streaming data...")
    streams = load_2025_data()
    print(f"Loaded {len(streams)} streams from 2025")

    print("\nAnalyzing listening context around Pommelien Thijs...")
    before_artists, after_artists, sessions = analyze_context(streams)

    print("\nAnalyzing daily patterns...")
    pommelien_only_days, mixed_days, no_pommelien_days, daily_stats = analyze_daily_patterns(streams)

    print("\nGetting top artists...")
    top_artists = get_top_artists(streams, n=15)

    # Print summary
    print("\n" + "="*60)
    print("POMMELIEN THIJS ECOSYSTEM ANALYSIS (2025)")
    print("="*60)

    print("\n--- TOP 10 WARM-UP ARTISTS (Before Pommelien) ---")
    for artist, count in before_artists.most_common(10):
        print(f"  {artist}: {count} times")

    print("\n--- TOP 10 COOL-DOWN ARTISTS (After Pommelien) ---")
    for artist, count in after_artists.most_common(10):
        print(f"  {artist}: {count} times")

    print("\n--- TOP 15 ARTISTS BY LISTENING TIME ---")
    for i, (artist, ms) in enumerate(top_artists, 1):
        hours = ms / 3600000
        marker = " <-- POMMELIEN" if artist == 'Pommelien Thijs' else ""
        print(f"  {i}. {artist}: {hours:.2f} hours{marker}")

    print(f"\n--- LISTENING DAY PATTERNS ---")
    print(f"  Days with ONLY Pommelien: {len(pommelien_only_days)}")
    print(f"  Days with Pommelien + others: {len(mixed_days)}")
    print(f"  Days without Pommelien: {len(no_pommelien_days)}")

    if pommelien_only_days:
        print("\n  Pommelien-only days:")
        for date, stats in pommelien_only_days[:5]:
            mins = stats['pommelien_ms'] / 60000
            print(f"    {date}: {mins:.1f} minutes, {stats['pommelien_tracks']} tracks")

    print(f"\n--- SESSION STATISTICS ---")
    print(f"  Total Pommelien sessions: {len(sessions)}")
    if sessions:
        avg_duration = sum(s['ms_played'] for s in sessions) / len(sessions) / 60000
        max_duration = max(s['ms_played'] for s in sessions) / 60000
        avg_tracks = sum(len(s['tracks']) for s in sessions) / len(sessions)
        print(f"  Average session duration: {avg_duration:.1f} minutes")
        print(f"  Longest session: {max_duration:.1f} minutes")
        print(f"  Average tracks per session: {avg_tracks:.1f}")

    # Create visualizations
    print("\nCreating visualizations...")

    # Main dashboard
    fig1 = create_visualization(
        before_artists, after_artists, sessions,
        pommelien_only_days, mixed_days, daily_stats,
        top_artists, streams
    )

    # Network visualization
    fig2 = create_network_visualization(before_artists, after_artists, top_artists)

    # Combine into single HTML
    output_path = r"C:\Users\Nouri\Documents\GitHub\Nouri-Mabrouk\experiments\artist_context.html"

    # Create combined HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Pommelien Thijs Artist Ecosystem (2025)</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white;
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }}
        .header {{
            text-align: center;
            padding: 30px;
            margin-bottom: 20px;
        }}
        h1 {{
            font-size: 2.5em;
            margin: 0;
            background: linear-gradient(45deg, #FF6B9D, #FFE66D);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .subtitle {{
            color: #aaa;
            margin-top: 10px;
            font-size: 1.2em;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px auto;
            max-width: 1400px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255, 107, 157, 0.3);
        }}
        .stat-value {{
            font-size: 2.5em;
            color: #FF6B9D;
            font-weight: bold;
        }}
        .stat-label {{
            color: #aaa;
            margin-top: 5px;
        }}
        .section {{
            max-width: 1400px;
            margin: 40px auto;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
        }}
        .section-title {{
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #4ECDC4;
            border-bottom: 2px solid #4ECDC4;
            padding-bottom: 10px;
        }}
        .insight-box {{
            background: rgba(255, 107, 157, 0.1);
            border-left: 4px solid #FF6B9D;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 10px 10px 0;
        }}
        .chart-container {{
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Pommelien Thijs: Artist Ecosystem</h1>
        <div class="subtitle">Analyzing listening context and patterns in 2025</div>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{len(sessions)}</div>
            <div class="stat-label">Pommelien Sessions</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(before_artists)}</div>
            <div class="stat-label">Warm-up Artists</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(after_artists)}</div>
            <div class="stat-label">Cool-down Artists</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(pommelien_only_days)}</div>
            <div class="stat-label">Pommelien-Only Days</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(mixed_days)}</div>
            <div class="stat-label">Mixed Listening Days</div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Artist Transition Flow</div>
        <div class="insight-box">
            <strong>How to read:</strong> Left side shows artists played <em>before</em> Pommelien,
            right side shows artists played <em>after</em>. Flow thickness indicates frequency.
        </div>
        <div class="chart-container" id="sankey"></div>
    </div>

    <div class="section">
        <div class="section-title">Comprehensive Dashboard</div>
        <div class="chart-container" id="dashboard"></div>
    </div>

    <div class="section">
        <div class="section-title">Key Patterns</div>
        <div class="insight-box">
            <strong>Top Warm-up Artist:</strong> {before_artists.most_common(1)[0][0] if before_artists else 'N/A'}
            ({before_artists.most_common(1)[0][1] if before_artists else 0} times)
        </div>
        <div class="insight-box">
            <strong>Top Cool-down Artist:</strong> {after_artists.most_common(1)[0][0] if after_artists else 'N/A'}
            ({after_artists.most_common(1)[0][1] if after_artists else 0} times)
        </div>
        <div class="insight-box">
            <strong>Listening Pattern:</strong>
            {len(pommelien_only_days)} days of pure Pommelien listening vs
            {len(mixed_days)} days with mixed artists
        </div>
    </div>

    <script>
        // Dashboard
        var dashboardData = {fig1.to_json()};
        Plotly.newPlot('dashboard', dashboardData.data, dashboardData.layout);

        // Sankey
        var sankeyData = {fig2.to_json()};
        Plotly.newPlot('sankey', sankeyData.data, sankeyData.layout);
    </script>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\nVisualization saved to: {output_path}")

    # Open in browser
    import webbrowser
    webbrowser.open(f'file:///{output_path}')
    print("Opening in browser...")

if __name__ == "__main__":
    main()
