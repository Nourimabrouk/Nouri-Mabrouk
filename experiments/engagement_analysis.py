"""
Pommelien Thijs 2025 Engagement Analysis
Analyzes skip rates, completion rates, play patterns, and listening behavior
"""

import json
import os
from collections import defaultdict
from pathlib import Path

# Load all streaming history files for 2024-2025 and 2025
data_dir = Path(r"C:\Users\Nouri\Desktop\easiest game of my life\mabrouk_spotify_dataset\Spotify Extended Streaming History")

all_streams = []
for file_path in data_dir.glob("*.json"):
    if "Audio" in file_path.name and ("2024-2025" in file_path.name or "2025" in file_path.name):
        with open(file_path, 'r', encoding='utf-8') as f:
            streams = json.load(f)
            all_streams.extend(streams)

print(f"Total streams loaded: {len(all_streams)}")

# Filter for Pommelien Thijs in 2025
pommelien_2025 = [
    s for s in all_streams
    if s.get('master_metadata_album_artist_name') == 'Pommelien Thijs'
    and s.get('ts', '').startswith('2025')
]

print(f"Pommelien Thijs 2025 streams: {len(pommelien_2025)}")

# Track-level analysis
track_stats = defaultdict(lambda: {
    'plays': 0,
    'skipped': 0,
    'total_ms_played': 0,
    'max_ms_played': 0,  # Approximation for track length
    'reason_start': defaultdict(int),
    'reason_end': defaultdict(int),
    'shuffle_plays': 0,
    'intentional_plays': 0,
    'offline_plays': 0,
    'online_plays': 0,
    'album': '',
    'uri': ''
})

for stream in pommelien_2025:
    track = stream.get('master_metadata_track_name', 'Unknown')
    stats = track_stats[track]

    stats['plays'] += 1
    stats['total_ms_played'] += stream.get('ms_played', 0)
    stats['max_ms_played'] = max(stats['max_ms_played'], stream.get('ms_played', 0))
    stats['album'] = stream.get('master_metadata_album_album_name', '')
    stats['uri'] = stream.get('spotify_track_uri', '')

    if stream.get('skipped', False):
        stats['skipped'] += 1

    reason_start = stream.get('reason_start', 'unknown')
    reason_end = stream.get('reason_end', 'unknown')
    stats['reason_start'][reason_start] += 1
    stats['reason_end'][reason_end] += 1

    if stream.get('shuffle', False):
        stats['shuffle_plays'] += 1
    else:
        stats['intentional_plays'] += 1

    if stream.get('offline', False):
        stats['offline_plays'] += 1
    else:
        stats['online_plays'] += 1

# Calculate derived metrics
engagement_data = []
for track, stats in track_stats.items():
    if stats['plays'] > 0:
        skip_rate = (stats['skipped'] / stats['plays']) * 100
        avg_ms_played = stats['total_ms_played'] / stats['plays']
        # Estimate completion rate using max_ms_played as proxy for track length
        completion_rate = (avg_ms_played / stats['max_ms_played'] * 100) if stats['max_ms_played'] > 0 else 0

        # Count intentional seeks (clickrow means user clicked to play)
        intentional_seeks = stats['reason_start'].get('clickrow', 0)
        auto_plays = stats['reason_start'].get('trackdone', 0)

        shuffle_pct = (stats['shuffle_plays'] / stats['plays']) * 100
        offline_pct = (stats['offline_plays'] / stats['plays']) * 100

        engagement_data.append({
            'track': track,
            'album': stats['album'],
            'plays': stats['plays'],
            'skip_rate': round(skip_rate, 1),
            'avg_duration_sec': round(avg_ms_played / 1000, 1),
            'max_duration_sec': round(stats['max_ms_played'] / 1000, 1),
            'completion_rate': round(min(completion_rate, 100), 1),
            'intentional_seeks': intentional_seeks,
            'auto_plays': auto_plays,
            'shuffle_pct': round(shuffle_pct, 1),
            'offline_pct': round(offline_pct, 1),
            'reason_start': dict(stats['reason_start']),
            'reason_end': dict(stats['reason_end'])
        })

# Sort by plays
engagement_data.sort(key=lambda x: x['plays'], reverse=True)

# Print summary
print("\n" + "="*80)
print("POMMELIEN THIJS 2025 ENGAGEMENT ANALYSIS")
print("="*80)

for track in engagement_data:
    print(f"\n{track['track']} ({track['album']})")
    print(f"  Plays: {track['plays']} | Skip Rate: {track['skip_rate']}%")
    print(f"  Avg Duration: {track['avg_duration_sec']}s | Max: {track['max_duration_sec']}s | Completion: {track['completion_rate']}%")
    print(f"  Intentional Seeks: {track['intentional_seeks']} | Auto-plays: {track['auto_plays']}")
    print(f"  Shuffle: {track['shuffle_pct']}% | Offline: {track['offline_pct']}%")
    print(f"  Start reasons: {track['reason_start']}")
    print(f"  End reasons: {track['reason_end']}")

# Generate HTML visualization
html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pommelien Thijs 2025 - Engagement Analysis</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e0e0e0;
            padding: 20px;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #1DB954;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(29, 185, 84, 0.3);
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .stats-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(29, 185, 84, 0.2);
        }
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #1DB954;
        }
        .stat-label {
            color: #aaa;
            margin-top: 5px;
            font-size: 0.9em;
        }
        .chart-container {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .chart-title {
            color: #1DB954;
            font-size: 1.4em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(29, 185, 84, 0.3);
        }
        .track-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .track-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s;
        }
        .track-card:hover {
            transform: scale(1.02);
            border-color: #1DB954;
        }
        .track-name {
            font-size: 1.2em;
            color: #fff;
            margin-bottom: 5px;
        }
        .track-album {
            color: #888;
            font-size: 0.9em;
            margin-bottom: 15px;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .metric-label {
            color: #888;
        }
        .metric-value {
            color: #1DB954;
            font-weight: bold;
        }
        .metric-value.warning {
            color: #ff6b6b;
        }
        .metric-value.good {
            color: #4ecdc4;
        }
        .bar-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            height: 8px;
            margin-top: 5px;
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        .bar-fill.green { background: linear-gradient(90deg, #1DB954, #4ecdc4); }
        .bar-fill.red { background: linear-gradient(90deg, #ff6b6b, #ffa502); }
        .bar-fill.blue { background: linear-gradient(90deg, #3498db, #9b59b6); }
        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
            margin: 20px 0;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .legend-color {
            width: 15px;
            height: 15px;
            border-radius: 3px;
        }
        .insights {
            background: linear-gradient(135deg, rgba(29, 185, 84, 0.1), rgba(78, 205, 196, 0.1));
            border-radius: 15px;
            padding: 25px;
            margin-top: 30px;
            border: 1px solid rgba(29, 185, 84, 0.3);
        }
        .insights h3 {
            color: #1DB954;
            margin-bottom: 15px;
        }
        .insights ul {
            list-style: none;
            padding-left: 0;
        }
        .insights li {
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .insights li:last-child {
            border-bottom: none;
        }
        .insights li::before {
            content: "\\2022";
            color: #1DB954;
            font-weight: bold;
            margin-right: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Pommelien Thijs</h1>
        <p class="subtitle">2025 Spotify Engagement Analysis</p>

        <div class="stats-summary">
            <div class="stat-card">
                <div class="stat-value" id="total-plays">0</div>
                <div class="stat-label">Total Plays</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="unique-tracks">0</div>
                <div class="stat-label">Unique Tracks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avg-skip-rate">0%</div>
                <div class="stat-label">Average Skip Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avg-completion">0%</div>
                <div class="stat-label">Avg Completion Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="intentional-rate">0%</div>
                <div class="stat-label">Intentional Plays</div>
            </div>
        </div>

        <div class="chart-container">
            <h2 class="chart-title">Engagement Overview by Track</h2>
            <div id="engagement-chart"></div>
        </div>

        <div class="chart-container">
            <h2 class="chart-title">Skip Rate vs Completion Rate</h2>
            <div id="scatter-chart"></div>
        </div>

        <div class="chart-container">
            <h2 class="chart-title">Play Initiation Patterns</h2>
            <div id="start-reasons-chart"></div>
        </div>

        <div class="chart-container">
            <h2 class="chart-title">Shuffle vs Intentional Play</h2>
            <div id="shuffle-chart"></div>
        </div>

        <div class="chart-container">
            <h2 class="chart-title">Detailed Track Metrics</h2>
            <div class="track-grid" id="track-grid"></div>
        </div>

        <div class="insights" id="insights">
            <h3>Key Insights</h3>
            <ul id="insights-list"></ul>
        </div>
    </div>

    <script>
        const engagementData = ''' + json.dumps(engagement_data) + ''';

        // Calculate summary stats
        const totalPlays = engagementData.reduce((sum, t) => sum + t.plays, 0);
        const uniqueTracks = engagementData.length;
        const avgSkipRate = engagementData.reduce((sum, t) => sum + t.skip_rate * t.plays, 0) / totalPlays;
        const avgCompletion = engagementData.reduce((sum, t) => sum + t.completion_rate * t.plays, 0) / totalPlays;
        const totalIntentional = engagementData.reduce((sum, t) => sum + t.intentional_seeks, 0);
        const intentionalRate = (totalIntentional / totalPlays) * 100;

        document.getElementById('total-plays').textContent = totalPlays;
        document.getElementById('unique-tracks').textContent = uniqueTracks;
        document.getElementById('avg-skip-rate').textContent = avgSkipRate.toFixed(1) + '%';
        document.getElementById('avg-completion').textContent = avgCompletion.toFixed(1) + '%';
        document.getElementById('intentional-rate').textContent = intentionalRate.toFixed(1) + '%';

        // Plotly config
        const plotlyConfig = {responsive: true, displayModeBar: false};
        const darkLayout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {color: '#e0e0e0'},
            margin: {l: 50, r: 30, t: 30, b: 100}
        };

        // Engagement Overview Chart
        const tracks = engagementData.map(t => t.track);
        const plays = engagementData.map(t => t.plays);
        const skipRates = engagementData.map(t => t.skip_rate);
        const completionRates = engagementData.map(t => t.completion_rate);

        Plotly.newPlot('engagement-chart', [
            {
                x: tracks,
                y: plays,
                type: 'bar',
                name: 'Total Plays',
                marker: {color: '#1DB954'},
                yaxis: 'y'
            },
            {
                x: tracks,
                y: completionRates,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Completion Rate %',
                marker: {color: '#4ecdc4', size: 10},
                line: {color: '#4ecdc4', width: 3},
                yaxis: 'y2'
            },
            {
                x: tracks,
                y: skipRates,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Skip Rate %',
                marker: {color: '#ff6b6b', size: 10},
                line: {color: '#ff6b6b', width: 3},
                yaxis: 'y2'
            }
        ], {
            ...darkLayout,
            yaxis: {title: 'Plays', gridcolor: 'rgba(255,255,255,0.1)'},
            yaxis2: {title: 'Rate (%)', overlaying: 'y', side: 'right', range: [0, 100], gridcolor: 'rgba(255,255,255,0.1)'},
            xaxis: {tickangle: -45, gridcolor: 'rgba(255,255,255,0.1)'},
            legend: {orientation: 'h', y: 1.15},
            barmode: 'group'
        }, plotlyConfig);

        // Scatter Chart - Skip Rate vs Completion
        Plotly.newPlot('scatter-chart', [{
            x: skipRates,
            y: completionRates,
            mode: 'markers+text',
            type: 'scatter',
            text: tracks,
            textposition: 'top center',
            marker: {
                size: plays.map(p => Math.max(15, p * 3)),
                color: plays,
                colorscale: [[0, '#3498db'], [0.5, '#1DB954'], [1, '#f1c40f']],
                showscale: true,
                colorbar: {title: 'Plays'}
            },
            hovertemplate: '<b>%{text}</b><br>Skip Rate: %{x}%<br>Completion: %{y}%<extra></extra>'
        }], {
            ...darkLayout,
            xaxis: {title: 'Skip Rate (%)', gridcolor: 'rgba(255,255,255,0.1)'},
            yaxis: {title: 'Completion Rate (%)', gridcolor: 'rgba(255,255,255,0.1)'},
            shapes: [
                {type: 'line', x0: 0, x1: 100, y0: 80, y1: 80, line: {color: '#4ecdc4', dash: 'dash', width: 1}},
                {type: 'line', x0: 20, x1: 20, y0: 0, y1: 100, line: {color: '#ff6b6b', dash: 'dash', width: 1}}
            ],
            annotations: [
                {x: 10, y: 95, text: 'High Engagement Zone', showarrow: false, font: {color: '#4ecdc4'}},
                {x: 60, y: 50, text: 'Low Engagement Zone', showarrow: false, font: {color: '#ff6b6b'}}
            ]
        }, plotlyConfig);

        // Start Reasons Chart
        const startReasons = {};
        engagementData.forEach(t => {
            Object.entries(t.reason_start).forEach(([reason, count]) => {
                startReasons[reason] = (startReasons[reason] || 0) + count;
            });
        });

        const reasonLabels = {
            'clickrow': 'Intentional Click',
            'trackdone': 'Auto-play',
            'fwdbtn': 'Forward Button',
            'backbtn': 'Back Button',
            'appload': 'App Load',
            'playbtn': 'Play Button'
        };

        Plotly.newPlot('start-reasons-chart', [{
            labels: Object.keys(startReasons).map(r => reasonLabels[r] || r),
            values: Object.values(startReasons),
            type: 'pie',
            hole: 0.5,
            marker: {
                colors: ['#1DB954', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
            },
            textinfo: 'label+percent',
            textfont: {color: '#fff'}
        }], {
            ...darkLayout,
            showlegend: true,
            legend: {font: {color: '#e0e0e0'}}
        }, plotlyConfig);

        // Shuffle Chart
        const shufflePlays = engagementData.reduce((sum, t) => sum + (t.shuffle_pct * t.plays / 100), 0);
        const intentionalPlays = totalPlays - shufflePlays;

        Plotly.newPlot('shuffle-chart', [{
            x: tracks,
            y: engagementData.map(t => 100 - t.shuffle_pct),
            type: 'bar',
            name: 'Intentional',
            marker: {color: '#1DB954'}
        }, {
            x: tracks,
            y: engagementData.map(t => t.shuffle_pct),
            type: 'bar',
            name: 'Shuffle',
            marker: {color: '#9b59b6'}
        }], {
            ...darkLayout,
            barmode: 'stack',
            xaxis: {tickangle: -45, gridcolor: 'rgba(255,255,255,0.1)'},
            yaxis: {title: 'Percentage', gridcolor: 'rgba(255,255,255,0.1)'},
            legend: {orientation: 'h', y: 1.15}
        }, plotlyConfig);

        // Track Grid
        const trackGrid = document.getElementById('track-grid');
        engagementData.forEach(t => {
            const card = document.createElement('div');
            card.className = 'track-card';
            card.innerHTML = `
                <div class="track-name">${t.track}</div>
                <div class="track-album">${t.album}</div>
                <div class="metric-row">
                    <span class="metric-label">Total Plays</span>
                    <span class="metric-value">${t.plays}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Skip Rate</span>
                    <span class="metric-value ${t.skip_rate > 30 ? 'warning' : 'good'}">${t.skip_rate}%</span>
                </div>
                <div class="bar-container">
                    <div class="bar-fill ${t.skip_rate > 30 ? 'red' : 'green'}" style="width: ${t.skip_rate}%"></div>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Completion Rate</span>
                    <span class="metric-value ${t.completion_rate < 70 ? 'warning' : 'good'}">${t.completion_rate}%</span>
                </div>
                <div class="bar-container">
                    <div class="bar-fill green" style="width: ${t.completion_rate}%"></div>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Avg Duration</span>
                    <span class="metric-value">${t.avg_duration_sec}s / ${t.max_duration_sec}s</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Intentional Seeks</span>
                    <span class="metric-value">${t.intentional_seeks}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Shuffle %</span>
                    <span class="metric-value">${t.shuffle_pct}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Offline %</span>
                    <span class="metric-value">${t.offline_pct}%</span>
                </div>
            `;
            trackGrid.appendChild(card);
        });

        // Generate Insights
        const insights = [];

        // Most played track
        const mostPlayed = engagementData[0];
        insights.push(`<strong>"${mostPlayed.track}"</strong> is the most played track with ${mostPlayed.plays} plays and ${mostPlayed.completion_rate}% completion rate.`);

        // Lowest skip rate
        const lowestSkip = [...engagementData].sort((a, b) => a.skip_rate - b.skip_rate)[0];
        insights.push(`<strong>"${lowestSkip.track}"</strong> has the lowest skip rate at ${lowestSkip.skip_rate}% - highly engaging content.`);

        // Highest skip rate
        const highestSkip = [...engagementData].sort((a, b) => b.skip_rate - a.skip_rate)[0];
        if (highestSkip.skip_rate > 20) {
            insights.push(`<strong>"${highestSkip.track}"</strong> has the highest skip rate at ${highestSkip.skip_rate}% - may need different listening context.`);
        }

        // Most intentionally sought
        const mostSought = [...engagementData].sort((a, b) => b.intentional_seeks - a.intentional_seeks)[0];
        insights.push(`<strong>"${mostSought.track}"</strong> is most intentionally sought with ${mostSought.intentional_seeks} direct clicks - a clear favorite.`);

        // Shuffle vs Intentional summary
        insights.push(`${intentionalRate.toFixed(0)}% of Pommelien Thijs plays are intentional (not shuffle) - indicating deliberate fan engagement.`);

        // Completion rate insight
        if (avgCompletion > 80) {
            insights.push(`Average completion rate of ${avgCompletion.toFixed(1)}% shows strong listener commitment to finishing tracks.`);
        } else if (avgCompletion < 60) {
            insights.push(`Average completion rate of ${avgCompletion.toFixed(1)}% suggests tracks may be sampled more than fully listened.`);
        }

        // Offline insight
        const offlinePlays = engagementData.filter(t => t.offline_pct > 0);
        if (offlinePlays.length > 0) {
            const avgOffline = offlinePlays.reduce((sum, t) => sum + t.offline_pct, 0) / offlinePlays.length;
            insights.push(`${offlinePlays.length} track(s) have been played offline, averaging ${avgOffline.toFixed(1)}% offline plays - indicates downloaded favorites.`);
        }

        const insightsList = document.getElementById('insights-list');
        insights.forEach(insight => {
            const li = document.createElement('li');
            li.innerHTML = insight;
            insightsList.appendChild(li);
        });
    </script>
</body>
</html>'''

# Save the HTML file
output_path = Path(r"C:\Users\Nouri\Documents\GitHub\Nouri-Mabrouk\experiments\engagement_analysis.html")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n\nVisualization saved to: {output_path}")
print("\nOpening in browser...")

# Open the HTML file
import webbrowser
webbrowser.open(str(output_path))
