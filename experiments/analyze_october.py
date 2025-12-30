"""
Analyze Spotify listening data for October 22-25, 2025 - Pommelien Thijs synchronicity window.
Creates an hour-by-hour breakdown and interactive visualization.
"""

import json
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

# Load the JSON file
data_path = Path(r"C:\Users\Nouri\Desktop\easiest game of my life\mabrouk_spotify_dataset\Spotify Extended Streaming History\Streaming_History_Audio_2025_4.json")

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Filter for October 22-25, 2025 and Pommelien Thijs
october_data = []
for entry in data:
    ts = entry.get('ts', '')
    if ts.startswith('2025-10-2') and ts[9] in '2345':
        day = int(ts[8:10])
        if 22 <= day <= 25:
            october_data.append(entry)

# Filter for Pommelien Thijs specifically
pommelien_data = [e for e in october_data if e.get('master_metadata_album_artist_name') == 'Pommelien Thijs']

print(f"Total tracks October 22-25: {len(october_data)}")
print(f"Pommelien Thijs tracks: {len(pommelien_data)}")

# Organize by day and hour
daily_data = defaultdict(list)
hourly_data = defaultdict(lambda: defaultdict(list))

for entry in october_data:
    ts = entry['ts']
    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
    # Convert to local time (Netherlands, UTC+1/2 in October = CEST ends Oct 27, so still UTC+2)
    local_dt = dt + timedelta(hours=2)
    day = local_dt.strftime('%Y-%m-%d')
    hour = local_dt.hour

    daily_data[day].append({
        'time': local_dt,
        'track': entry.get('master_metadata_track_name', 'Unknown'),
        'artist': entry.get('master_metadata_album_artist_name', 'Unknown'),
        'ms_played': entry.get('ms_played', 0),
        'is_pommelien': entry.get('master_metadata_album_artist_name') == 'Pommelien Thijs'
    })
    hourly_data[day][hour].append(daily_data[day][-1])

# Analyze each day
print("\n" + "="*80)
print("OCTOBER 22-25, 2025 - POMMELIEN THIJS SYNCHRONICITY WINDOW")
print("="*80)

for day in sorted(daily_data.keys()):
    tracks = daily_data[day]
    if not tracks:
        continue

    print(f"\n{'='*80}")
    print(f"  {day} ({datetime.strptime(day, '%Y-%m-%d').strftime('%A')})")
    print(f"{'='*80}")

    # Session start/end
    first_track = min(tracks, key=lambda x: x['time'])
    last_track = max(tracks, key=lambda x: x['time'])
    print(f"  First track: {first_track['time'].strftime('%H:%M:%S')} - {first_track['track']} by {first_track['artist']}")
    print(f"  Last track:  {last_track['time'].strftime('%H:%M:%S')} - {last_track['track']} by {last_track['artist']}")

    # Total listening time
    total_ms = sum(t['ms_played'] for t in tracks)
    pommelien_ms = sum(t['ms_played'] for t in tracks if t['is_pommelien'])
    total_hours = total_ms / (1000 * 60 * 60)
    pommelien_hours = pommelien_ms / (1000 * 60 * 60)

    print(f"\n  Total listening: {total_hours:.2f} hours")
    print(f"  Pommelien Thijs: {pommelien_hours:.2f} hours ({100*pommelien_hours/total_hours:.1f}%)")

    # Hourly breakdown
    print(f"\n  Hourly Breakdown:")
    print(f"  {'-'*76}")

    for hour in sorted(hourly_data[day].keys()):
        hour_tracks = hourly_data[day][hour]
        pommelien_count = sum(1 for t in hour_tracks if t['is_pommelien'])
        total_count = len(hour_tracks)

        hour_str = f"{hour:02d}:00-{hour+1:02d}:00"
        bar_pom = '#' * pommelien_count
        bar_other = '.' * (total_count - pommelien_count)

        print(f"  {hour_str} | {bar_pom}{bar_other} ({pommelien_count}/{total_count})")

        # Show Pommelien tracks in this hour
        for t in hour_tracks:
            if t['is_pommelien']:
                print(f"           -> {t['time'].strftime('%H:%M')} {t['track'][:40]}")

# Create the HTML visualization
html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>October 22-25, 2025 - Pommelien Thijs Synchronicity Window</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #e4e4e7;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #1db954;
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(29, 185, 84, 0.5);
        }
        .subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 1.1rem;
            margin-bottom: 30px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(29, 185, 84, 0.3);
        }
        .stat-value {
            font-size: 2.5rem;
            color: #1db954;
            font-weight: bold;
        }
        .stat-label {
            color: #94a3b8;
            margin-top: 5px;
        }
        .timeline-container {
            position: relative;
            margin: 40px 0;
        }
        .day-section {
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            margin-bottom: 30px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .day-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .day-title {
            font-size: 1.5rem;
            color: #1db954;
        }
        .day-date {
            color: #94a3b8;
            font-size: 0.9rem;
        }
        .day-stats {
            text-align: right;
        }
        .hour-row {
            display: grid;
            grid-template-columns: 80px 1fr auto;
            gap: 15px;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .hour-label {
            font-family: monospace;
            color: #64748b;
            font-size: 0.9rem;
        }
        .hour-bar {
            height: 24px;
            display: flex;
            border-radius: 4px;
            overflow: hidden;
        }
        .bar-pommelien {
            background: linear-gradient(90deg, #1db954, #4ade80);
            transition: width 0.3s ease;
        }
        .bar-other {
            background: rgba(255,255,255,0.1);
        }
        .hour-count {
            font-family: monospace;
            color: #94a3b8;
            min-width: 50px;
            text-align: right;
        }
        .track-list {
            margin-left: 95px;
            margin-top: 5px;
        }
        .track-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 5px 10px;
            margin: 3px 0;
            background: rgba(29, 185, 84, 0.1);
            border-radius: 6px;
            font-size: 0.85rem;
            border-left: 3px solid #1db954;
        }
        .track-time {
            color: #1db954;
            font-family: monospace;
        }
        .track-name {
            color: #e4e4e7;
            flex: 1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .track-duration {
            color: #64748b;
            font-size: 0.8rem;
        }
        .pattern-section {
            background: rgba(29, 185, 84, 0.05);
            border: 1px solid rgba(29, 185, 84, 0.3);
            border-radius: 16px;
            padding: 25px;
            margin-top: 30px;
        }
        .pattern-title {
            color: #1db954;
            font-size: 1.3rem;
            margin-bottom: 15px;
        }
        .pattern-list {
            list-style: none;
        }
        .pattern-list li {
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }
        .pattern-list li:before {
            content: "\\2728";
        }
        .heatmap-container {
            margin: 30px 0;
        }
        .heatmap-title {
            color: #1db954;
            font-size: 1.3rem;
            margin-bottom: 15px;
        }
        .heatmap {
            display: grid;
            grid-template-columns: 60px repeat(24, 1fr);
            gap: 2px;
        }
        .heatmap-hour-label {
            font-size: 0.7rem;
            color: #64748b;
            text-align: center;
            padding: 3px;
        }
        .heatmap-day-label {
            font-size: 0.8rem;
            color: #94a3b8;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
        }
        .heatmap-cell {
            aspect-ratio: 1;
            border-radius: 3px;
            min-height: 20px;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            color: #64748b;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>The Synchronicity Window</h1>
        <div class="subtitle">October 22-25, 2025 - Pommelien Thijs Listening Analysis</div>

        <div class="stats-grid">
"""

# Calculate overall stats
total_tracks = len(october_data)
pommelien_tracks = len(pommelien_data)
total_listening_hours = sum(e.get('ms_played', 0) for e in october_data) / (1000 * 60 * 60)
pommelien_listening_hours = sum(e.get('ms_played', 0) for e in pommelien_data) / (1000 * 60 * 60)

html_content += f"""
            <div class="stat-card">
                <div class="stat-value">{pommelien_tracks}</div>
                <div class="stat-label">Pommelien Thijs Plays</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{pommelien_listening_hours:.1f}h</div>
                <div class="stat-label">Hours of Pommelien</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{100*pommelien_tracks/total_tracks:.0f}%</div>
                <div class="stat-label">Of All Plays</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">4</div>
                <div class="stat-label">Days of Immersion</div>
            </div>
        </div>
"""

# Add heatmap
html_content += """
        <div class="heatmap-container">
            <h3 class="heatmap-title">24-Hour Listening Heatmap</h3>
            <div class="heatmap">
                <div></div>
"""
for h in range(24):
    html_content += f'<div class="heatmap-hour-label">{h:02d}</div>'

for day in sorted(daily_data.keys()):
    day_name = datetime.strptime(day, '%Y-%m-%d').strftime('%a %d')
    html_content += f'<div class="heatmap-day-label">{day_name}</div>'

    for hour in range(24):
        hour_tracks = hourly_data[day].get(hour, [])
        pommelien_count = sum(1 for t in hour_tracks if t['is_pommelien'])
        total_count = len(hour_tracks)

        if total_count == 0:
            color = "rgba(255,255,255,0.02)"
        elif pommelien_count > 0:
            intensity = min(1, pommelien_count / 5)
            color = f"rgba(29, 185, 84, {0.3 + 0.7 * intensity})"
        else:
            color = "rgba(255,255,255,0.1)"

        html_content += f'<div class="heatmap-cell" style="background: {color};" title="{day} {hour:02d}:00 - {pommelien_count}/{total_count} Pommelien"></div>'

html_content += """
            </div>
        </div>
"""

# Add daily sections
for day in sorted(daily_data.keys()):
    tracks = daily_data[day]
    if not tracks:
        continue

    day_name = datetime.strptime(day, '%Y-%m-%d').strftime('%A')
    first_track = min(tracks, key=lambda x: x['time'])
    last_track = max(tracks, key=lambda x: x['time'])

    total_ms = sum(t['ms_played'] for t in tracks)
    pommelien_ms = sum(t['ms_played'] for t in tracks if t['is_pommelien'])
    pommelien_count = sum(1 for t in tracks if t['is_pommelien'])

    html_content += f"""
        <div class="day-section">
            <div class="day-header">
                <div>
                    <div class="day-title">{day_name}</div>
                    <div class="day-date">{day} | {first_track['time'].strftime('%H:%M')} - {last_track['time'].strftime('%H:%M')}</div>
                </div>
                <div class="day-stats">
                    <div style="color: #1db954; font-size: 1.3rem; font-weight: bold;">{pommelien_count} Pommelien tracks</div>
                    <div style="color: #94a3b8;">{pommelien_ms/(1000*60):.0f} minutes</div>
                </div>
            </div>
"""

    for hour in sorted(hourly_data[day].keys()):
        hour_tracks = hourly_data[day][hour]
        pommelien_count = sum(1 for t in hour_tracks if t['is_pommelien'])
        total_count = len(hour_tracks)

        max_bar = 20  # Max tracks for full bar
        pom_width = min(100, (pommelien_count / max_bar) * 100)
        other_width = min(100 - pom_width, ((total_count - pommelien_count) / max_bar) * 100)

        html_content += f"""
            <div class="hour-row">
                <div class="hour-label">{hour:02d}:00</div>
                <div class="hour-bar">
                    <div class="bar-pommelien" style="width: {pom_width}%"></div>
                    <div class="bar-other" style="width: {other_width}%"></div>
                </div>
                <div class="hour-count">{pommelien_count}/{total_count}</div>
            </div>
"""

        # Show Pommelien tracks
        pommelien_in_hour = [t for t in hour_tracks if t['is_pommelien']]
        if pommelien_in_hour:
            html_content += '<div class="track-list">'
            for t in pommelien_in_hour:
                duration_min = t['ms_played'] / (1000 * 60)
                html_content += f"""
                <div class="track-item">
                    <span class="track-time">{t['time'].strftime('%H:%M')}</span>
                    <span class="track-name">{t['track']}</span>
                    <span class="track-duration">{duration_min:.1f}min</span>
                </div>
"""
            html_content += '</div>'

    html_content += """
        </div>
"""

# Identify patterns
patterns = []

# Check for morning rituals
for day in sorted(daily_data.keys()):
    morning_pom = sum(1 for t in daily_data[day] if t['is_pommelien'] and 6 <= t['time'].hour < 10)
    if morning_pom > 3:
        patterns.append(f"<strong>{datetime.strptime(day, '%Y-%m-%d').strftime('%A')}</strong>: Morning ritual with {morning_pom} Pommelien tracks (6-10 AM)")

# Check for late night sessions
for day in sorted(daily_data.keys()):
    night_pom = sum(1 for t in daily_data[day] if t['is_pommelien'] and t['time'].hour >= 22)
    if night_pom > 5:
        patterns.append(f"<strong>{datetime.strptime(day, '%Y-%m-%d').strftime('%A')}</strong>: Late night session with {night_pom} Pommelien tracks (10 PM onwards)")

# Check for extended sessions
for day in sorted(daily_data.keys()):
    tracks = daily_data[day]
    pommelien_tracks = [t for t in tracks if t['is_pommelien']]
    if len(pommelien_tracks) >= 2:
        sorted_pom = sorted(pommelien_tracks, key=lambda x: x['time'])
        for i in range(len(sorted_pom) - 1):
            gap = (sorted_pom[i+1]['time'] - sorted_pom[i]['time']).total_seconds() / 60
            if gap < 5:
                # Count consecutive plays
                consecutive = 2
                for j in range(i+2, len(sorted_pom)):
                    if (sorted_pom[j]['time'] - sorted_pom[j-1]['time']).total_seconds() / 60 < 5:
                        consecutive += 1
                    else:
                        break
                if consecutive >= 5:
                    patterns.append(f"<strong>{datetime.strptime(day, '%Y-%m-%d').strftime('%A')}</strong> at {sorted_pom[i]['time'].strftime('%H:%M')}: {consecutive} consecutive Pommelien tracks")
                    break

# Get most played tracks
track_counts = defaultdict(int)
for entry in pommelien_data:
    track = entry.get('master_metadata_track_name', 'Unknown')
    track_counts[track] += 1

top_tracks = sorted(track_counts.items(), key=lambda x: x[1], reverse=True)[:5]
patterns.append(f"<strong>Most played</strong>: " + ", ".join([f'"{t[0]}" ({t[1]}x)' for t in top_tracks]))

html_content += """
        <div class="pattern-section">
            <h3 class="pattern-title">Detected Patterns</h3>
            <ul class="pattern-list">
"""

for pattern in patterns:
    html_content += f'<li>{pattern}</li>'

html_content += """
            </ul>
        </div>

        <div class="footer">
            <p>Generated from Spotify Extended Streaming History</p>
            <p>The synchronicity window reveals the musical immersion during October 22-25, 2025</p>
        </div>
    </div>
</body>
</html>
"""

# Write the HTML file
output_path = Path(r"C:\Users\Nouri\Documents\GitHub\Nouri-Mabrouk\experiments\october_hourly.html")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n\nVisualization saved to: {output_path}")
print("Opening in browser...")

import webbrowser
webbrowser.open(str(output_path))
