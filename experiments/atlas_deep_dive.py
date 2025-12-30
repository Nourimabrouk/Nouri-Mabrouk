"""
ATLAS by Pommelien Thijs - Deep Dive Analysis
Analyzing the #1 track of 2025 with 413 plays and 19+ hours of listening
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import html

# Data paths
DATA_DIR = Path(r"C:\Users\Nouri\Desktop\easiest game of my life\mabrouk_spotify_dataset\Spotify Extended Streaming History")
OUTPUT_FILE = Path(r"C:\Users\Nouri\Documents\GitHub\Nouri-Mabrouk\experiments\atlas_deep_dive.html")

def load_all_streaming_data():
    """Load all JSON streaming history files"""
    all_streams = []
    json_files = sorted(DATA_DIR.glob("Streaming_History_Audio_*.json"))

    for json_file in json_files:
        print(f"Loading {json_file.name}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_streams.extend(data)

    print(f"Total streams loaded: {len(all_streams)}")
    return all_streams

def filter_atlas_plays(streams):
    """Filter for Atlas by Pommelien Thijs plays"""
    atlas_plays = []

    for stream in streams:
        track_name = stream.get('master_metadata_track_name', '')
        artist_name = stream.get('master_metadata_album_artist_name', '')

        if track_name and artist_name:
            # Match Atlas by Pommelien Thijs
            if 'atlas' in track_name.lower() and 'pommelien' in artist_name.lower():
                atlas_plays.append(stream)

    return atlas_plays

def analyze_atlas(atlas_plays, all_streams):
    """Perform deep analysis on Atlas plays"""

    # Parse timestamps
    for play in atlas_plays:
        play['datetime'] = datetime.fromisoformat(play['ts'].replace('Z', '+00:00'))

    # Sort by timestamp
    atlas_plays.sort(key=lambda x: x['datetime'])

    # 1. First play date
    first_play = atlas_plays[0]['datetime']
    last_play = atlas_plays[-1]['datetime']

    # 2. Plays over time (by date)
    plays_by_date = defaultdict(int)
    ms_by_date = defaultdict(int)

    for play in atlas_plays:
        date_str = play['datetime'].strftime('%Y-%m-%d')
        plays_by_date[date_str] += 1
        ms_by_date[date_str] += play.get('ms_played', 0)

    # 3. Time of day analysis
    plays_by_hour = defaultdict(int)
    ms_by_hour = defaultdict(int)

    for play in atlas_plays:
        hour = play['datetime'].hour
        plays_by_hour[hour] += 1
        ms_by_hour[hour] += play.get('ms_played', 0)

    # 4. Day of week analysis
    plays_by_weekday = defaultdict(int)
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for play in atlas_plays:
        weekday = play['datetime'].weekday()
        plays_by_weekday[weekday_names[weekday]] += 1

    # 5. Plays by month
    plays_by_month = defaultdict(int)

    for play in atlas_plays:
        month_str = play['datetime'].strftime('%Y-%m')
        plays_by_month[month_str] += 1

    # 6. Before/After track analysis
    # Parse all streams with timestamps
    for stream in all_streams:
        stream['datetime'] = datetime.fromisoformat(stream['ts'].replace('Z', '+00:00'))

    all_streams.sort(key=lambda x: x['datetime'])

    # Find tracks played immediately before/after Atlas
    before_atlas = []
    after_atlas = []

    # Create a lookup for stream positions
    for i, stream in enumerate(all_streams):
        track_name = stream.get('master_metadata_track_name', '')
        artist_name = stream.get('master_metadata_album_artist_name', '')

        if track_name and artist_name:
            if 'atlas' in track_name.lower() and 'pommelien' in artist_name.lower():
                # Get previous track
                if i > 0:
                    prev = all_streams[i-1]
                    prev_track = prev.get('master_metadata_track_name', 'Unknown')
                    prev_artist = prev.get('master_metadata_album_artist_name', 'Unknown')
                    if prev_track and prev_track != 'Atlas':
                        before_atlas.append(f"{prev_track} - {prev_artist}")

                # Get next track
                if i < len(all_streams) - 1:
                    next_s = all_streams[i+1]
                    next_track = next_s.get('master_metadata_track_name', 'Unknown')
                    next_artist = next_s.get('master_metadata_album_artist_name', 'Unknown')
                    if next_track and next_track != 'Atlas':
                        after_atlas.append(f"{next_track} - {next_artist}")

    before_counter = Counter(before_atlas)
    after_counter = Counter(after_atlas)

    # 7. Platform analysis
    platforms = Counter(play.get('platform', 'unknown') for play in atlas_plays)

    # 8. Skip rate
    skipped = sum(1 for play in atlas_plays if play.get('skipped', False))

    # 9. Total listening time
    total_ms = sum(play.get('ms_played', 0) for play in atlas_plays)
    total_hours = total_ms / 1000 / 60 / 60

    # 10. Average play duration
    avg_ms = total_ms / len(atlas_plays) if atlas_plays else 0
    avg_minutes = avg_ms / 1000 / 60

    # 11. Identify "Atlas phases" - periods of intense listening
    # Group by week
    plays_by_week = defaultdict(int)
    for play in atlas_plays:
        week_str = play['datetime'].strftime('%Y-W%W')
        plays_by_week[week_str] += 1

    # Find peak weeks
    peak_weeks = sorted(plays_by_week.items(), key=lambda x: x[1], reverse=True)[:5]

    # 12. Listening streaks (consecutive days with Atlas plays)
    dates_with_atlas = sorted(set(play['datetime'].date() for play in atlas_plays))
    streaks = []
    current_streak = [dates_with_atlas[0]] if dates_with_atlas else []

    for i in range(1, len(dates_with_atlas)):
        if (dates_with_atlas[i] - dates_with_atlas[i-1]).days == 1:
            current_streak.append(dates_with_atlas[i])
        else:
            if len(current_streak) >= 2:
                streaks.append(current_streak)
            current_streak = [dates_with_atlas[i]]

    if len(current_streak) >= 2:
        streaks.append(current_streak)

    longest_streak = max(streaks, key=len) if streaks else []

    return {
        'total_plays': len(atlas_plays),
        'first_play': first_play,
        'last_play': last_play,
        'total_hours': total_hours,
        'avg_minutes': avg_minutes,
        'plays_by_date': dict(plays_by_date),
        'plays_by_hour': dict(plays_by_hour),
        'plays_by_weekday': dict(plays_by_weekday),
        'plays_by_month': dict(plays_by_month),
        'plays_by_week': dict(plays_by_week),
        'before_atlas': before_counter.most_common(15),
        'after_atlas': after_counter.most_common(15),
        'platforms': dict(platforms),
        'skip_rate': skipped / len(atlas_plays) * 100 if atlas_plays else 0,
        'peak_weeks': peak_weeks,
        'longest_streak': longest_streak,
        'dates_count': len(dates_with_atlas)
    }

def generate_html(analysis):
    """Generate interactive HTML visualization"""

    # Prepare data for charts
    dates = sorted(analysis['plays_by_date'].keys())
    date_plays = [analysis['plays_by_date'][d] for d in dates]

    hours = list(range(24))
    hour_plays = [analysis['plays_by_hour'].get(h, 0) for h in hours]

    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_plays = [analysis['plays_by_weekday'].get(w, 0) for w in weekdays]

    months = sorted(analysis['plays_by_month'].keys())
    month_plays = [analysis['plays_by_month'][m] for m in months]

    weeks = sorted(analysis['plays_by_week'].keys())
    week_plays = [analysis['plays_by_week'][w] for w in weeks]

    # Before/After tracks
    before_tracks = analysis['before_atlas']
    after_tracks = analysis['after_atlas']

    # Streak info
    streak_info = ""
    if analysis['longest_streak']:
        streak_start = analysis['longest_streak'][0].strftime('%B %d')
        streak_end = analysis['longest_streak'][-1].strftime('%B %d, %Y')
        streak_info = f"{len(analysis['longest_streak'])} days ({streak_start} - {streak_end})"

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ATLAS Deep Dive - Pommelien Thijs</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .hero {{
            text-align: center;
            padding: 60px 20px;
            background: linear-gradient(135deg, rgba(230, 126, 34, 0.3), rgba(155, 89, 182, 0.3));
            border-radius: 20px;
            margin-bottom: 40px;
            position: relative;
            overflow: hidden;
        }}

        .hero::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
            animation: rotate 20s linear infinite;
        }}

        @keyframes rotate {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        .hero h1 {{
            font-size: 4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #e67e22, #9b59b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
            position: relative;
            z-index: 1;
        }}

        .hero .artist {{
            font-size: 1.8rem;
            color: #b0b0b0;
            margin-bottom: 30px;
            position: relative;
            z-index: 1;
        }}

        .hero-stats {{
            display: flex;
            justify-content: center;
            gap: 60px;
            flex-wrap: wrap;
            position: relative;
            z-index: 1;
        }}

        .hero-stat {{
            text-align: center;
        }}

        .hero-stat .number {{
            font-size: 3rem;
            font-weight: 700;
            color: #e67e22;
        }}

        .hero-stat .label {{
            font-size: 1rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 2px;
        }}

        .section {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        .section h2 {{
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: #e67e22;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
        }}

        .chart-container {{
            position: relative;
            height: 300px;
        }}

        .chart-container.tall {{
            height: 400px;
        }}

        .insight-box {{
            background: rgba(230, 126, 34, 0.1);
            border-left: 4px solid #e67e22;
            padding: 20px;
            border-radius: 0 10px 10px 0;
            margin-bottom: 20px;
        }}

        .insight-box h3 {{
            color: #e67e22;
            margin-bottom: 10px;
        }}

        .track-list {{
            list-style: none;
        }}

        .track-list li {{
            padding: 12px 15px;
            background: rgba(255, 255, 255, 0.03);
            margin-bottom: 8px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.2s;
        }}

        .track-list li:hover {{
            background: rgba(255, 255, 255, 0.08);
        }}

        .track-list .count {{
            background: #e67e22;
            color: #000;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
        }}

        .timeline-insight {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}

        .timeline-card {{
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }}

        .timeline-card .value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #e67e22;
        }}

        .timeline-card .label {{
            color: #888;
            font-size: 0.9rem;
            margin-top: 5px;
        }}

        .phase-indicator {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }}

        .phase-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #e67e22;
        }}

        .footer {{
            text-align: center;
            padding: 40px;
            color: #666;
            font-size: 0.9rem;
        }}

        @media (max-width: 768px) {{
            .hero h1 {{
                font-size: 2.5rem;
            }}
            .hero-stats {{
                gap: 30px;
            }}
            .hero-stat .number {{
                font-size: 2rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="hero">
            <h1>ATLAS</h1>
            <p class="artist">by Pommelien Thijs</p>
            <div class="hero-stats">
                <div class="hero-stat">
                    <div class="number">{analysis['total_plays']}</div>
                    <div class="label">Total Plays</div>
                </div>
                <div class="hero-stat">
                    <div class="number">{analysis['total_hours']:.1f}h</div>
                    <div class="label">Hours Listened</div>
                </div>
                <div class="hero-stat">
                    <div class="number">{analysis['dates_count']}</div>
                    <div class="label">Days with Atlas</div>
                </div>
                <div class="hero-stat">
                    <div class="number">{analysis['avg_minutes']:.1f}m</div>
                    <div class="label">Avg Play Duration</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>The Atlas Journey</h2>
            <div class="insight-box">
                <h3>First Play</h3>
                <p>Atlas entered your life on <strong>{analysis['first_play'].strftime('%B %d, %Y at %H:%M')}</strong></p>
            </div>
            <div class="timeline-insight">
                <div class="timeline-card">
                    <div class="value">{analysis['first_play'].strftime('%b %d')}</div>
                    <div class="label">First Listen</div>
                </div>
                <div class="timeline-card">
                    <div class="value">{analysis['last_play'].strftime('%b %d')}</div>
                    <div class="label">Most Recent</div>
                </div>
                <div class="timeline-card">
                    <div class="value">{(analysis['last_play'] - analysis['first_play']).days}</div>
                    <div class="label">Days of Atlas</div>
                </div>
                <div class="timeline-card">
                    <div class="value">{analysis['skip_rate']:.1f}%</div>
                    <div class="label">Skip Rate</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Atlas Plays Over Time</h2>
            <div class="chart-container tall">
                <canvas id="timelineChart"></canvas>
            </div>
        </div>

        <div class="grid">
            <div class="section">
                <h2>Monthly Breakdown</h2>
                <div class="chart-container">
                    <canvas id="monthlyChart"></canvas>
                </div>
            </div>

            <div class="section">
                <h2>Weekly Intensity</h2>
                <div class="chart-container">
                    <canvas id="weeklyChart"></canvas>
                </div>
            </div>
        </div>

        <div class="grid">
            <div class="section">
                <h2>Time of Day</h2>
                <div class="chart-container">
                    <canvas id="hourlyChart"></canvas>
                </div>
            </div>

            <div class="section">
                <h2>Day of Week</h2>
                <div class="chart-container">
                    <canvas id="weekdayChart"></canvas>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>The Atlas Phases</h2>
            <div class="insight-box">
                <h3>Peak Listening Weeks</h3>
                <p>Your most intense Atlas weeks reveal the emotional journey:</p>
            </div>
            <div class="timeline-insight">
                {"".join(f'''<div class="timeline-card">
                    <div class="phase-indicator"><span class="phase-dot"></span>{week}</div>
                    <div class="value">{plays}</div>
                    <div class="label">plays</div>
                </div>''' for week, plays in analysis['peak_weeks'])}
            </div>
            {f'''<div class="insight-box" style="margin-top: 20px;">
                <h3>Longest Streak</h3>
                <p>Your longest consecutive days listening to Atlas: <strong>{streak_info}</strong></p>
            </div>''' if streak_info else ''}
        </div>

        <div class="grid">
            <div class="section">
                <h2>Before Atlas</h2>
                <p style="color: #888; margin-bottom: 15px;">Tracks that led you to Atlas</p>
                <ul class="track-list">
                    {"".join(f'<li><span class="track">{html.escape(track)}</span><span class="count">{count}</span></li>' for track, count in before_tracks[:10])}
                </ul>
            </div>

            <div class="section">
                <h2>After Atlas</h2>
                <p style="color: #888; margin-bottom: 15px;">Where Atlas takes you</p>
                <ul class="track-list">
                    {"".join(f'<li><span class="track">{html.escape(track)}</span><span class="count">{count}</span></li>' for track, count in after_tracks[:10])}
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>Platform Distribution</h2>
            <div class="timeline-insight">
                {"".join(f'''<div class="timeline-card">
                    <div class="value">{count}</div>
                    <div class="label">{platform}</div>
                </div>''' for platform, count in analysis['platforms'].items())}
            </div>
        </div>

        <div class="footer">
            <p>ATLAS Deep Dive Analysis | Generated from Spotify Extended Streaming History</p>
            <p style="margin-top: 10px;">Data from {analysis['first_play'].strftime('%B %Y')} to {analysis['last_play'].strftime('%B %Y')}</p>
        </div>
    </div>

    <script>
        // Chart.js defaults
        Chart.defaults.color = '#888';
        Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';

        const gradientOrange = (ctx) => {{
            const gradient = ctx.chart.ctx.createLinearGradient(0, 0, 0, 300);
            gradient.addColorStop(0, 'rgba(230, 126, 34, 0.8)');
            gradient.addColorStop(1, 'rgba(230, 126, 34, 0.1)');
            return gradient;
        }};

        // Timeline Chart
        new Chart(document.getElementById('timelineChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(dates)},
                datasets: [{{
                    label: 'Plays',
                    data: {json.dumps(date_plays)},
                    borderColor: '#e67e22',
                    backgroundColor: gradientOrange,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 3,
                    pointBackgroundColor: '#e67e22'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ maxTicksLimit: 10 }}
                    }},
                    y: {{
                        beginAtZero: true,
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}
                    }}
                }}
            }}
        }});

        // Monthly Chart
        new Chart(document.getElementById('monthlyChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(months)},
                datasets: [{{
                    label: 'Plays',
                    data: {json.dumps(month_plays)},
                    backgroundColor: '#e67e22',
                    borderRadius: 8
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    x: {{ grid: {{ display: false }} }},
                    y: {{ beginAtZero: true, grid: {{ color: 'rgba(255, 255, 255, 0.05)' }} }}
                }}
            }}
        }});

        // Weekly Chart
        new Chart(document.getElementById('weeklyChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(weeks)},
                datasets: [{{
                    label: 'Plays',
                    data: {json.dumps(week_plays)},
                    backgroundColor: 'rgba(155, 89, 182, 0.8)',
                    borderRadius: 4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ maxTicksLimit: 10, maxRotation: 45 }}
                    }},
                    y: {{ beginAtZero: true, grid: {{ color: 'rgba(255, 255, 255, 0.05)' }} }}
                }}
            }}
        }});

        // Hourly Chart
        new Chart(document.getElementById('hourlyChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps([f"{h:02d}:00" for h in hours])},
                datasets: [{{
                    label: 'Plays',
                    data: {json.dumps(hour_plays)},
                    backgroundColor: (context) => {{
                        const hour = context.dataIndex;
                        if (hour >= 6 && hour < 12) return 'rgba(241, 196, 15, 0.8)';  // Morning
                        if (hour >= 12 && hour < 18) return 'rgba(230, 126, 34, 0.8)'; // Afternoon
                        if (hour >= 18 && hour < 22) return 'rgba(155, 89, 182, 0.8)'; // Evening
                        return 'rgba(52, 73, 94, 0.8)';  // Night
                    }},
                    borderRadius: 4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ maxTicksLimit: 12 }}
                    }},
                    y: {{ beginAtZero: true, grid: {{ color: 'rgba(255, 255, 255, 0.05)' }} }}
                }}
            }}
        }});

        // Weekday Chart
        new Chart(document.getElementById('weekdayChart'), {{
            type: 'polarArea',
            data: {{
                labels: {json.dumps(weekdays)},
                datasets: [{{
                    data: {json.dumps(weekday_plays)},
                    backgroundColor: [
                        'rgba(230, 126, 34, 0.8)',
                        'rgba(155, 89, 182, 0.8)',
                        'rgba(52, 152, 219, 0.8)',
                        'rgba(46, 204, 113, 0.8)',
                        'rgba(241, 196, 15, 0.8)',
                        'rgba(231, 76, 60, 0.8)',
                        'rgba(149, 165, 166, 0.8)'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'right',
                        labels: {{ padding: 15 }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
'''

    return html_content

def main():
    print("=" * 60)
    print("ATLAS DEEP DIVE - Pommelien Thijs")
    print("=" * 60)

    # Load data
    print("\nLoading streaming history...")
    all_streams = load_all_streaming_data()

    # Filter Atlas plays
    print("\nFiltering Atlas plays...")
    atlas_plays = filter_atlas_plays(all_streams)
    print(f"Found {len(atlas_plays)} Atlas plays")

    # Analyze
    print("\nPerforming deep analysis...")
    analysis = analyze_atlas(atlas_plays, all_streams)

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print(f"Total plays: {analysis['total_plays']}")
    print(f"First play: {analysis['first_play']}")
    print(f"Last play: {analysis['last_play']}")
    print(f"Total hours: {analysis['total_hours']:.2f}")
    print(f"Average play duration: {analysis['avg_minutes']:.2f} minutes")
    print(f"Skip rate: {analysis['skip_rate']:.1f}%")
    print(f"Days with Atlas: {analysis['dates_count']}")

    print("\nPeak weeks:")
    for week, plays in analysis['peak_weeks']:
        print(f"  {week}: {plays} plays")

    print("\nMost common tracks before Atlas:")
    for track, count in analysis['before_atlas'][:5]:
        print(f"  {track}: {count}")

    print("\nMost common tracks after Atlas:")
    for track, count in analysis['after_atlas'][:5]:
        print(f"  {track}: {count}")

    # Generate HTML
    print("\nGenerating visualization...")
    html_content = generate_html(analysis)

    # Save HTML
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\nVisualization saved to: {OUTPUT_FILE}")

    # Open in browser
    import webbrowser
    webbrowser.open(str(OUTPUT_FILE))
    print("Opening in browser...")

    return analysis

if __name__ == "__main__":
    main()
