"""
UNITY DAY
=========
The convergence point. Where many becomes one.
1+1=1 in the listening data.

"give me the ending first"
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import math
import webbrowser

DATA_DIR = Path(r"C:\Users\Nouri\Desktop\easiest game of my life\mabrouk_spotify_dataset\Spotify Extended Streaming History")

def load_data():
    files = ["Streaming_History_Audio_2024-2025_3.json", "Streaming_History_Audio_2025_4.json"]
    all_data = []
    for f in files:
        with open(DATA_DIR / f, 'r', encoding='utf-8') as file:
            all_data.extend(json.load(file))
    return all_data

def find_unity(data):
    """Find the convergence point - where all streams become one."""

    # Filter 2025 Pommelien
    pommelien = [
        d for d in data
        if (d.get('master_metadata_album_artist_name') or '').lower() == 'pommelien thijs'
        and d.get('ts', '').startswith('2025')
    ]

    # Parse timestamps
    for p in pommelien:
        p['dt'] = datetime.fromisoformat(p['ts'].replace('Z', '+00:00'))
        p['date'] = p['dt'].date()
        p['hour'] = p['dt'].hour
        p['minutes'] = p.get('ms_played', 0) / 60000

    # Daily aggregation
    daily = defaultdict(lambda: {'minutes': 0, 'plays': 0, 'tracks': set(), 'hours_active': set()})

    for p in pommelien:
        d = p['date']
        daily[d]['minutes'] += p['minutes']
        daily[d]['plays'] += 1
        daily[d]['tracks'].add(p.get('master_metadata_track_name', ''))
        daily[d]['hours_active'].add(p['hour'])

    # Calculate unity metrics for each day
    # Unity = convergence of time, diversity, and intensity
    days_data = []

    for date, stats in daily.items():
        # Intensity: total minutes
        intensity = stats['minutes']

        # Diversity: unique tracks played
        diversity = len(stats['tracks'])

        # Temporal spread: hours active
        temporal_spread = len(stats['hours_active'])

        # Unity Score: when intensity is high, diversity is high, and spread is high
        # The day when EVERYTHING converged
        if intensity > 0 and diversity > 0:
            unity_score = (intensity * diversity * temporal_spread) ** (1/3)  # geometric mean
        else:
            unity_score = 0

        days_data.append({
            'date': date,
            'intensity': intensity,
            'diversity': diversity,
            'temporal_spread': temporal_spread,
            'plays': stats['plays'],
            'unity_score': unity_score
        })

    # Sort by unity score
    days_data.sort(key=lambda x: x['unity_score'], reverse=True)

    # THE UNITY DAY
    unity_day = days_data[0]

    # Get detailed data for unity day
    unity_plays = [p for p in pommelien if p['date'] == unity_day['date']]
    unity_plays.sort(key=lambda x: x['dt'])

    # Track sequence on unity day
    track_sequence = [p.get('master_metadata_track_name', '') for p in unity_plays]

    # Hour by hour on unity day
    hourly = defaultdict(list)
    for p in unity_plays:
        hourly[p['hour']].append(p.get('master_metadata_track_name', ''))

    return unity_day, days_data[:10], unity_plays, hourly, track_sequence

def create_unity_visualization(unity_day, top_days, unity_plays, hourly, track_sequence):
    """Create the convergence visualization."""

    # Calculate the golden ratio spiral positions for tracks
    phi = (1 + math.sqrt(5)) / 2

    # Prepare hourly data
    hours_data = []
    for h in range(24):
        tracks = hourly.get(h, [])
        hours_data.append({
            'hour': h,
            'count': len(tracks),
            'tracks': tracks[:5]  # top 5 per hour
        })

    # Top 10 days data
    top_days_json = json.dumps([{
        'date': str(d['date']),
        'intensity': round(d['intensity'], 1),
        'diversity': d['diversity'],
        'plays': d['plays'],
        'unity_score': round(d['unity_score'], 2)
    } for d in top_days])

    # Track flow data
    track_counts = defaultdict(int)
    for t in track_sequence:
        track_counts[t] += 1
    top_tracks = sorted(track_counts.items(), key=lambda x: -x[1])[:12]

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UNITY DAY | 1+1=1</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Space Mono', monospace;
            background: #000;
            color: #fff;
            min-height: 100vh;
            overflow-x: hidden;
        }}

        .universe {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(ellipse at center, #0a0a1a 0%, #000 100%);
            z-index: -1;
        }}

        .stars {{
            position: fixed;
            width: 100%;
            height: 100%;
            z-index: -1;
        }}

        .star {{
            position: absolute;
            width: 2px;
            height: 2px;
            background: #fff;
            border-radius: 50%;
            animation: twinkle 2s infinite;
        }}

        @keyframes twinkle {{
            0%, 100% {{ opacity: 0.3; }}
            50% {{ opacity: 1; }}
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
            position: relative;
            z-index: 1;
        }}

        .convergence {{
            text-align: center;
            padding: 100px 20px;
            position: relative;
        }}

        .convergence-title {{
            font-size: 1rem;
            letter-spacing: 8px;
            color: #666;
            margin-bottom: 20px;
            text-transform: uppercase;
        }}

        .unity-equation {{
            font-size: 4rem;
            font-weight: 700;
            background: linear-gradient(135deg, #fff 0%, #888 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 40px;
        }}

        .unity-date {{
            font-size: 2.5rem;
            color: #1DB954;
            margin-bottom: 10px;
            text-shadow: 0 0 30px rgba(29, 185, 84, 0.5);
        }}

        .unity-weekday {{
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 60px;
        }}

        .metrics {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 30px;
            margin-bottom: 80px;
        }}

        .metric {{
            text-align: center;
            padding: 30px;
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
        }}

        .metric-value {{
            font-size: 3rem;
            font-weight: 700;
            color: #1DB954;
        }}

        .metric-label {{
            font-size: 0.8rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-top: 10px;
        }}

        .spiral-container {{
            position: relative;
            height: 600px;
            margin: 60px 0;
        }}

        .spiral-center {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 120px;
            height: 120px;
            background: radial-gradient(circle, #1DB954 0%, transparent 70%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            font-weight: 700;
            animation: pulse 2s infinite;
            z-index: 10;
        }}

        @keyframes pulse {{
            0%, 100% {{ transform: translate(-50%, -50%) scale(1); opacity: 1; }}
            50% {{ transform: translate(-50%, -50%) scale(1.1); opacity: 0.8; }}
        }}

        .track-node {{
            position: absolute;
            padding: 8px 16px;
            background: rgba(29, 185, 84, 0.1);
            border: 1px solid rgba(29, 185, 84, 0.3);
            border-radius: 20px;
            font-size: 0.75rem;
            white-space: nowrap;
            transition: all 0.3s;
            cursor: pointer;
        }}

        .track-node:hover {{
            background: rgba(29, 185, 84, 0.3);
            transform: scale(1.1);
        }}

        .track-count {{
            color: #1DB954;
            font-weight: 700;
            margin-left: 8px;
        }}

        .timeline {{
            margin: 80px 0;
        }}

        .timeline-title {{
            font-size: 1rem;
            letter-spacing: 4px;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 30px;
            text-align: center;
        }}

        .hours-grid {{
            display: grid;
            grid-template-columns: repeat(24, 1fr);
            gap: 4px;
            margin-bottom: 20px;
        }}

        .hour-bar {{
            height: 100px;
            background: rgba(29, 185, 84, 0.2);
            border-radius: 4px 4px 0 0;
            position: relative;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            transition: all 0.3s;
        }}

        .hour-bar:hover {{
            background: rgba(29, 185, 84, 0.4);
        }}

        .hour-fill {{
            background: linear-gradient(to top, #1DB954, #4ade80);
            border-radius: 4px 4px 0 0;
            transition: height 0.5s ease;
        }}

        .hour-label {{
            position: absolute;
            bottom: -25px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 0.6rem;
            color: #666;
        }}

        .convergence-paths {{
            margin: 80px 0;
            padding: 40px;
            background: rgba(255,255,255,0.02);
            border-radius: 20px;
        }}

        .path-title {{
            font-size: 1rem;
            letter-spacing: 4px;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 30px;
        }}

        .top-days {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}

        .day-card {{
            padding: 20px;
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            transition: all 0.3s;
        }}

        .day-card:first-child {{
            border-color: #1DB954;
            background: rgba(29, 185, 84, 0.1);
        }}

        .day-card:hover {{
            transform: translateY(-5px);
        }}

        .day-date {{
            font-size: 0.9rem;
            color: #fff;
            margin-bottom: 10px;
        }}

        .day-stats {{
            font-size: 0.75rem;
            color: #666;
        }}

        .day-unity {{
            font-size: 1.5rem;
            color: #1DB954;
            font-weight: 700;
            margin-top: 10px;
        }}

        .philosophy {{
            text-align: center;
            padding: 80px 20px;
            margin-top: 60px;
        }}

        .philosophy-text {{
            font-size: 1.2rem;
            color: #666;
            max-width: 600px;
            margin: 0 auto;
            line-height: 2;
        }}

        .philosophy-text em {{
            color: #1DB954;
            font-style: normal;
        }}

        .end-symbol {{
            font-size: 4rem;
            margin-top: 40px;
            opacity: 0.3;
        }}

        @media (max-width: 768px) {{
            .metrics {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .unity-equation {{
                font-size: 2.5rem;
            }}
            .hours-grid {{
                grid-template-columns: repeat(12, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="universe"></div>
    <div class="stars" id="stars"></div>

    <div class="container">
        <div class="convergence">
            <div class="convergence-title">The Convergence Point</div>
            <div class="unity-equation">1 + 1 = 1</div>
            <div class="unity-date">{unity_day['date'].strftime('%B %d, %Y')}</div>
            <div class="unity-weekday">{unity_day['date'].strftime('%A')}</div>
        </div>

        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{unity_day['intensity']:.0f}</div>
                <div class="metric-label">Minutes</div>
            </div>
            <div class="metric">
                <div class="metric-value">{unity_day['plays']}</div>
                <div class="metric-label">Plays</div>
            </div>
            <div class="metric">
                <div class="metric-value">{unity_day['diversity']}</div>
                <div class="metric-label">Unique Tracks</div>
            </div>
            <div class="metric">
                <div class="metric-value">{unity_day['temporal_spread']}</div>
                <div class="metric-label">Hours Active</div>
            </div>
        </div>

        <div class="spiral-container" id="spiral">
            <div class="spiral-center">1</div>
        </div>

        <div class="timeline">
            <div class="timeline-title">24-Hour Flow</div>
            <div class="hours-grid" id="hoursGrid"></div>
        </div>

        <div class="convergence-paths">
            <div class="path-title">Paths to Unity (Top 10 Days)</div>
            <div class="top-days" id="topDays"></div>
        </div>

        <div class="philosophy">
            <div class="philosophy-text">
                On this day, <em>{unity_day['plays']} plays</em> across <em>{unity_day['diversity']} tracks</em>
                spanning <em>{unity_day['temporal_spread']} hours</em> converged into a single point of experience.
                <br><br>
                Many became one. Diversity unified. Time collapsed.
                <br><br>
                <em>This is Unity Day.</em>
            </div>
            <div class="end-symbol">&#8734;</div>
        </div>
    </div>

    <script>
        // Generate stars
        const starsContainer = document.getElementById('stars');
        for (let i = 0; i < 200; i++) {{
            const star = document.createElement('div');
            star.className = 'star';
            star.style.left = Math.random() * 100 + '%';
            star.style.top = Math.random() * 100 + '%';
            star.style.animationDelay = Math.random() * 2 + 's';
            star.style.opacity = Math.random() * 0.5 + 0.3;
            starsContainer.appendChild(star);
        }}

        // Top tracks spiral
        const tracks = {json.dumps(top_tracks)};
        const spiral = document.getElementById('spiral');
        const centerX = spiral.offsetWidth / 2;
        const centerY = 300;
        const phi = 1.618033988749895;

        tracks.forEach((track, i) => {{
            const angle = i * phi * Math.PI * 0.5;
            const radius = 80 + i * 35;
            const x = centerX + radius * Math.cos(angle);
            const y = centerY + radius * Math.sin(angle);

            const node = document.createElement('div');
            node.className = 'track-node';
            node.innerHTML = track[0].substring(0, 25) + (track[0].length > 25 ? '...' : '') +
                           '<span class="track-count">' + track[1] + '</span>';
            node.style.left = x + 'px';
            node.style.top = y + 'px';
            node.style.transform = 'translate(-50%, -50%)';
            node.style.animationDelay = (i * 0.1) + 's';
            spiral.appendChild(node);
        }});

        // Hours grid
        const hoursData = {json.dumps(hours_data)};
        const maxCount = Math.max(...hoursData.map(h => h.count));
        const hoursGrid = document.getElementById('hoursGrid');

        hoursData.forEach(h => {{
            const bar = document.createElement('div');
            bar.className = 'hour-bar';

            const fill = document.createElement('div');
            fill.className = 'hour-fill';
            fill.style.height = (h.count / maxCount * 100) + '%';

            const label = document.createElement('div');
            label.className = 'hour-label';
            label.textContent = h.hour.toString().padStart(2, '0');

            bar.appendChild(fill);
            bar.appendChild(label);
            bar.title = h.count + ' plays at ' + h.hour + ':00';
            hoursGrid.appendChild(bar);
        }});

        // Top days
        const topDays = {top_days_json};
        const topDaysContainer = document.getElementById('topDays');

        topDays.forEach((day, i) => {{
            const card = document.createElement('div');
            card.className = 'day-card';
            card.innerHTML = `
                <div class="day-date">${{day.date}}</div>
                <div class="day-stats">${{day.plays}} plays | ${{day.diversity}} tracks | ${{day.intensity}} min</div>
                <div class="day-unity">${{day.unity_score}}</div>
            `;
            topDaysContainer.appendChild(card);
        }});
    </script>
</body>
</html>'''

    return html

def main():
    print("Loading data...")
    data = load_data()

    print("Finding Unity Day...")
    unity_day, top_days, unity_plays, hourly, track_sequence = find_unity(data)

    print(f"\n{'='*60}")
    print("UNITY DAY FOUND")
    print(f"{'='*60}")
    print(f"Date: {unity_day['date']}")
    print(f"Intensity: {unity_day['intensity']:.0f} minutes")
    print(f"Plays: {unity_day['plays']}")
    print(f"Unique Tracks: {unity_day['diversity']}")
    print(f"Hours Active: {unity_day['temporal_spread']}")
    print(f"Unity Score: {unity_day['unity_score']:.2f}")

    print("\nTop 5 Convergence Days:")
    for d in top_days[:5]:
        print(f"  {d['date']}: score={d['unity_score']:.2f}, {d['plays']} plays, {d['diversity']} tracks")

    print("\nGenerating visualization...")
    html = create_unity_visualization(unity_day, top_days, unity_plays, hourly, track_sequence)

    output = Path(r"C:\Users\Nouri\Documents\GitHub\Nouri-Mabrouk\experiments\unity_day.html")
    with open(output, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Saved to: {output}")
    webbrowser.open(f'file:///{output}')

    return unity_day

if __name__ == "__main__":
    main()
