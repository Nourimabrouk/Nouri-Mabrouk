"""
Pommelien Thijs Track Transitions Analysis for 2025
Analyzes track-to-track transitions and creates a chord diagram visualization
"""

import json
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
import webbrowser

# Data paths
DATA_DIR = Path(r"C:\Users\Nouri\Desktop\easiest game of my life\mabrouk_spotify_dataset\Spotify Extended Streaming History")
OUTPUT_PATH = Path(r"C:\Users\Nouri\Documents\GitHub\Nouri-Mabrouk\experiments\track_transitions.html")

def load_spotify_data():
    """Load all JSON files and filter for 2025 data"""
    all_plays = []

    for json_file in DATA_DIR.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for play in data:
                ts = play.get('ts', '')
                if ts.startswith('2025'):
                    all_plays.append(play)

    # Sort by timestamp
    all_plays.sort(key=lambda x: x.get('ts', ''))
    return all_plays

def filter_pommelien_context(plays, session_gap_minutes=30):
    """
    Extract sessions that contain Pommelien Thijs tracks
    A session is defined as consecutive plays with gaps < session_gap_minutes
    """
    sessions = []
    current_session = []
    prev_time = None

    for play in plays:
        ts_str = play.get('ts', '')
        if not ts_str:
            continue

        current_time = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))

        # Check if this is a new session
        if prev_time and (current_time - prev_time) > timedelta(minutes=session_gap_minutes):
            if current_session:
                sessions.append(current_session)
            current_session = []

        current_session.append(play)
        prev_time = current_time

    # Don't forget the last session
    if current_session:
        sessions.append(current_session)

    # Filter to sessions containing Pommelien Thijs
    pommelien_sessions = []
    for session in sessions:
        has_pommelien = any(
            (play.get('master_metadata_album_artist_name') or '').lower() == 'pommelien thijs'
            for play in session
        )
        if has_pommelien:
            pommelien_sessions.append(session)

    return pommelien_sessions

def get_pommelien_only_plays(plays):
    """Get only Pommelien Thijs plays from all 2025 data"""
    return [
        play for play in plays
        if (play.get('master_metadata_album_artist_name') or '').lower() == 'pommelien thijs'
    ]

def build_transition_matrix(sessions):
    """
    Build a transition matrix for Pommelien Thijs tracks
    Only counts transitions between consecutive Pommelien tracks in sessions
    """
    transitions = defaultdict(lambda: defaultdict(int))

    for session in sessions:
        # Get only Pommelien tracks in this session, maintaining order
        pommelien_tracks = [
            play for play in session
            if (play.get('master_metadata_album_artist_name') or '').lower() == 'pommelien thijs'
            and play.get('ms_played', 0) > 30000  # More than 30 seconds played
        ]

        # Count transitions within Pommelien tracks
        for i in range(len(pommelien_tracks) - 1):
            from_track = pommelien_tracks[i].get('master_metadata_track_name', 'Unknown')
            to_track = pommelien_tracks[i + 1].get('master_metadata_track_name', 'Unknown')
            transitions[from_track][to_track] += 1

    return transitions

def find_session_starters_and_enders(sessions):
    """Find which Pommelien tracks start and end sessions most often"""
    starters = Counter()
    enders = Counter()

    for session in sessions:
        pommelien_tracks = [
            play for play in session
            if (play.get('master_metadata_album_artist_name') or '').lower() == 'pommelien thijs'
            and play.get('ms_played', 0) > 30000
        ]

        if pommelien_tracks:
            first_track = pommelien_tracks[0].get('master_metadata_track_name', 'Unknown')
            last_track = pommelien_tracks[-1].get('master_metadata_track_name', 'Unknown')
            starters[first_track] += 1
            enders[last_track] += 1

    return starters, enders

def find_playlist_loops(sessions, min_sequence_length=3):
    """Detect repeated sequences (playlist loops)"""
    sequences = []

    for session in sessions:
        pommelien_tracks = [
            play.get('master_metadata_track_name', 'Unknown')
            for play in session
            if (play.get('master_metadata_album_artist_name') or '').lower() == 'pommelien thijs'
            and play.get('ms_played', 0) > 30000
        ]

        if len(pommelien_tracks) >= min_sequence_length:
            # Create sequence tuple
            seq = tuple(pommelien_tracks)
            sequences.append(seq)

    # Find repeated sequences
    sequence_counts = Counter(sequences)
    repeated = {seq: count for seq, count in sequence_counts.items() if count > 1}

    # Also look for subsequences
    all_subsequences = []
    for session in sessions:
        pommelien_tracks = [
            play.get('master_metadata_track_name', 'Unknown')
            for play in session
            if (play.get('master_metadata_album_artist_name') or '').lower() == 'pommelien thijs'
            and play.get('ms_played', 0) > 30000
        ]

        # Extract all subsequences of length 2-5
        for length in range(2, min(6, len(pommelien_tracks) + 1)):
            for i in range(len(pommelien_tracks) - length + 1):
                subseq = tuple(pommelien_tracks[i:i+length])
                all_subsequences.append(subseq)

    subseq_counts = Counter(all_subsequences)
    repeated_subsequences = {seq: count for seq, count in subseq_counts.items() if count > 1}

    return repeated, repeated_subsequences

def find_ritualistic_patterns(plays):
    """Find patterns based on time of day or day of week"""
    time_patterns = defaultdict(list)
    day_patterns = defaultdict(list)

    pommelien_plays = [
        play for play in plays
        if (play.get('master_metadata_album_artist_name') or '').lower() == 'pommelien thijs'
        and play.get('ms_played', 0) > 30000
    ]

    for play in pommelien_plays:
        ts_str = play.get('ts', '')
        if not ts_str:
            continue

        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        hour = dt.hour
        day = dt.strftime('%A')
        track = play.get('master_metadata_track_name', 'Unknown')

        # Group by hour ranges
        if 5 <= hour < 12:
            time_patterns['morning'].append(track)
        elif 12 <= hour < 17:
            time_patterns['afternoon'].append(track)
        elif 17 <= hour < 21:
            time_patterns['evening'].append(track)
        else:
            time_patterns['night'].append(track)

        day_patterns[day].append(track)

    # Find most common tracks per time period
    time_favorites = {
        period: Counter(tracks).most_common(5)
        for period, tracks in time_patterns.items()
    }

    day_favorites = {
        day: Counter(tracks).most_common(3)
        for day, tracks in day_patterns.items()
    }

    return time_favorites, day_favorites

def create_chord_diagram_html(transitions, starters, enders, loops, subseq_loops, time_patterns, day_patterns):
    """Create an interactive chord diagram visualization"""

    # Get all unique tracks
    all_tracks = set()
    for from_track, to_tracks in transitions.items():
        all_tracks.add(from_track)
        for to_track in to_tracks.keys():
            all_tracks.add(to_track)

    all_tracks = sorted(list(all_tracks))
    track_to_idx = {track: idx for idx, track in enumerate(all_tracks)}

    # Build matrix for chord diagram
    n = len(all_tracks)
    matrix = [[0] * n for _ in range(n)]

    for from_track, to_tracks in transitions.items():
        for to_track, count in to_tracks.items():
            if from_track in track_to_idx and to_track in track_to_idx:
                matrix[track_to_idx[from_track]][track_to_idx[to_track]] = count

    # Calculate total transitions for each track
    track_totals = {}
    for track in all_tracks:
        outgoing = sum(transitions.get(track, {}).values())
        incoming = sum(transitions.get(other, {}).get(track, 0) for other in all_tracks)
        track_totals[track] = outgoing + incoming

    # Format analysis results for display
    starters_html = ""
    for track, count in starters.most_common(10):
        starters_html += f"<tr><td>{track}</td><td>{count}</td></tr>"

    enders_html = ""
    for track, count in enders.most_common(10):
        enders_html += f"<tr><td>{track}</td><td>{count}</td></tr>"

    # Format loops
    loops_html = ""
    if loops:
        for seq, count in sorted(loops.items(), key=lambda x: -x[1])[:10]:
            loops_html += f"<tr><td>{' -> '.join(seq)}</td><td>{count}</td></tr>"
    else:
        loops_html = "<tr><td colspan='2'>No exact repeated sequences found</td></tr>"

    # Format subsequence loops (more interesting)
    subseq_html = ""
    sorted_subseqs = sorted(subseq_loops.items(), key=lambda x: (-x[1], -len(x[0])))[:15]
    for seq, count in sorted_subseqs:
        subseq_html += f"<tr><td>{' -> '.join(seq)}</td><td>{count}</td></tr>"

    # Format time patterns
    time_html = ""
    for period in ['morning', 'afternoon', 'evening', 'night']:
        if period in time_patterns:
            tracks = ", ".join([f"{t[0]} ({t[1]})" for t in time_patterns[period][:3]])
            time_html += f"<tr><td>{period.capitalize()}</td><td>{tracks}</td></tr>"

    # Format transition matrix as table
    matrix_html = "<table class='matrix-table'><tr><th></th>"
    for track in all_tracks:
        short_name = track[:15] + "..." if len(track) > 15 else track
        matrix_html += f"<th title='{track}'>{short_name}</th>"
    matrix_html += "</tr>"

    for i, from_track in enumerate(all_tracks):
        short_from = from_track[:15] + "..." if len(from_track) > 15 else from_track
        matrix_html += f"<tr><td title='{from_track}'><strong>{short_from}</strong></td>"
        for j in range(n):
            val = matrix[i][j]
            bg_color = f"rgba(99, 102, 241, {min(val / 5, 1)})" if val > 0 else "transparent"
            matrix_html += f"<td style='background-color: {bg_color}'>{val if val > 0 else ''}</td>"
        matrix_html += "</tr>"
    matrix_html += "</table>"

    # Create the HTML with D3.js chord diagram
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pommelien Thijs Track Transitions 2025</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        h1 {{
            text-align: center;
            color: #7c3aed;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}

        .subtitle {{
            text-align: center;
            color: #a78bfa;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}

        .dashboard {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}

        .card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        .card h2 {{
            color: #a78bfa;
            margin-bottom: 15px;
            font-size: 1.3em;
            border-bottom: 2px solid rgba(167, 139, 250, 0.3);
            padding-bottom: 10px;
        }}

        .full-width {{
            grid-column: 1 / -1;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}

        th {{
            color: #a78bfa;
            font-weight: 600;
        }}

        tr:hover {{
            background: rgba(255,255,255,0.05);
        }}

        #chord-diagram {{
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 600px;
        }}

        .chord-tooltip {{
            position: absolute;
            background: rgba(30, 30, 60, 0.95);
            border: 1px solid #7c3aed;
            border-radius: 8px;
            padding: 10px 15px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 1000;
        }}

        .matrix-table {{
            font-size: 0.75em;
            overflow-x: auto;
            display: block;
        }}

        .matrix-table th, .matrix-table td {{
            padding: 4px 6px;
            text-align: center;
            white-space: nowrap;
            min-width: 40px;
        }}

        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 20px;
            justify-content: center;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 0.85em;
        }}

        .legend-color {{
            width: 15px;
            height: 15px;
            border-radius: 3px;
        }}

        .highlight {{
            color: #f59e0b;
            font-weight: bold;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            text-align: center;
        }}

        .stat-box {{
            background: rgba(124, 58, 237, 0.2);
            border-radius: 10px;
            padding: 15px;
        }}

        .stat-number {{
            font-size: 2em;
            color: #7c3aed;
            font-weight: bold;
        }}

        .stat-label {{
            font-size: 0.85em;
            color: #a78bfa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Pommelien Thijs Track Transitions</h1>
        <p class="subtitle">Analyzing listening patterns in 2025</p>

        <div class="dashboard">
            <div class="card full-width">
                <h2>Quick Stats</h2>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-number">{len(all_tracks)}</div>
                        <div class="stat-label">Unique Tracks</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{sum(sum(row) for row in matrix)}</div>
                        <div class="stat-label">Total Transitions</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{sum(starters.values())}</div>
                        <div class="stat-label">Session Starts</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{len(subseq_loops)}</div>
                        <div class="stat-label">Repeated Patterns</div>
                    </div>
                </div>
            </div>

            <div class="card full-width">
                <h2>Track Flow Network</h2>
                <div id="chord-diagram"></div>
                <div class="legend" id="legend"></div>
            </div>

            <div class="card">
                <h2>Session Starters</h2>
                <p style="margin-bottom: 10px; font-size: 0.9em; color: #888;">Which tracks start Pommelien listening sessions</p>
                <table>
                    <tr><th>Track</th><th>Count</th></tr>
                    {starters_html}
                </table>
            </div>

            <div class="card">
                <h2>Session Enders</h2>
                <p style="margin-bottom: 10px; font-size: 0.9em; color: #888;">Which tracks end Pommelien listening sessions</p>
                <table>
                    <tr><th>Track</th><th>Count</th></tr>
                    {enders_html}
                </table>
            </div>

            <div class="card">
                <h2>Repeated Sequences (Loops)</h2>
                <p style="margin-bottom: 10px; font-size: 0.9em; color: #888;">Track patterns that repeat across sessions</p>
                <table>
                    <tr><th>Sequence</th><th>Count</th></tr>
                    {subseq_html}
                </table>
            </div>

            <div class="card">
                <h2>Time-of-Day Patterns</h2>
                <p style="margin-bottom: 10px; font-size: 0.9em; color: #888;">Favorite tracks by time of day</p>
                <table>
                    <tr><th>Period</th><th>Top Tracks</th></tr>
                    {time_html}
                </table>
            </div>

            <div class="card full-width">
                <h2>Transition Matrix</h2>
                <p style="margin-bottom: 10px; font-size: 0.9em; color: #888;">Rows = From Track, Columns = To Track. Darker = More Transitions</p>
                <div style="overflow-x: auto;">
                    {matrix_html}
                </div>
            </div>
        </div>
    </div>

    <div class="chord-tooltip" id="tooltip"></div>

    <script>
        // Data for chord diagram
        const tracks = {json.dumps(all_tracks)};
        const matrix = {json.dumps(matrix)};
        const trackTotals = {json.dumps(track_totals)};

        // Create chord diagram
        const width = 700;
        const height = 700;
        const innerRadius = Math.min(width, height) * 0.35;
        const outerRadius = innerRadius + 20;

        // Color scale
        const color = d3.scaleOrdinal()
            .domain(tracks)
            .range(d3.schemeTableau10.concat(d3.schemePastel1));

        // Create SVG
        const svg = d3.select("#chord-diagram")
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .append("g")
            .attr("transform", `translate(${{width/2}},${{height/2}})`);

        // Create chord layout
        const chord = d3.chord()
            .padAngle(0.05)
            .sortSubgroups(d3.descending)
            .sortChords(d3.descending);

        const chords = chord(matrix);

        // Create arc generator
        const arc = d3.arc()
            .innerRadius(innerRadius)
            .outerRadius(outerRadius);

        // Create ribbon generator
        const ribbon = d3.ribbon()
            .radius(innerRadius);

        // Tooltip
        const tooltip = d3.select("#tooltip");

        // Draw outer arcs (track groups)
        const group = svg.append("g")
            .selectAll("g")
            .data(chords.groups)
            .join("g");

        group.append("path")
            .attr("fill", d => color(tracks[d.index]))
            .attr("stroke", "#fff")
            .attr("stroke-width", 0.5)
            .attr("d", arc)
            .on("mouseover", function(event, d) {{
                tooltip.style("opacity", 1)
                    .html(`<strong>${{tracks[d.index]}}</strong><br>Total connections: ${{trackTotals[tracks[d.index]] || 0}}`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 10) + "px");
            }})
            .on("mouseout", () => tooltip.style("opacity", 0));

        // Add labels
        group.append("text")
            .each(d => {{ d.angle = (d.startAngle + d.endAngle) / 2; }})
            .attr("dy", "0.35em")
            .attr("transform", d => `
                rotate(${{(d.angle * 180 / Math.PI - 90)}})
                translate(${{outerRadius + 10}})
                ${{d.angle > Math.PI ? "rotate(180)" : ""}}
            `)
            .attr("text-anchor", d => d.angle > Math.PI ? "end" : null)
            .attr("fill", "#e0e0e0")
            .attr("font-size", "10px")
            .text(d => tracks[d.index].length > 20 ? tracks[d.index].substring(0, 20) + "..." : tracks[d.index]);

        // Draw ribbons (transitions)
        svg.append("g")
            .attr("fill-opacity", 0.67)
            .selectAll("path")
            .data(chords)
            .join("path")
            .attr("d", ribbon)
            .attr("fill", d => color(tracks[d.source.index]))
            .attr("stroke", "#fff")
            .attr("stroke-width", 0.3)
            .on("mouseover", function(event, d) {{
                d3.select(this).attr("fill-opacity", 1);
                tooltip.style("opacity", 1)
                    .html(`<strong>${{tracks[d.source.index]}}</strong> -> <strong>${{tracks[d.target.index]}}</strong><br>Transitions: ${{d.source.value}}`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 10) + "px");
            }})
            .on("mouseout", function() {{
                d3.select(this).attr("fill-opacity", 0.67);
                tooltip.style("opacity", 0);
            }});

        // Create legend
        const legend = d3.select("#legend");
        tracks.forEach((track, i) => {{
            const item = legend.append("div").attr("class", "legend-item");
            item.append("div")
                .attr("class", "legend-color")
                .style("background-color", color(track));
            item.append("span").text(track.length > 25 ? track.substring(0, 25) + "..." : track);
        }});
    </script>
</body>
</html>
'''

    return html_content

def main():
    print("Loading Spotify data...")
    plays = load_spotify_data()
    print(f"Found {len(plays)} plays in 2025")

    pommelien_plays = get_pommelien_only_plays(plays)
    print(f"Found {len(pommelien_plays)} Pommelien Thijs plays in 2025")

    print("\nExtracting sessions...")
    sessions = filter_pommelien_context(plays)
    print(f"Found {len(sessions)} sessions containing Pommelien Thijs")

    print("\nBuilding transition matrix...")
    transitions = build_transition_matrix(sessions)

    print("\nFinding session starters and enders...")
    starters, enders = find_session_starters_and_enders(sessions)

    print("\nDetecting playlist loops...")
    loops, subseq_loops = find_playlist_loops(sessions)

    print("\nAnalyzing ritualistic patterns...")
    time_patterns, day_patterns = find_ritualistic_patterns(plays)

    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)

    print("\n--- Top Session Starters ---")
    for track, count in starters.most_common(5):
        print(f"  {track}: {count} times")

    print("\n--- Top Session Enders ---")
    for track, count in enders.most_common(5):
        print(f"  {track}: {count} times")

    print("\n--- Most Common Transitions ---")
    flat_transitions = []
    for from_track, to_tracks in transitions.items():
        for to_track, count in to_tracks.items():
            flat_transitions.append((from_track, to_track, count))
    flat_transitions.sort(key=lambda x: -x[2])
    for from_t, to_t, count in flat_transitions[:10]:
        print(f"  {from_t} -> {to_t}: {count} times")

    print("\n--- Repeated Sequences ---")
    sorted_subseqs = sorted(subseq_loops.items(), key=lambda x: (-x[1], -len(x[0])))[:5]
    for seq, count in sorted_subseqs:
        print(f"  {' -> '.join(seq)}: {count} times")

    print("\n--- Time of Day Patterns ---")
    for period, tracks in time_patterns.items():
        if tracks:
            top = tracks[0]
            print(f"  {period.capitalize()}: {top[0]} ({top[1]} plays)")

    print("\nGenerating visualization...")
    html = create_chord_diagram_html(
        transitions, starters, enders, loops, subseq_loops,
        time_patterns, day_patterns
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nVisualization saved to: {OUTPUT_PATH}")

    # Open in browser
    webbrowser.open(str(OUTPUT_PATH))
    print("Opened in browser!")

if __name__ == "__main__":
    main()
