"""
Spotify Extended Streaming History Analysis
============================================
Comprehensive analysis of 14+ years of listening data (2011-2025)
"""

import json
import os
import sys
import io
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Fix stdout encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Data source path
DATA_PATH = Path(r"C:\Users\Nouri\Desktop\easiest game of my life\mabrouk_spotify_dataset\Spotify Extended Streaming History")

def load_all_streaming_data():
    """Load all JSON streaming history files."""
    all_streams = []
    files = sorted(DATA_PATH.glob("Streaming_History_Audio_*.json"))

    for file in files:
        print(f"Loading {file.name}...")
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_streams.extend(data)
            print(f"  -> {len(data):,} streams")

    print(f"\nTotal streams loaded: {len(all_streams):,}")
    return all_streams

def analyze_streams(streams):
    """Comprehensive analysis of streaming data."""

    # Initialize counters and aggregators
    artist_plays = Counter()
    artist_time = Counter()
    track_plays = Counter()
    track_time = Counter()
    album_plays = Counter()
    year_plays = Counter()
    month_plays = Counter()
    hour_plays = Counter()
    weekday_plays = Counter()
    platform_plays = Counter()
    country_plays = Counter()

    # Detailed tracking
    artist_tracks = defaultdict(set)
    yearly_top_artists = defaultdict(Counter)
    monthly_listening_hours = defaultdict(float)
    skip_rate = {'skipped': 0, 'completed': 0}
    shuffle_stats = {'shuffle': 0, 'no_shuffle': 0}

    # Track evolution over time
    first_listen = {}
    artist_first_listen = {}

    total_ms = 0
    valid_streams = 0

    for stream in streams:
        # Skip entries without track name
        if not stream.get('master_metadata_track_name'):
            continue

        valid_streams += 1

        artist = stream.get('master_metadata_album_artist_name', 'Unknown')
        track = stream.get('master_metadata_track_name', 'Unknown')
        album = stream.get('master_metadata_album_album_name', 'Unknown')
        ms_played = stream.get('ms_played', 0)
        platform = stream.get('platform', 'Unknown')
        country = stream.get('conn_country', 'Unknown')
        skipped = stream.get('skipped', False)
        shuffle = stream.get('shuffle', False)

        # Parse timestamp
        ts_str = stream.get('ts', '')
        try:
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            year = ts.year
            month = f"{ts.year}-{ts.month:02d}"
            hour = ts.hour
            weekday = ts.strftime('%A')
        except:
            year = 0
            month = "Unknown"
            hour = 0
            weekday = "Unknown"

        # Aggregate stats
        total_ms += ms_played

        artist_plays[artist] += 1
        artist_time[artist] += ms_played

        track_key = f"{track} - {artist}"
        track_plays[track_key] += 1
        track_time[track_key] += ms_played

        album_key = f"{album} - {artist}"
        album_plays[album_key] += 1

        year_plays[year] += 1
        month_plays[month] += 1
        hour_plays[hour] += 1
        weekday_plays[weekday] += 1
        platform_plays[platform] += 1
        country_plays[country] += 1

        artist_tracks[artist].add(track)
        yearly_top_artists[year][artist] += 1
        monthly_listening_hours[month] += ms_played / 3600000  # Convert to hours

        if skipped:
            skip_rate['skipped'] += 1
        else:
            skip_rate['completed'] += 1

        if shuffle:
            shuffle_stats['shuffle'] += 1
        else:
            shuffle_stats['no_shuffle'] += 1

        # Track first listens
        if track_key not in first_listen:
            first_listen[track_key] = ts_str
        if artist not in artist_first_listen:
            artist_first_listen[artist] = ts_str

    return {
        'total_streams': valid_streams,
        'total_ms': total_ms,
        'artist_plays': artist_plays,
        'artist_time': artist_time,
        'track_plays': track_plays,
        'track_time': track_time,
        'album_plays': album_plays,
        'year_plays': year_plays,
        'month_plays': month_plays,
        'hour_plays': hour_plays,
        'weekday_plays': weekday_plays,
        'platform_plays': platform_plays,
        'country_plays': country_plays,
        'artist_tracks': artist_tracks,
        'yearly_top_artists': yearly_top_artists,
        'monthly_listening_hours': monthly_listening_hours,
        'skip_rate': skip_rate,
        'shuffle_stats': shuffle_stats,
        'first_listen': first_listen,
        'artist_first_listen': artist_first_listen,
    }

def format_time(ms):
    """Convert milliseconds to human-readable format."""
    hours = ms / 3600000
    if hours >= 24:
        days = hours / 24
        return f"{days:.1f} days ({hours:.0f} hours)"
    return f"{hours:.1f} hours"

def print_report(stats):
    """Print comprehensive analysis report."""

    print("\n" + "="*80)
    print("SPOTIFY LISTENING HISTORY ANALYSIS")
    print("="*80)

    # Overview
    total_hours = stats['total_ms'] / 3600000
    total_days = total_hours / 24
    print(f"\n[OVERVIEW]")
    print(f"   Total streams: {stats['total_streams']:,}")
    print(f"   Total listening time: {total_hours:,.0f} hours ({total_days:.1f} days)")
    print(f"   Unique artists: {len(stats['artist_plays']):,}")
    print(f"   Unique tracks: {len(stats['track_plays']):,}")
    print(f"   Unique albums: {len(stats['album_plays']):,}")

    # Date range
    years = [y for y in stats['year_plays'].keys() if y > 0]
    if years:
        print(f"   Date range: {min(years)} - {max(years)}")

    # Skip rate
    total_plays = stats['skip_rate']['skipped'] + stats['skip_rate']['completed']
    if total_plays > 0:
        skip_pct = stats['skip_rate']['skipped'] / total_plays * 100
        print(f"   Skip rate: {skip_pct:.1f}%")

    # Shuffle stats
    total_shuffle = stats['shuffle_stats']['shuffle'] + stats['shuffle_stats']['no_shuffle']
    if total_shuffle > 0:
        shuffle_pct = stats['shuffle_stats']['shuffle'] / total_shuffle * 100
        print(f"   Shuffle mode: {shuffle_pct:.1f}%")

    # Top Artists by Play Count
    print(f"\n[TOP 25 ARTISTS by play count]")
    print("-" * 60)
    for i, (artist, count) in enumerate(stats['artist_plays'].most_common(25), 1):
        time_ms = stats['artist_time'][artist]
        unique_tracks = len(stats['artist_tracks'][artist])
        print(f"   {i:2}. {artist[:40]:<40} {count:>6,} plays | {format_time(time_ms):>15} | {unique_tracks:>3} tracks")

    # Top Artists by Time
    print(f"\n[TOP 25 ARTISTS by listening time]")
    print("-" * 60)
    for i, (artist, time_ms) in enumerate(stats['artist_time'].most_common(25), 1):
        plays = stats['artist_plays'][artist]
        avg_per_play = (time_ms / plays / 1000) if plays > 0 else 0
        print(f"   {i:2}. {artist[:40]:<40} {format_time(time_ms):>15} | {plays:>6,} plays | {avg_per_play:.0f}s avg")

    # Top Tracks
    print(f"\n[TOP 50 TRACKS by play count]")
    print("-" * 80)
    for i, (track, count) in enumerate(stats['track_plays'].most_common(50), 1):
        time_ms = stats['track_time'][track]
        print(f"   {i:2}. {track[:60]:<60} {count:>5,} | {format_time(time_ms):>10}")

    # Top Tracks by Time
    print(f"\n[TOP 25 TRACKS by listening time]")
    print("-" * 80)
    for i, (track, time_ms) in enumerate(stats['track_time'].most_common(25), 1):
        plays = stats['track_plays'][track]
        print(f"   {i:2}. {track[:60]:<60} {format_time(time_ms):>10} | {plays:>5,}")

    # Yearly breakdown
    print(f"\n[YEARLY BREAKDOWN]")
    print("-" * 60)
    for year in sorted([y for y in stats['year_plays'].keys() if y > 0]):
        plays = stats['year_plays'][year]
        yearly_top = stats['yearly_top_artists'][year].most_common(3)
        top_artists_str = ", ".join([f"{a}" for a, _ in yearly_top])
        print(f"   {year}: {plays:>6,} plays | Top: {top_artists_str[:50]}")

    # Top artists per year
    print(f"\n[TOP ARTISTS PER YEAR]")
    print("-" * 60)
    for year in sorted([y for y in stats['yearly_top_artists'].keys() if y > 0]):
        top = stats['yearly_top_artists'][year].most_common(5)
        print(f"   {year}:")
        for i, (artist, count) in enumerate(top, 1):
            print(f"      {i}. {artist[:40]:<40} {count:>5,} plays")

    # Listening by hour
    print(f"\n[LISTENING BY HOUR OF DAY]")
    print("-" * 60)
    max_hour_plays = max(stats['hour_plays'].values()) if stats['hour_plays'] else 1
    for hour in range(24):
        plays = stats['hour_plays'].get(hour, 0)
        bar_len = int(plays / max_hour_plays * 40)
        bar = "#" * bar_len
        print(f"   {hour:02d}:00 {bar:<40} {plays:>6,}")

    # Listening by weekday
    print(f"\n[LISTENING BY WEEKDAY]")
    print("-" * 40)
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in weekday_order:
        plays = stats['weekday_plays'].get(day, 0)
        print(f"   {day:<12} {plays:>8,}")

    # Platform breakdown
    print(f"\n[PLATFORM BREAKDOWN]")
    print("-" * 60)
    for platform, count in stats['platform_plays'].most_common(10):
        pct = count / stats['total_streams'] * 100
        print(f"   {platform[:50]:<50} {count:>8,} ({pct:>5.1f}%)")

    # Country breakdown
    print(f"\n[COUNTRY BREAKDOWN]")
    print("-" * 40)
    for country, count in stats['country_plays'].most_common(10):
        pct = count / stats['total_streams'] * 100
        print(f"   {country:<10} {count:>8,} ({pct:>5.1f}%)")

    # Pommelien Thijs deep dive (special interest)
    pommelien_key = "Pommelien Thijs"
    if pommelien_key in stats['artist_plays']:
        print(f"\n*** POMMELIEN THIJS DEEP DIVE ***")
        print("-" * 60)
        plays = stats['artist_plays'][pommelien_key]
        time_ms = stats['artist_time'][pommelien_key]
        tracks = stats['artist_tracks'][pommelien_key]
        first = stats['artist_first_listen'].get(pommelien_key, "Unknown")

        print(f"   Total plays: {plays:,}")
        print(f"   Total listening time: {format_time(time_ms)}")
        print(f"   Unique tracks: {len(tracks)}")
        print(f"   First listen: {first}")
        print(f"   Tracks listened:")
        for track in sorted(tracks):
            track_key = f"{track} - {pommelien_key}"
            track_count = stats['track_plays'].get(track_key, 0)
            track_ms = stats['track_time'].get(track_key, 0)
            print(f"      - {track}: {track_count} plays ({format_time(track_ms)})")

    # Monthly listening trends
    print(f"\n[MONTHLY LISTENING HOURS - Last 24 months]")
    print("-" * 60)
    sorted_months = sorted(stats['monthly_listening_hours'].keys())[-24:]
    max_hours = max([stats['monthly_listening_hours'][m] for m in sorted_months]) if sorted_months else 1
    for month in sorted_months:
        hours = stats['monthly_listening_hours'][month]
        bar_len = int(hours / max_hours * 30)
        bar = "#" * bar_len
        print(f"   {month} {bar:<30} {hours:>6.1f} hours")

    return stats

def save_detailed_json(stats, output_path):
    """Save detailed stats to JSON for further analysis."""

    # Convert Counter objects to dicts for JSON serialization
    output = {
        'summary': {
            'total_streams': stats['total_streams'],
            'total_listening_hours': stats['total_ms'] / 3600000,
            'unique_artists': len(stats['artist_plays']),
            'unique_tracks': len(stats['track_plays']),
            'unique_albums': len(stats['album_plays']),
        },
        'top_artists_by_plays': dict(stats['artist_plays'].most_common(100)),
        'top_artists_by_time_ms': dict(stats['artist_time'].most_common(100)),
        'top_tracks_by_plays': dict(stats['track_plays'].most_common(200)),
        'top_tracks_by_time_ms': dict(stats['track_time'].most_common(200)),
        'yearly_plays': dict(sorted([(k, v) for k, v in stats['year_plays'].items() if k > 0])),
        'hourly_distribution': dict(stats['hour_plays']),
        'weekday_distribution': dict(stats['weekday_plays']),
        'platform_breakdown': dict(stats['platform_plays'].most_common(20)),
        'monthly_hours': dict(stats['monthly_listening_hours']),
        'skip_rate': stats['skip_rate'],
        'shuffle_stats': stats['shuffle_stats'],
    }

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] Detailed stats saved to: {output_path}")

def main():
    """Main analysis pipeline."""
    print("Starting Spotify listening history analysis...")
    print("="*80)

    # Load data
    streams = load_all_streaming_data()

    # Analyze
    print("\nAnalyzing streams...")
    stats = analyze_streams(streams)

    # Print report
    print_report(stats)

    # Save JSON
    output_dir = Path(r"C:\Users\Nouri\Documents\GitHub\Nouri-Mabrouk\corpus")
    output_dir.mkdir(exist_ok=True)
    save_detailed_json(stats, output_dir / "spotify_stats.json")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
