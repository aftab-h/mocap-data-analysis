"""Download motion capture data from various sources."""

import os
import requests
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict

# SFU Motion Capture Database
# URL pattern: http://mocap.cs.sfu.ca/nusmocap/{subject}_{motion}.bvh
SFU_BASE_URL = "http://mocap.cs.sfu.ca/nusmocap"
SFU_NEW_BASE_URL = "http://mocap.cs.sfu.ca/nusmocapnew"

SFU_MOCAP_FILES = {
    'locomotion': [
        # Walking
        ('0005_Walking001', 'Walking - Subject 0005'),
        ('0005_BackwardsWalk001', 'Backwards walk - Subject 0005'),
        ('0005_Stomping001', 'Stomping - Subject 0005'),
        ('0007_Walking001', 'Walking - Subject 0007'),
        ('0008_Walking001', 'Walking - Subject 0008'),
        ('0008_Walking002', 'Walking variation - Subject 0008'),
        ('0018_Walking001', 'Walking - Subject 0018'),
        ('0018_Catwalk001', 'Catwalk - Subject 0018'),
        ('0018_TipToe001', 'Tip toe walk - Subject 0018'),
        # Running/Jogging
        ('0005_Jogging001', 'Jogging - Subject 0005'),
        ('0005_SlowTrot001', 'Slow trot - Subject 0005'),
        ('0008_Skipping001', 'Skipping - Subject 0008'),
        # Jumping
        ('0005_2FeetJump001', 'Two feet jump - Subject 0005'),
        ('0005_JumpRope001', 'Jump rope - Subject 0005'),
        ('0005_SideSkip001', 'Side skip - Subject 0005'),
        # Rolls and Cartwheels
        ('0007_Cartwheel001', 'Cartwheel - Subject 0007'),
        ('0017_ParkourRoll001', 'Parkour roll - Subject 0017'),
        # Other
        ('0007_Balance001', 'Balance - Subject 0007'),
        ('0007_Crawling001', 'Crawling - Subject 0007'),
    ],
    'dance': [
        # Latin
        ('0008_ChaCha001', 'ChaCha - Subject 0008'),
        # Pop
        ('0018_Moonwalk001', 'Moonwalk - Subject 0018'),
        # Chinese Dance
        ('0018_DanceTurns001', 'Dance turns - Subject 0018'),
        ('0018_DanceTurns002', 'Dance turns 2 - Subject 0018'),
        ('0018_Bridge001', 'Bridge - Subject 0018'),
        ('0018_TraditionalChineseDance001', 'Traditional Chinese dance - Subject 0018'),
        ('0018_XinJiang001', 'XinJiang dance - Subject 0018'),
        ('0018_XinJiang002', 'XinJiang dance 2 - Subject 0018'),
        ('0018_XinJiang003', 'XinJiang dance 3 - Subject 0018'),
    ],
    'martial_arts': [
        # Kendo
        ('0015_BasicKendo001', 'Basic Kendo - Subject 0015'),
        ('0015_Kirikaeshi001', 'Kirikaeshi - Subject 0015'),
        ('0015_KendoKata001', 'Kendo Kata - Subject 0015'),
        # Wushu
        ('0017_WushuKicks001', 'Wushu kicks - Subject 0017'),
    ],
    'sports': [
        ('0008_Yoga001', 'Yoga - Subject 0008'),
    ],
    'obstacles': [
        # Jump and Roll
        ('0012_JumpAndRoll001', 'Jump and roll - Subject 0012'),
        ('0017_JumpAndRoll001', 'Jump and roll - Subject 0017'),
        # Jumping Over Obstacle
        ('0015_HopOverObstacle001', 'Hop over obstacle - Subject 0015'),
        ('0015_JumpOverObstacle001', 'Jump over obstacle - Subject 0015'),
        # Vaulting
        ('0012_SpeedVault001', 'Speed vault - Subject 0012'),
        ('0012_SpeedVault002', 'Speed vault 2 - Subject 0012'),
        ('0017_MonkeyVault001', 'Monkey vault - Subject 0017'),
        ('0017_SpeedVault001', 'Speed vault - Subject 0017'),
        ('0017_SpeedVault002', 'Speed vault 2 - Subject 0017'),
        # Traveling On/Off Steps
        ('0017_JumpingOnBench001', 'Jumping on bench - Subject 0017'),
        ('0017_RunningOnBench001', 'Running on bench - Subject 0017'),
        ('0017_RunningOnBench002', 'Running on bench 2 - Subject 0017'),
    ],
}

# Bollywood dance uses the new folder
SFU_NEW_MOCAP_FILES = {
    'bollywood': [
        ('0019_BasicBollywoodDance001', 'Basic Bollywood - Subject 0019'),
        ('0019_AdvanceBollywoodDance001', 'Advanced Bollywood - Subject 0019'),
    ],
}

# BVH files from reliable GitHub sources
# These are actual BVH files with walking/running/locomotion data
BVH_SOURCES = {
    'walking': [
        {
            'url': 'https://raw.githubusercontent.com/20tab/bvh-python/master/tests/test_freebvh.bvh',
            'filename': 'mixamo_walk_01.bvh',
            'description': 'Walking motion (Mixamo rig)'
        },
        {
            'url': 'https://raw.githubusercontent.com/kevinzakka/clip2mesh/main/data/motion/walk_01.bvh',
            'filename': 'walk_01.bvh',
            'description': 'Walking motion'
        },
    ],
    'running': [
        {
            'url': 'https://raw.githubusercontent.com/kevinzakka/clip2mesh/main/data/motion/run_01.bvh',
            'filename': 'run_01.bvh',
            'description': 'Running motion'
        },
    ],
    'jumping': [
        {
            'url': 'https://raw.githubusercontent.com/kevinzakka/clip2mesh/main/data/motion/jump_01.bvh',
            'filename': 'jump_01.bvh',
            'description': 'Jumping motion'
        },
    ],
    'general': [],
}

# Backup: LaFAN1 dataset samples (if available)
LAFAN1_BASE = "https://raw.githubusercontent.com/ubisoft/ubisoft-laforge-animation-dataset/master/lafan1/lafan1"

# Alternative: Sample BVH files from motion matching projects
MOTION_MATCHING_SAMPLES = [
    {
        'url': 'https://raw.githubusercontent.com/orangeduck/Motion-Matching/main/resources/database.bin',
        'filename': 'motion_matching_db.bin',
        'description': 'Motion matching database'
    }
]


def get_bvh_url(subject: str, trial: str) -> str:
    """Construct URL for a CMU BVH file (legacy - CMU uses AMC not BVH)."""
    return f"http://mocap.cs.cmu.edu/subjects/{subject}/{subject}_{trial}.bvh"


def download_file(url: str, output_path: Path, verify_ssl: bool = False) -> bool:
    """
    Download a file from URL to output path.

    Args:
        url: Source URL
        output_path: Destination path
        verify_ssl: Whether to verify SSL certificate

    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, verify=verify_ssl, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                with tqdm(total=total_size, unit='iB', unit_scale=True,
                         desc=output_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return False


def download_trials(categories: List[str] = None, output_dir: str = "data/raw") -> List[Path]:
    """
    Download motion capture BVH files from various sources.

    Args:
        categories: List of categories to download ('walking', 'running', 'jumping', 'general')
                   If None, downloads all categories
        output_dir: Directory to save files

    Returns:
        List of paths to downloaded files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if categories is None:
        categories = list(BVH_SOURCES.keys())

    downloaded_files = []

    for category in categories:
        if category not in BVH_SOURCES:
            print(f"Unknown category: {category}")
            continue

        sources = BVH_SOURCES[category]
        if not sources:
            print(f"\nNo sources available for {category}")
            continue

        print(f"\nDownloading {category} files...")

        for source in sources:
            filename = source['filename']
            file_path = output_path / filename

            if file_path.exists():
                print(f"  {filename} already exists, skipping")
                downloaded_files.append(file_path)
                continue

            url = source['url']
            print(f"  Downloading {filename} ({source['description']})...")

            if download_file(url, file_path):
                downloaded_files.append(file_path)
                print(f"  Success: {filename}")
            else:
                print(f"  Failed to download {filename}")

    return downloaded_files


def download_sfu_mocap(
    categories: List[str] = None,
    output_dir: str = "data/sfu",
    file_format: str = "bvh"
) -> List[Path]:
    """
    Download motion capture files from SFU Motion Capture Database.

    Args:
        categories: List of categories to download. Options:
                   'locomotion', 'dance', 'martial_arts', 'sports', 'obstacles', 'bollywood'
                   If None, downloads all categories
        output_dir: Directory to save files
        file_format: File format to download ('bvh', 'fbx', 'c3d', 'txt')

    Returns:
        List of paths to downloaded files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if categories is None:
        categories = list(SFU_MOCAP_FILES.keys()) + list(SFU_NEW_MOCAP_FILES.keys())

    downloaded_files = []
    failed_files = []

    for category in categories:
        # Determine which file list and base URL to use
        if category in SFU_MOCAP_FILES:
            files = SFU_MOCAP_FILES[category]
            base_url = SFU_BASE_URL
        elif category in SFU_NEW_MOCAP_FILES:
            files = SFU_NEW_MOCAP_FILES[category]
            base_url = SFU_NEW_BASE_URL
        else:
            print(f"Unknown SFU category: {category}")
            continue

        print(f"\nDownloading SFU {category} files...")

        for motion_name, description in files:
            filename = f"{motion_name}.{file_format}"
            file_path = output_path / filename

            if file_path.exists():
                print(f"  {filename} already exists, skipping")
                downloaded_files.append(file_path)
                continue

            url = f"{base_url}/{filename}"
            print(f"  Downloading {filename} ({description})...")

            if download_file(url, file_path, verify_ssl=False):
                # Verify it's not an error page
                if file_path.stat().st_size > 1000:  # BVH files should be > 1KB
                    downloaded_files.append(file_path)
                    print(f"  Success: {filename}")
                else:
                    print(f"  Warning: {filename} seems too small, may be invalid")
                    failed_files.append(filename)
                    file_path.unlink()  # Remove invalid file
            else:
                failed_files.append(filename)
                print(f"  Failed to download {filename}")

    if failed_files:
        print(f"\n{len(failed_files)} files failed to download")

    return downloaded_files


def get_trial_info(filename: str) -> dict:
    """
    Get metadata about a trial from its filename.

    Args:
        filename: BVH filename (e.g., "walk_01.bvh" or "mixamo_walk_01.bvh")

    Returns:
        Dictionary with trial metadata
    """
    name = Path(filename).stem.lower()

    # Determine activity type based on filename
    if 'walk' in name:
        activity = 'walking'
    elif 'run' in name:
        activity = 'running'
    elif 'jump' in name:
        activity = 'jumping'
    else:
        activity = 'general'

    # Extract subject/trial info if present
    parts = name.split('_')
    subject = parts[0] if len(parts) > 1 else 'unknown'
    trial = parts[-1] if len(parts) > 1 else '01'

    return {
        'subject': subject,
        'trial': trial,
        'activity': activity,
        'filename': filename
    }


def list_available_files(data_dir: str = "data/raw") -> List[dict]:
    """
    List all available BVH files in the data directory.

    Args:
        data_dir: Directory to search

    Returns:
        List of trial info dictionaries
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    files = []
    for bvh_file in data_path.glob("*.bvh"):
        info = get_trial_info(bvh_file.name)
        info['path'] = str(bvh_file)
        info['size_kb'] = bvh_file.stat().st_size / 1024
        files.append(info)

    return sorted(files, key=lambda x: (x['activity'], x['subject'], x['trial']))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download motion capture data from various sources")
    parser.add_argument('--source', choices=['github', 'sfu', 'all'], default='all',
                       help='Data source to download from (default: all)')
    parser.add_argument('--categories', nargs='+',
                       help='Categories to download (varies by source)')
    parser.add_argument('--output', default='data/raw',
                       help='Output directory for GitHub sources')
    parser.add_argument('--sfu-output', default='data/sfu',
                       help='Output directory for SFU mocap data')
    parser.add_argument('--format', choices=['bvh', 'fbx', 'c3d', 'txt'], default='bvh',
                       help='File format for SFU downloads (default: bvh)')
    parser.add_argument('--list', action='store_true',
                       help='List available files instead of downloading')

    args = parser.parse_args()

    if args.list:
        # List files from both sources
        print("\n=== GitHub/Raw BVH Files ===")
        files = list_available_files(args.output)
        if files:
            print(f"Available files in {args.output}:")
            for f in files:
                print(f"  {f['filename']} - {f['activity']} (Subject {f['subject']}, Trial {f['trial']}) - {f['size_kb']:.1f} KB")
        else:
            print(f"No BVH files found in {args.output}")

        print("\n=== SFU Mocap Files ===")
        sfu_files = list_available_files(args.sfu_output)
        if sfu_files:
            print(f"Available files in {args.sfu_output}:")
            for f in sfu_files:
                print(f"  {f['filename']} - {f['activity']} (Subject {f['subject']}, Trial {f['trial']}) - {f['size_kb']:.1f} KB")
        else:
            print(f"No BVH files found in {args.sfu_output}")

        print("\n=== Available SFU Categories ===")
        print("  Main: " + ", ".join(SFU_MOCAP_FILES.keys()))
        print("  New: " + ", ".join(SFU_NEW_MOCAP_FILES.keys()))
    else:
        total_downloaded = 0

        if args.source in ['github', 'all']:
            github_categories = args.categories if args.categories else None
            # Filter to valid GitHub categories
            if github_categories:
                github_categories = [c for c in github_categories if c in BVH_SOURCES]
            downloaded = download_trials(github_categories, args.output)
            print(f"\nDownloaded {len(downloaded)} files from GitHub sources to {args.output}")
            total_downloaded += len(downloaded)

        if args.source in ['sfu', 'all']:
            sfu_categories = args.categories if args.categories else None
            # Filter to valid SFU categories
            valid_sfu = list(SFU_MOCAP_FILES.keys()) + list(SFU_NEW_MOCAP_FILES.keys())
            if sfu_categories:
                sfu_categories = [c for c in sfu_categories if c in valid_sfu]
            downloaded = download_sfu_mocap(sfu_categories, args.sfu_output, args.format)
            print(f"\nDownloaded {len(downloaded)} files from SFU to {args.sfu_output}")
            total_downloaded += len(downloaded)

        print(f"\n=== Total: {total_downloaded} files downloaded ===")
