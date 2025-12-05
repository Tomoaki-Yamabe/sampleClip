"""
nuScenes Dataset Verification Script

This script verifies that the nuScenes dataset is properly downloaded and extracted.

要件: 1.1
"""

import argparse
import sys
from pathlib import Path
import json


def verify_dataset(dataroot: str) -> bool:
    """
    Verify nuScenes dataset structure and contents
    
    Args:
        dataroot: Path to nuScenes dataset root
        
    Returns:
        True if dataset is valid
    """
    dataroot_path = Path(dataroot)
    
    print(f"Verifying nuScenes dataset at: {dataroot_path.absolute()}")
    print("-" * 70)
    
    # Check if directory exists
    if not dataroot_path.exists():
        print(f"❌ Directory not found: {dataroot_path}")
        return False
    
    print(f"✅ Dataset directory exists")
    
    # Check for v1.0-mini directory
    version_dir = dataroot_path / "v1.0-mini"
    if not version_dir.exists():
        print(f"❌ Version directory not found: {version_dir}")
        print("   Expected: v1.0-mini/")
        return False
    
    print(f"✅ Version directory found: v1.0-mini/")
    
    # Check for required JSON files
    required_files = [
        "scene.json",
        "sample.json",
        "sample_data.json",
        "sensor.json",
        "calibrated_sensor.json",
        "ego_pose.json",
        "log.json",
        "category.json",
        "attribute.json",
        "visibility.json",
        "instance.json",
        "sample_annotation.json",
        "map.json"
    ]
    
    missing_files = []
    for filename in required_files:
        filepath = version_dir / filename
        if not filepath.exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"❌ Missing required files:")
        for filename in missing_files:
            print(f"   - {filename}")
        return False
    
    print(f"✅ All required metadata files found ({len(required_files)} files)")
    
    # Load and verify scene data
    try:
        with open(version_dir / "scene.json", 'r') as f:
            scenes = json.load(f)
        
        num_scenes = len(scenes)
        print(f"✅ Scenes: {num_scenes}")
        
        if num_scenes < 10:
            print(f"⚠️  Warning: Expected 10 scenes, found {num_scenes}")
    except Exception as e:
        print(f"❌ Error reading scene.json: {e}")
        return False
    
    # Load and verify sample data
    try:
        with open(version_dir / "sample.json", 'r') as f:
            samples = json.load(f)
        
        num_samples = len(samples)
        print(f"✅ Samples: {num_samples}")
        
        if num_samples < 400:
            print(f"⚠️  Warning: Expected ~404 samples, found {num_samples}")
    except Exception as e:
        print(f"❌ Error reading sample.json: {e}")
        return False
    
    # Check for samples directory (camera images)
    samples_dir = dataroot_path / "samples"
    if samples_dir.exists():
        print(f"✅ Samples directory found")
        
        # Check camera directories
        camera_dirs = [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT"
        ]
        
        found_cameras = []
        for cam_dir in camera_dirs:
            cam_path = samples_dir / cam_dir
            if cam_path.exists():
                num_images = len(list(cam_path.glob("*.jpg")))
                found_cameras.append((cam_dir, num_images))
        
        if found_cameras:
            print(f"✅ Camera images found:")
            total_images = 0
            for cam_name, num_images in found_cameras:
                print(f"   - {cam_name}: {num_images} images")
                total_images += num_images
            print(f"   Total: {total_images} images")
        else:
            print(f"⚠️  No camera images found in samples directory")
            print(f"   This is OK if you only downloaded metadata")
    else:
        print(f"⚠️  Samples directory not found: {samples_dir}")
        print(f"   This is OK if you only downloaded metadata")
    
    # Check for sweeps directory (sensor sweeps)
    sweeps_dir = dataroot_path / "sweeps"
    if sweeps_dir.exists():
        print(f"✅ Sweeps directory found")
    else:
        print(f"⚠️  Sweeps directory not found")
        print(f"   This is OK if you only downloaded metadata")
    
    # Try to load with nuScenes devkit
    print("\nTesting nuScenes devkit...")
    try:
        from nuscenes.nuscenes import NuScenes
        
        nusc = NuScenes(version='v1.0-mini', dataroot=str(dataroot_path), verbose=False)
        
        print(f"✅ nuScenes devkit loaded successfully")
        print(f"   Version: {nusc.version}")
        print(f"   Scenes: {len(nusc.scene)}")
        print(f"   Samples: {len(nusc.sample)}")
        
        # Show sample scene info
        if len(nusc.scene) > 0:
            sample_scene = nusc.scene[0]
            print(f"\nSample scene info:")
            print(f"   Name: {sample_scene['name']}")
            print(f"   Description: {sample_scene['description']}")
            print(f"   Samples: {sample_scene['nbr_samples']}")
        
    except ImportError:
        print(f"⚠️  nuScenes devkit not installed")
        print(f"   Install with: pip install nuscenes-devkit")
    except Exception as e:
        print(f"❌ Error loading with nuScenes devkit: {e}")
        return False
    
    # Calculate dataset size
    try:
        total_size = sum(f.stat().st_size for f in dataroot_path.rglob('*') if f.is_file())
        size_gb = total_size / (1024 ** 3)
        print(f"\n✅ Dataset size: {size_gb:.2f} GB")
    except Exception as e:
        print(f"⚠️  Could not calculate dataset size: {e}")
    
    print("\n" + "=" * 70)
    print("✅ nuScenes Mini dataset verified successfully!")
    print("=" * 70)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Verify nuScenes dataset')
    parser.add_argument(
        '--dataroot',
        type=str,
        required=True,
        help='Path to nuScenes dataset root directory'
    )
    
    args = parser.parse_args()
    
    success = verify_dataset(args.dataroot)
    
    if not success:
        print("\n❌ Dataset verification failed")
        print("Please check the error messages above and ensure:")
        print("1. The dataset is fully downloaded")
        print("2. The dataset is properly extracted")
        print("3. The path points to the correct directory")
        sys.exit(1)
    else:
        print("\nYou can now proceed with scene extraction:")
        print(f"  python extract_nuscenes.py --nuscenes-path {args.dataroot} --num-scenes 50")
        sys.exit(0)


if __name__ == "__main__":
    main()
