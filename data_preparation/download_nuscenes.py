"""
nuScenes Mini Dataset Download Helper

This script provides an interactive guide for downloading the nuScenes Mini dataset.
Since the dataset requires authentication, this script guides users through the process.

要件: 1.1
"""

import os
import sys
from pathlib import Path
import subprocess


def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_step(step_num: int, text: str):
    """Print a formatted step"""
    print(f"\n[Step {step_num}] {text}")
    print("-" * 70)


def check_disk_space(required_gb: float = 15.0) -> bool:
    """
    Check if there's enough disk space
    
    Args:
        required_gb: Required space in GB
        
    Returns:
        True if enough space is available
    """
    try:
        import shutil
        stat = shutil.disk_usage(Path.cwd())
        available_gb = stat.free / (1024 ** 3)
        
        print(f"Available disk space: {available_gb:.2f} GB")
        print(f"Required disk space: {required_gb:.2f} GB")
        
        if available_gb < required_gb:
            print(f"⚠️  Warning: Low disk space! You need at least {required_gb} GB")
            return False
        else:
            print("✅ Sufficient disk space available")
            return True
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True  # Proceed anyway


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"⚠️  Warning: Python {version.major}.{version.minor} detected")
        print("   Recommended: Python 3.8 or higher")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor} detected")
        return True


def install_devkit():
    """Install nuScenes devkit"""
    print("\nInstalling nuscenes-devkit...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "nuscenes-devkit"],
            check=True
        )
        print("✅ nuscenes-devkit installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install nuscenes-devkit")
        print("   Try manually: pip install nuscenes-devkit")
        return False


def main():
    print_header("nuScenes Mini Dataset Download Helper")
    
    print("This script will guide you through downloading the nuScenes Mini dataset.")
    print("The dataset is approximately 4.2 GB compressed and 10 GB extracted.")
    
    # Step 1: Check prerequisites
    print_step(1, "Checking Prerequisites")
    
    check_python_version()
    check_disk_space(required_gb=15.0)
    
    # Step 2: Create download directory
    print_step(2, "Creating Download Directory")
    
    download_dir = Path("nuscenes_mini")
    download_dir.mkdir(exist_ok=True)
    print(f"✅ Download directory created: {download_dir.absolute()}")
    
    # Step 3: Download instructions
    print_step(3, "Download Instructions")
    
    print("""
The nuScenes dataset requires registration and authentication.
Please follow these steps:

1. Visit: https://www.nuscenes.org/nuscenes
2. Click on "Download" in the navigation menu
3. Register for a free account (if you haven't already)
4. Accept the Terms of Use
5. Download the following files to the 'nuscenes_mini' directory:
   
   Required files:
   ✓ v1.0-mini.tgz (~4.2 GB) - Metadata and annotations
   
   Optional (for full sensor data):
   ✓ Mini splits - Camera images, LiDAR, radar data

6. After downloading, the files should be in:
   {download_dir.absolute()}

Press Enter when you have downloaded the files...
""")
    
    input()
    
    # Step 4: Check if files exist
    print_step(4, "Verifying Downloaded Files")
    
    mini_file = download_dir / "v1.0-mini.tgz"
    
    if mini_file.exists():
        size_mb = mini_file.stat().st_size / (1024 ** 2)
        print(f"✅ Found v1.0-mini.tgz ({size_mb:.2f} MB)")
    else:
        print(f"❌ v1.0-mini.tgz not found in {download_dir}")
        print("\nPlease download the file and place it in the directory above.")
        print("Then run this script again.")
        return
    
    # Step 5: Extract files
    print_step(5, "Extracting Dataset")
    
    print("Extracting v1.0-mini.tgz...")
    print("This may take several minutes...")
    
    try:
        import tarfile
        with tarfile.open(mini_file, 'r:gz') as tar:
            tar.extractall(path=download_dir)
        print("✅ Extraction complete!")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        print("\nYou can extract manually using:")
        print(f"  cd {download_dir}")
        print(f"  tar -xzf v1.0-mini.tgz")
        return
    
    # Step 6: Install devkit
    print_step(6, "Installing nuScenes DevKit")
    
    try:
        import nuscenes
        print("✅ nuscenes-devkit already installed")
    except ImportError:
        print("nuscenes-devkit not found. Installing...")
        install_devkit()
    
    # Step 7: Verify installation
    print_step(7, "Verifying Dataset")
    
    print("\nRunning verification script...")
    try:
        subprocess.run(
            [sys.executable, "verify_nuscenes.py", "--dataroot", str(download_dir)],
            check=True
        )
    except subprocess.CalledProcessError:
        print("⚠️  Verification script not found or failed")
        print("   You can verify manually by running:")
        print(f"   python verify_nuscenes.py --dataroot {download_dir}")
    except FileNotFoundError:
        print("⚠️  verify_nuscenes.py not found")
        print("   Skipping verification step")
    
    # Final summary
    print_header("Download Complete!")
    
    print(f"""
✅ nuScenes Mini dataset is ready!

Dataset location: {download_dir.absolute()}

Next steps:
1. Extract scenes for the search system:
   python extract_nuscenes.py --nuscenes-path {download_dir} --num-scenes 50

2. Generate embeddings:
   python generate_embeddings.py

3. Create vector database:
   python create_vector_db.py

For more information, see: NUSCENES_DOWNLOAD_GUIDE.md
""")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
