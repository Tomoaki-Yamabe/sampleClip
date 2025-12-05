# nuScenes Mini Dataset Download Guide

This guide provides step-by-step instructions for downloading and setting up the nuScenes Mini dataset for the multimodal search system.

## Overview

- **Dataset**: nuScenes Mini (v1.0)
- **Size**: ~4.2 GB (compressed), ~10 GB (extracted)
- **Scenes**: 10 scenes with full sensor data
- **Official Website**: https://www.nuscenes.org/

## Prerequisites

1. **Storage Space**: At least 15 GB of free disk space
2. **Python**: Python 3.8 or higher
3. **Internet Connection**: Stable connection for downloading large files

## Step 1: Register and Download

### Option A: Manual Download (Recommended)

1. Visit the nuScenes website: https://www.nuscenes.org/nuscenes
2. Click on "Download" in the navigation menu
3. Register for a free account if you haven't already
4. Accept the Terms of Use
5. Download the following files:
   - **v1.0-mini.tgz** (~4.2 GB) - Contains metadata and annotations
   - **v1.0-mini_meta.tgz** (~1 MB) - Contains scene metadata
   
   Optional (for full sensor data):
   - **Mini splits** - Contains camera images, LiDAR, radar data

### Option B: Using wget (Linux/Mac)

```bash
# Navigate to data directory
cd data_preparation

# Create download directory
mkdir -p nuscenes_mini
cd nuscenes_mini

# Download using wget (requires authentication token from website)
# You'll need to get the download URLs from the website after logging in
wget -O v1.0-mini.tgz "YOUR_DOWNLOAD_URL_HERE"
```

### Option C: Using Python Script

We provide a helper script that guides you through the download process:

```bash
cd data_preparation
python download_nuscenes.py
```

## Step 2: Extract the Dataset

### Extract the downloaded files:

```bash
# Navigate to download directory
cd data_preparation/nuscenes_mini

# Extract metadata
tar -xzf v1.0-mini.tgz

# The extracted structure should look like:
# nuscenes_mini/
# ├── v1.0-mini/
# │   ├── attribute.json
# │   ├── calibrated_sensor.json
# │   ├── category.json
# │   ├── ego_pose.json
# │   ├── instance.json
# │   ├── log.json
# │   ├── map.json
# │   ├── sample.json
# │   ├── sample_annotation.json
# │   ├── sample_data.json
# │   ├── scene.json
# │   ├── sensor.json
# │   └── visibility.json
# ├── samples/
# │   ├── CAM_BACK/
# │   ├── CAM_BACK_LEFT/
# │   ├── CAM_BACK_RIGHT/
# │   ├── CAM_FRONT/
# │   ├── CAM_FRONT_LEFT/
# │   ├── CAM_FRONT_RIGHT/
# │   ├── LIDAR_TOP/
# │   ├── RADAR_BACK_LEFT/
# │   ├── RADAR_BACK_RIGHT/
# │   ├── RADAR_FRONT/
# │   ├── RADAR_FRONT_LEFT/
# │   └── RADAR_FRONT_RIGHT/
# └── sweeps/
```

## Step 3: Verify the Dataset

Run the verification script to ensure the dataset is properly extracted:

```bash
cd data_preparation
python verify_nuscenes.py --dataroot nuscenes_mini
```

Expected output:
```
✅ nuScenes Mini dataset verified successfully!
   Version: v1.0-mini
   Scenes: 10
   Samples: 404
   Camera images: 2424
   Dataset size: ~10 GB
```

## Step 4: Install nuScenes DevKit

Install the official nuScenes development kit:

```bash
pip install nuscenes-devkit
```

Or if using the project's requirements file:

```bash
cd data_preparation
pip install -r requirements.txt
```

## Dataset Structure

The nuScenes Mini dataset contains:

- **10 scenes**: Each scene is ~20 seconds of driving
- **404 samples**: Keyframes at 2 Hz
- **6 cameras**: Front, Front-Left, Front-Right, Back, Back-Left, Back-Right
- **1 LiDAR**: Top-mounted 32-beam LiDAR
- **5 radars**: Front, Front-Left, Front-Right, Back-Left, Back-Right
- **Annotations**: 3D bounding boxes, tracking IDs, attributes

### Scene Locations

The Mini dataset includes scenes from:
- **Boston**: Urban and suburban driving
- **Singapore**: Dense urban environment

### Weather and Time Conditions

- Day and night scenes
- Clear and rainy weather
- Various traffic densities

## Troubleshooting

### Issue: Download fails or is interrupted

**Solution**: Use a download manager or resume the download:
```bash
wget -c -O v1.0-mini.tgz "YOUR_DOWNLOAD_URL"
```

### Issue: Extraction fails with "disk full" error

**Solution**: Ensure you have at least 15 GB of free space:
```bash
df -h  # Check available disk space
```

### Issue: Cannot import nuscenes module

**Solution**: Install the devkit:
```bash
pip install nuscenes-devkit
```

### Issue: Permission denied when extracting

**Solution**: Check file permissions:
```bash
chmod +x v1.0-mini.tgz
```

## Next Steps

After successfully downloading and extracting the dataset:

1. **Extract scenes**: Run the extraction script to process 50-100 scenes
   ```bash
   python extract_nuscenes.py --nuscenes-path nuscenes_mini --num-scenes 50
   ```

2. **Generate embeddings**: Create vector embeddings for search
   ```bash
   python generate_embeddings.py
   ```

3. **Create vector database**: Build the searchable vector database
   ```bash
   python create_vector_db.py
   ```

## Alternative: Using Sample Data

If you cannot download the full nuScenes Mini dataset, the system can work with generated sample data:

```bash
python extract_nuscenes.py --use-sample --num-scenes 10
```

This will create synthetic scenes for testing purposes.

## Resources

- **Official Documentation**: https://www.nuscenes.org/nuscenes
- **GitHub Repository**: https://github.com/nutonomy/nuscenes-devkit
- **Paper**: "nuScenes: A multimodal dataset for autonomous driving" (CVPR 2020)
- **Forum**: https://forum.nuscenes.org/

## Dataset Citation

If you use the nuScenes dataset in your work, please cite:

```bibtex
@inproceedings{caesar2020nuscenes,
  title={nuScenes: A multimodal dataset for autonomous driving},
  author={Caesar, Holger and Bankiti, Varun and Lang, Alex H and Vora, Sourabh and 
          Liong, Venice Erin and Xu, Qiang and Krishnan, Anush and Pan, Yu and 
          Baldan, Giancarlo and Beijbom, Oscar},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={11621--11631},
  year={2020}
}
```

## License

The nuScenes dataset is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (CC BY-NC-SA 4.0).

Please review the full license terms at: https://www.nuscenes.org/terms-of-use
