"""
統合テスト実行スクリプト

このスクリプトは、データ準備からDocker起動、パフォーマンステストまでの
全プロセスを自動化します。

要件: 2.3, 3.3
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
import json


def run_command(cmd: list, cwd: str = None, check: bool = True) -> bool:
    """
    コマンドを実行
    
    Args:
        cmd: コマンドリスト
        cwd: 作業ディレクトリ
        check: エラー時に例外を発生させるか
        
    Returns:
        成功した場合True
    """
    try:
        print(f"\n▶ Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=check,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            print(result.stdout)
        
        return result.returncode == 0
    
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with exit code {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def check_prerequisites() -> bool:
    """
    前提条件をチェック
    
    Returns:
        すべての前提条件が満たされている場合True
    """
    print("\n" + "=" * 70)
    print("Checking Prerequisites")
    print("=" * 70)
    
    all_ok = True
    
    # Python version
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}")
    else:
        print(f"✗ Python {version.major}.{version.minor} (requires 3.8+)")
        all_ok = False
    
    # Docker
    if run_command(["docker", "--version"], check=False):
        print("✓ Docker installed")
    else:
        print("✗ Docker not found")
        all_ok = False
    
    # Docker Compose
    if run_command(["docker-compose", "--version"], check=False):
        print("✓ Docker Compose installed")
    else:
        print("✗ Docker Compose not found")
        all_ok = False
    
    return all_ok


def extract_scenes(num_scenes: int, use_sample: bool = False) -> bool:
    """
    シーンを抽出
    
    Args:
        num_scenes: 抽出するシーン数
        use_sample: サンプルデータを使用するか
        
    Returns:
        成功した場合True
    """
    print("\n" + "=" * 70)
    print(f"Step 1: Extracting {num_scenes} Scenes")
    print("=" * 70)
    
    cmd = [
        sys.executable,
        "extract_nuscenes.py",
        "--num-scenes", str(num_scenes)
    ]
    
    if use_sample:
        cmd.append("--use-sample")
    
    return run_command(cmd)


def generate_embeddings() -> bool:
    """
    埋め込みを生成
    
    Returns:
        成功した場合True
    """
    print("\n" + "=" * 70)
    print("Step 2: Generating Embeddings")
    print("=" * 70)
    
    cmd = [
        sys.executable,
        "generate_embeddings.py",
        "--batch-size", "8",
        "--save-interval", "10"
    ]
    
    return run_command(cmd)


def create_vector_db() -> bool:
    """
    ベクトルDBを作成
    
    Returns:
        成功した場合True
    """
    print("\n" + "=" * 70)
    print("Step 3: Creating Vector Database")
    print("=" * 70)
    
    cmd = [sys.executable, "create_vector_db.py"]
    return run_command(cmd)


def generate_umap() -> bool:
    """
    UMAP座標を生成
    
    Returns:
        成功した場合True
    """
    print("\n" + "=" * 70)
    print("Step 4: Generating UMAP Coordinates")
    print("=" * 70)
    
    cmd = [sys.executable, "generate_umap.py"]
    return run_command(cmd)


def copy_data_to_backend() -> bool:
    """
    データをバックエンドにコピー
    
    Returns:
        成功した場合True
    """
    print("\n" + "=" * 70)
    print("Step 5: Copying Data to Backend")
    print("=" * 70)
    
    import shutil
    
    try:
        # ソースパス
        vector_db = Path("extracted_data/vector_db.json")
        umap_data = Path("extracted_data/scenes_with_umap.json")
        images_dir = Path("extracted_data/images")
        
        # デスティネーションパス
        backend_model = Path("../integ-app/backend/app/model")
        backend_static = Path("../integ-app/backend/app/static/scenes")
        
        # コピー
        if vector_db.exists():
            shutil.copy(vector_db, backend_model / "vector_db.json")
            print(f"✓ Copied {vector_db} to {backend_model}")
        else:
            print(f"✗ {vector_db} not found")
            return False
        
        if umap_data.exists():
            shutil.copy(umap_data, backend_model / "scenes_with_umap.json")
            print(f"✓ Copied {umap_data} to {backend_model}")
        else:
            print(f"⚠ {umap_data} not found (optional)")
        
        if images_dir.exists():
            backend_static.mkdir(parents=True, exist_ok=True)
            shutil.copytree(images_dir, backend_static, dirs_exist_ok=True)
            print(f"✓ Copied {images_dir} to {backend_static}")
        else:
            print(f"✗ {images_dir} not found")
            return False
        
        return True
    
    except Exception as e:
        print(f"✗ Error copying data: {e}")
        return False


def start_docker() -> bool:
    """
    Dockerコンテナを起動
    
    Returns:
        成功した場合True
    """
    print("\n" + "=" * 70)
    print("Step 6: Starting Docker Containers")
    print("=" * 70)
    
    # docker-compose up -d
    cmd = ["docker-compose", "up", "-d", "--build"]
    
    if not run_command(cmd, cwd="../integ-app"):
        return False
    
    # コンテナの起動を待つ
    print("\nWaiting for containers to start...")
    time.sleep(10)
    
    # ヘルスチェック
    import requests
    
    for i in range(30):  # 最大30秒待つ
        try:
            response = requests.get("http://localhost:8000/", timeout=2)
            if response.status_code == 200:
                print("✓ Backend is ready")
                return True
        except Exception:
            pass
        
        time.sleep(1)
        print(".", end="", flush=True)
    
    print("\n✗ Backend did not start in time")
    return False


def run_performance_tests() -> bool:
    """
    パフォーマンステストを実行
    
    Returns:
        成功した場合True
    """
    print("\n" + "=" * 70)
    print("Step 7: Running Performance Tests")
    print("=" * 70)
    
    cmd = [
        sys.executable,
        "test_performance.py",
        "--iterations", "10",
        "--workers", "5"
    ]
    
    return run_command(cmd)


def stop_docker():
    """Dockerコンテナを停止"""
    print("\n" + "=" * 70)
    print("Stopping Docker Containers")
    print("=" * 70)
    
    cmd = ["docker-compose", "down"]
    run_command(cmd, cwd="../integ-app", check=False)


def main():
    parser = argparse.ArgumentParser(
        description='Run complete integration test pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script automates the entire integration test process:
1. Extract scenes from nuScenes dataset
2. Generate embeddings
3. Create vector database
4. Generate UMAP coordinates
5. Copy data to backend
6. Start Docker containers
7. Run performance tests

Examples:
  # Run with 50 sample scenes
  python run_integration_test.py --num-scenes 50 --use-sample
  
  # Run with real nuScenes data
  python run_integration_test.py --num-scenes 50
  
  # Skip data preparation (use existing data)
  python run_integration_test.py --skip-data-prep
        """
    )
    parser.add_argument(
        '--num-scenes',
        type=int,
        default=50,
        help='Number of scenes to extract (default: 50)'
    )
    parser.add_argument(
        '--use-sample',
        action='store_true',
        help='Use sample data instead of real nuScenes dataset'
    )
    parser.add_argument(
        '--skip-data-prep',
        action='store_true',
        help='Skip data preparation steps (use existing data)'
    )
    parser.add_argument(
        '--skip-docker',
        action='store_true',
        help='Skip Docker startup (assume already running)'
    )
    parser.add_argument(
        '--keep-running',
        action='store_true',
        help='Keep Docker containers running after tests'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Integration Test Pipeline")
    print("=" * 70)
    print(f"Scenes: {args.num_scenes}")
    print(f"Use sample data: {args.use_sample}")
    print(f"Skip data prep: {args.skip_data_prep}")
    print("=" * 70)
    
    # 前提条件チェック
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please install missing dependencies.")
        sys.exit(1)
    
    try:
        # データ準備
        if not args.skip_data_prep:
            if not extract_scenes(args.num_scenes, args.use_sample):
                print("\n❌ Scene extraction failed")
                sys.exit(1)
            
            if not generate_embeddings():
                print("\n❌ Embedding generation failed")
                sys.exit(1)
            
            if not create_vector_db():
                print("\n❌ Vector database creation failed")
                sys.exit(1)
            
            if not generate_umap():
                print("\n❌ UMAP generation failed")
                sys.exit(1)
            
            if not copy_data_to_backend():
                print("\n❌ Data copy failed")
                sys.exit(1)
        else:
            print("\n⏭  Skipping data preparation")
        
        # Docker起動
        if not args.skip_docker:
            if not start_docker():
                print("\n❌ Docker startup failed")
                sys.exit(1)
        else:
            print("\n⏭  Skipping Docker startup")
        
        # パフォーマンステスト
        if not run_performance_tests():
            print("\n⚠️  Performance tests completed with errors")
        
        # 成功
        print("\n" + "=" * 70)
        print("✅ Integration Test Pipeline Complete!")
        print("=" * 70)
        print("\nNext steps:")
        print("  - Review performance results: performance_results.json")
        print("  - Access frontend: http://localhost:3000")
        print("  - Access backend: http://localhost:8000")
        
        if not args.keep_running:
            print("\nStopping Docker containers...")
            stop_docker()
        else:
            print("\nDocker containers are still running.")
            print("To stop them: cd ../integ-app && docker-compose down")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        if not args.keep_running:
            stop_docker()
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        if not args.keep_running:
            stop_docker()
        sys.exit(1)


if __name__ == "__main__":
    main()
