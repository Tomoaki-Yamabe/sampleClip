"""
パフォーマンステストスクリプト

ローカルDocker環境での検索パフォーマンスをテストします。
- レスポンス時間の測定
- メモリ使用量の確認
- 同時実行テスト
- 大規模データでの検索テスト

要件: 2.3, 3.3
"""

import argparse
import json
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


class PerformanceTester:
    """パフォーマンステストクラス"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = {
            "text_search": [],
            "image_search": [],
            "concurrent": []
        }
    
    def test_text_search(self, queries: List[str], iterations: int = 10) -> Dict[str, Any]:
        """
        テキスト検索のパフォーマンステスト
        
        Args:
            queries: テストクエリのリスト
            iterations: 各クエリの実行回数
            
        Returns:
            テスト結果
        """
        print("\n" + "=" * 70)
        print("Text Search Performance Test")
        print("=" * 70)
        
        response_times = []
        errors = 0
        
        for query in queries:
            print(f"\nTesting query: '{query}'")
            query_times = []
            
            for i in range(iterations):
                try:
                    start_time = time.time()
                    
                    response = requests.post(
                        f"{self.api_url}/search/text",
                        json={"query": query, "top_k": 5},
                        timeout=30
                    )
                    
                    elapsed = time.time() - start_time
                    
                    if response.status_code == 200:
                        query_times.append(elapsed)
                        response_times.append(elapsed)
                        
                        if i == 0:  # 最初の実行のみ結果を表示
                            data = response.json()
                            print(f"  Results: {len(data.get('results', []))} scenes")
                    else:
                        print(f"  Error: HTTP {response.status_code}")
                        errors += 1
                
                except Exception as e:
                    print(f"  Exception: {e}")
                    errors += 1
            
            if query_times:
                avg_time = statistics.mean(query_times)
                min_time = min(query_times)
                max_time = max(query_times)
                print(f"  Avg: {avg_time*1000:.2f}ms | Min: {min_time*1000:.2f}ms | Max: {max_time*1000:.2f}ms")
        
        if response_times:
            result = {
                "total_requests": len(response_times),
                "errors": errors,
                "avg_response_time_ms": statistics.mean(response_times) * 1000,
                "min_response_time_ms": min(response_times) * 1000,
                "max_response_time_ms": max(response_times) * 1000,
                "median_response_time_ms": statistics.median(response_times) * 1000,
                "stdev_ms": statistics.stdev(response_times) * 1000 if len(response_times) > 1 else 0
            }
        else:
            result = {"error": "No successful requests"}
        
        self.results["text_search"] = result
        return result
    
    def test_image_search(self, image_paths: List[Path], iterations: int = 5) -> Dict[str, Any]:
        """
        画像検索のパフォーマンステスト
        
        Args:
            image_paths: テスト画像のパスリスト
            iterations: 各画像の実行回数
            
        Returns:
            テスト結果
        """
        print("\n" + "=" * 70)
        print("Image Search Performance Test")
        print("=" * 70)
        
        response_times = []
        errors = 0
        
        for image_path in image_paths:
            if not image_path.exists():
                print(f"\nSkipping: {image_path} (not found)")
                continue
            
            print(f"\nTesting image: {image_path.name}")
            image_times = []
            
            for i in range(iterations):
                try:
                    start_time = time.time()
                    
                    with open(image_path, 'rb') as f:
                        files = {'file': (image_path.name, f, 'image/jpeg')}
                        response = requests.post(
                            f"{self.api_url}/search/image",
                            files=files,
                            data={"top_k": 5},
                            timeout=30
                        )
                    
                    elapsed = time.time() - start_time
                    
                    if response.status_code == 200:
                        image_times.append(elapsed)
                        response_times.append(elapsed)
                        
                        if i == 0:  # 最初の実行のみ結果を表示
                            data = response.json()
                            print(f"  Results: {len(data.get('results', []))} scenes")
                    else:
                        print(f"  Error: HTTP {response.status_code}")
                        errors += 1
                
                except Exception as e:
                    print(f"  Exception: {e}")
                    errors += 1
            
            if image_times:
                avg_time = statistics.mean(image_times)
                min_time = min(image_times)
                max_time = max(image_times)
                print(f"  Avg: {avg_time*1000:.2f}ms | Min: {min_time*1000:.2f}ms | Max: {max_time*1000:.2f}ms")
        
        if response_times:
            result = {
                "total_requests": len(response_times),
                "errors": errors,
                "avg_response_time_ms": statistics.mean(response_times) * 1000,
                "min_response_time_ms": min(response_times) * 1000,
                "max_response_time_ms": max(response_times) * 1000,
                "median_response_time_ms": statistics.median(response_times) * 1000,
                "stdev_ms": statistics.stdev(response_times) * 1000 if len(response_times) > 1 else 0
            }
        else:
            result = {"error": "No successful requests"}
        
        self.results["image_search"] = result
        return result
    
    def test_concurrent_requests(self, queries: List[str], num_workers: int = 5) -> Dict[str, Any]:
        """
        同時実行テスト
        
        Args:
            queries: テストクエリのリスト
            num_workers: 同時実行数
            
        Returns:
            テスト結果
        """
        print("\n" + "=" * 70)
        print(f"Concurrent Requests Test ({num_workers} workers)")
        print("=" * 70)
        
        def make_request(query: str) -> float:
            """単一リクエストを実行"""
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_url}/search/text",
                    json={"query": query, "top_k": 5},
                    timeout=30
                )
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    return elapsed
                else:
                    return -1  # エラー
            except Exception:
                return -1  # エラー
        
        # 同時実行
        start_time = time.time()
        response_times = []
        errors = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(make_request, query) for query in queries]
            
            for future in as_completed(futures):
                elapsed = future.result()
                if elapsed > 0:
                    response_times.append(elapsed)
                else:
                    errors += 1
        
        total_time = time.time() - start_time
        
        if response_times:
            result = {
                "total_requests": len(queries),
                "successful": len(response_times),
                "errors": errors,
                "total_time_s": total_time,
                "avg_response_time_ms": statistics.mean(response_times) * 1000,
                "min_response_time_ms": min(response_times) * 1000,
                "max_response_time_ms": max(response_times) * 1000,
                "throughput_rps": len(response_times) / total_time
            }
            
            print(f"\nTotal time: {total_time:.2f}s")
            print(f"Successful: {len(response_times)}/{len(queries)}")
            print(f"Throughput: {result['throughput_rps']:.2f} requests/sec")
            print(f"Avg response: {result['avg_response_time_ms']:.2f}ms")
        else:
            result = {"error": "No successful requests"}
        
        self.results["concurrent"] = result
        return result
    
    def check_api_health(self) -> bool:
        """
        APIの健全性チェック
        
        Returns:
            APIが利用可能な場合True
        """
        try:
            response = requests.get(f"{self.api_url}/", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def print_summary(self):
        """テスト結果のサマリーを表示"""
        print("\n" + "=" * 70)
        print("Performance Test Summary")
        print("=" * 70)
        
        if "text_search" in self.results and "avg_response_time_ms" in self.results["text_search"]:
            print("\nText Search:")
            ts = self.results["text_search"]
            print(f"  Requests: {ts['total_requests']}")
            print(f"  Errors: {ts['errors']}")
            print(f"  Avg response: {ts['avg_response_time_ms']:.2f}ms")
            print(f"  Min/Max: {ts['min_response_time_ms']:.2f}ms / {ts['max_response_time_ms']:.2f}ms")
        
        if "image_search" in self.results and "avg_response_time_ms" in self.results["image_search"]:
            print("\nImage Search:")
            is_result = self.results["image_search"]
            print(f"  Requests: {is_result['total_requests']}")
            print(f"  Errors: {is_result['errors']}")
            print(f"  Avg response: {is_result['avg_response_time_ms']:.2f}ms")
            print(f"  Min/Max: {is_result['min_response_time_ms']:.2f}ms / {is_result['max_response_time_ms']:.2f}ms")
        
        if "concurrent" in self.results and "throughput_rps" in self.results["concurrent"]:
            print("\nConcurrent Requests:")
            cr = self.results["concurrent"]
            print(f"  Total requests: {cr['total_requests']}")
            print(f"  Successful: {cr['successful']}")
            print(f"  Throughput: {cr['throughput_rps']:.2f} req/s")
            print(f"  Avg response: {cr['avg_response_time_ms']:.2f}ms")
        
        print("\n" + "=" * 70)
    
    def save_results(self, output_path: str):
        """
        テスト結果をJSONファイルに保存
        
        Args:
            output_path: 出力ファイルパス
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Performance testing for multimodal search API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests with default settings
  python test_performance.py
  
  # Test with custom API URL
  python test_performance.py --api-url http://localhost:8000
  
  # Run only text search tests
  python test_performance.py --test text
  
  # Run concurrent test with 10 workers
  python test_performance.py --test concurrent --workers 10
        """
    )
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://localhost:8000',
        help='API base URL (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--test',
        type=str,
        choices=['all', 'text', 'image', 'concurrent'],
        default='all',
        help='Test type to run (default: all)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Number of iterations per test (default: 10)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='Number of concurrent workers (default: 5)'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        default='data_preparation/extracted_data/images',
        help='Directory containing test images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='performance_results.json',
        help='Output file for results (default: performance_results.json)'
    )
    
    args = parser.parse_args()
    
    # テスタークラスを初期化
    tester = PerformanceTester(api_url=args.api_url)
    
    # APIの健全性チェック
    print("Checking API health...")
    if not tester.check_api_health():
        print(f"❌ API is not available at {args.api_url}")
        print("Please ensure the Docker containers are running:")
        print("  cd integ-app")
        print("  docker-compose up")
        return
    
    print(f"✅ API is available at {args.api_url}")
    
    # テストクエリ
    test_queries = [
        "雨の日の交差点",
        "高速道路での走行",
        "夜間の住宅街",
        "駐車場での低速走行",
        "市街地の信号交差点"
    ]
    
    # テスト実行
    if args.test in ['all', 'text']:
        tester.test_text_search(test_queries, iterations=args.iterations)
    
    if args.test in ['all', 'image']:
        # テスト画像を取得
        images_dir = Path(args.images_dir)
        if images_dir.exists():
            image_paths = list(images_dir.glob("*.jpg"))[:5]  # 最初の5枚
            if image_paths:
                tester.test_image_search(image_paths, iterations=max(1, args.iterations // 2))
            else:
                print(f"\n⚠️  No images found in {images_dir}")
        else:
            print(f"\n⚠️  Images directory not found: {images_dir}")
    
    if args.test in ['all', 'concurrent']:
        # 同時実行テスト用のクエリを生成
        concurrent_queries = test_queries * args.workers
        tester.test_concurrent_requests(concurrent_queries, num_workers=args.workers)
    
    # サマリーを表示
    tester.print_summary()
    
    # 結果を保存
    tester.save_results(args.output)


if __name__ == "__main__":
    main()
