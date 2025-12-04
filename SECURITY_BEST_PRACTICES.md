# セキュリティベストプラクティス

## ✅ 完了した対策

### 1. Git履歴のクリーンアップ
- ✅ ローカルのGit履歴から機密情報を削除
- ✅ GitHubへ強制プッシュして履歴を上書き
- ✅ 機密情報が含まれていないことを確認

### 2. 安全なDockerfile
- ✅ 環境変数ベースのproxy設定に変更
- ✅ ハードコードされた認証情報を削除

## 🔒 今後の予防策

### 1. 環境変数の使用

**悪い例：**
```dockerfile
RUN pip install --proxy http://user:password@10.121.48.30:8080
```

**良い例：**
```dockerfile
ARG HTTP_PROXY
ENV HTTP_PROXY=${HTTP_PROXY}
RUN pip install --no-cache-dir -r requirements.txt
```

**使用方法：**
```bash
# ビルド時に環境変数を渡す
docker build \
  --build-arg HTTP_PROXY=http://user:password@proxy:8080 \
  -t my-image .
```

### 2. .gitignoreの活用

以下のファイルは必ずgitignoreに追加：
```
.env
.env.local
*.pem
*.key
*credentials*
*secrets*
```

### 3. .envファイルの管理

```bash
# .env.exampleをテンプレートとして提供
cp .env.example .env

# .envに実際の値を記入（このファイルはGitにコミットしない）
```

### 4. コミット前のチェック

Git Hooksを使って自動チェック：

```bash
# .git/hooks/pre-commit を作成
#!/bin/sh
if git diff --cached | grep -E 'password|secret|token|key|proxy.*@'; then
    echo "⚠️ 機密情報が含まれている可能性があります"
    exit 1
fi
```

### 5. 定期的なスキャン

```bash
# 機密情報のスキャン
git grep -n 'password\|secret\|token' $(git rev-list --all)

# または、専用ツールを使用
# - git-secrets: https://github.com/awslabs/git-secrets
# - truffleHog: https://github.com/trufflesecurity/trufflehog
```

## 📋 チェックリスト（今後のプロジェクト用）

プロジェクト開始時：
- [ ] .gitignoreを設定
- [ ] .env.exampleを作成
- [ ] Git Hooksを設定
- [ ] チームにセキュリティガイドラインを共有

コミット前：
- [ ] 機密情報が含まれていないか確認
- [ ] .envファイルがgitignoreされているか確認
- [ ] ハードコードされた認証情報がないか確認

定期的に：
- [ ] Git履歴をスキャン
- [ ] 依存パッケージの脆弱性チェック
- [ ] アクセストークンのローテーション

## 🔗 参考リソース

- [GitHub - 機密データの削除](https://docs.github.com/ja/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [AWS - 認証情報のベストプラクティス](https://docs.aws.amazon.com/ja_jp/general/latest/gr/aws-access-keys-best-practices.html)
- [git-secrets](https://github.com/awslabs/git-secrets)
- [12 Factor App - Config](https://12factor.net/ja/config)

## 🚨 緊急時の連絡先

- GitHub Support: https://support.github.com/contact
- 社内セキュリティチーム: [連絡先を記入]

---

**最終更新**: 2025年12月4日
