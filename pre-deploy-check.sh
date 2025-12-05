#!/bin/bash
# Pre-deployment check script
# Validates that all required files and dependencies are in place

set -e

echo "============================================================"
echo "Pre-Deployment Check"
echo "============================================================"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1 (missing)"
        ((ERRORS++))
    fi
}

check_dir() {
    if [ -d "$1" ] && [ "$(ls -A $1 2>/dev/null)" ]; then
        echo -e "${GREEN}✓${NC} $1 ($(ls $1 | wc -l) files)"
    else
        echo -e "${RED}✗${NC} $1 (missing or empty)"
        ((ERRORS++))
    fi
}

check_command() {
    if command -v $1 &> /dev/null; then
        VERSION=$($1 --version 2>&1 | head -n1)
        echo -e "${GREEN}✓${NC} $1 ($VERSION)"
    else
        echo -e "${RED}✗${NC} $1 (not found)"
        ((ERRORS++))
    fi
}

echo ""
echo "Checking required commands..."
check_command aws
check_command cdk
check_command node
check_command npm
check_command python

echo ""
echo "Checking AWS credentials..."
if aws sts get-caller-identity &> /dev/null; then
    ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
    REGION=$(aws configure get region)
    echo -e "${GREEN}✓${NC} AWS credentials configured"
    echo "  Account: $ACCOUNT"
    echo "  Region: ${REGION:-us-east-1}"
else
    echo -e "${RED}✗${NC} AWS credentials not configured"
    ((ERRORS++))
fi

echo ""
echo "Checking Lambda code..."
check_file "lambda/lambda_function.py"
check_file "lambda/encoders.py"
check_file "lambda/encoders_onnx.py"
check_file "lambda/vector_db.py"
check_file "lambda/exceptions.py"
check_file "lambda/Dockerfile"
check_file "lambda/requirements.txt"

echo ""
echo "Checking ONNX models..."
check_dir "lambda/models"
if [ -d "lambda/models" ]; then
    check_file "lambda/models/text_transformer.onnx"
    check_file "lambda/models/text_projector.onnx"
    check_file "lambda/models/image_features.onnx"
    check_file "lambda/models/image_projector.onnx"
fi

echo ""
echo "Checking data files..."
check_file "integ-app/backend/app/model/vector_db.json"
check_file "integ-app/backend/app/model/scenes_with_umap.json"
check_dir "data_preparation/extracted_data/images"

echo ""
echo "Checking CDK infrastructure..."
check_file "infrastructure/cdk/package.json"
check_file "infrastructure/cdk/cdk.json"
check_file "infrastructure/cdk/lib/nuscenes-search-stack.ts"

echo ""
echo "Checking frontend..."
check_file "integ-app/frontend/package.json"
check_file "integ-app/frontend/next.config.ts"
check_file "integ-app/frontend/src/app/page.tsx"

echo ""
echo "============================================================"
echo "Check Summary"
echo "============================================================"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "You can now run the deployment:"
    echo "  ./deploy.sh (Linux/Mac)"
    echo "  ./deploy.ps1 (Windows)"
    exit 0
else
    echo -e "${RED}✗ $ERRORS error(s) found${NC}"
    echo ""
    echo "Please fix the errors before deploying."
    exit 1
fi
