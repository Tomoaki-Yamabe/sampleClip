#!/bin/bash
# Integrated deployment script for nuScenes Multimodal Search System
# This script deploys the entire system to AWS

set -e  # Exit on error

echo "============================================================"
echo "nuScenes Multimodal Search - Integrated Deployment"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check prerequisites
echo ""
echo "Checking prerequisites..."

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI not found. Please install it first."
    exit 1
fi
print_success "AWS CLI found"

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    print_error "AWS credentials not configured. Run 'aws configure'"
    exit 1
fi
print_success "AWS credentials configured"

# Check CDK
if ! command -v cdk &> /dev/null; then
    print_error "AWS CDK not found. Install with: npm install -g aws-cdk"
    exit 1
fi
print_success "AWS CDK found"

# Check Node.js
if ! command -v node &> /dev/null; then
    print_error "Node.js not found. Please install it first."
    exit 1
fi
print_success "Node.js found"

# Check Python
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    print_error "Python not found. Please install it first."
    exit 1
fi
print_success "Python found"

# Step 1: Convert models to ONNX
echo ""
echo "============================================================"
echo "Step 1: Converting PyTorch models to ONNX"
echo "============================================================"

if [ ! -d "lambda/models" ] || [ -z "$(ls -A lambda/models 2>/dev/null)" ]; then
    echo "Converting models..."
    cd data_preparation
    uvx --with torch --with torchvision --with transformers --with pillow --with numpy --with onnxruntime --with onnxscript python convert_to_onnx.py
    cd ..
    print_success "Models converted to ONNX"
else
    print_warning "ONNX models already exist, skipping conversion"
fi

# Step 2: Install CDK dependencies
echo ""
echo "============================================================"
echo "Step 2: Installing CDK dependencies"
echo "============================================================"

cd infrastructure/cdk
if [ ! -d "node_modules" ]; then
    npm install
    print_success "CDK dependencies installed"
else
    print_warning "CDK dependencies already installed"
fi

# Step 3: Bootstrap CDK (if needed)
echo ""
echo "============================================================"
echo "Step 3: Bootstrapping CDK (if needed)"
echo "============================================================"

# Check if already bootstrapped
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)
REGION=${REGION:-us-east-1}

if aws cloudformation describe-stacks --stack-name CDKToolkit --region $REGION &> /dev/null; then
    print_warning "CDK already bootstrapped"
else
    echo "Bootstrapping CDK..."
    cdk bootstrap aws://$ACCOUNT_ID/$REGION
    print_success "CDK bootstrapped"
fi

# Step 4: Deploy CDK stack
echo ""
echo "============================================================"
echo "Step 4: Deploying CDK stack"
echo "============================================================"

echo "Deploying infrastructure..."
cdk deploy --require-approval never

# Get outputs
API_URL=$(aws cloudformation describe-stacks --stack-name NuScenesSearchStack --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" --output text)
DISTRIBUTION_URL=$(aws cloudformation describe-stacks --stack-name NuScenesSearchStack --query "Stacks[0].Outputs[?OutputKey=='DistributionUrl'].OutputValue" --output text)

print_success "CDK stack deployed"
echo "  API URL: $API_URL"
echo "  CloudFront URL: $DISTRIBUTION_URL"

cd ../..

# Step 5: Build and deploy frontend
echo ""
echo "============================================================"
echo "Step 5: Building and deploying frontend"
echo "============================================================"

cd integ-app/frontend

# Install frontend dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
    print_success "Frontend dependencies installed"
fi

# Build frontend
echo "Building frontend with API URL: $API_URL"
NEXT_PUBLIC_API_URL=$API_URL npm run build

if [ -d "out" ]; then
    print_success "Frontend built successfully"
    
    # Deploy to S3
    FRONTEND_BUCKET=$(aws cloudformation describe-stacks --stack-name NuScenesSearchStack --query "Stacks[0].Outputs[?OutputKey=='FrontendBucketName'].OutputValue" --output text)
    echo "Deploying to S3 bucket: $FRONTEND_BUCKET"
    aws s3 sync out/ s3://$FRONTEND_BUCKET/ --delete
    print_success "Frontend deployed to S3"
    
    # Invalidate CloudFront cache
    DISTRIBUTION_ID=$(aws cloudformation describe-stacks --stack-name NuScenesSearchStack --query "Stacks[0].Outputs[?OutputKey=='DistributionId'].OutputValue" --output text)
    echo "Invalidating CloudFront cache..."
    aws cloudfront create-invalidation --distribution-id $DISTRIBUTION_ID --paths "/*" > /dev/null
    print_success "CloudFront cache invalidated"
else
    print_error "Frontend build failed - out/ directory not found"
    exit 1
fi

cd ../..

# Step 6: Summary
echo ""
echo "============================================================"
echo "Deployment Complete!"
echo "============================================================"
echo ""
echo "Your application is now deployed:"
echo "  API Gateway URL:  $API_URL"
echo "  CloudFront URL:   $DISTRIBUTION_URL"
echo ""
echo "Next steps:"
echo "  1. Visit the CloudFront URL to access the application"
echo "  2. Test text search and image search"
echo "  3. Check CloudWatch Logs for any issues"
echo ""
echo "To clean up resources:"
echo "  cd infrastructure/cdk && cdk destroy"
echo ""
print_success "All done!"
