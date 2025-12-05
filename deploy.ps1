# Integrated deployment script for nuScenes Multimodal Search System
# This script deploys the entire system to AWS

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "nuScenes Multimodal Search - Integrated Deployment" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

function Print-Success {
    param($Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Print-Error {
    param($Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Print-Warning {
    param($Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

# Check prerequisites
Write-Host ""
Write-Host "Checking prerequisites..."

# Check AWS CLI
if (!(Get-Command aws -ErrorAction SilentlyContinue)) {
    Print-Error "AWS CLI not found. Please install it first."
    exit 1
}
Print-Success "AWS CLI found"

# Check AWS credentials
try {
    aws sts get-caller-identity | Out-Null
    Print-Success "AWS credentials configured"
} catch {
    Print-Error "AWS credentials not configured. Run 'aws configure'"
    exit 1
}

# Check CDK
if (!(Get-Command cdk -ErrorAction SilentlyContinue)) {
    Print-Error "AWS CDK not found. Install with: npm install -g aws-cdk"
    exit 1
}
Print-Success "AWS CDK found"

# Check Node.js
if (!(Get-Command node -ErrorAction SilentlyContinue)) {
    Print-Error "Node.js not found. Please install it first."
    exit 1
}
Print-Success "Node.js found"

# Check Python/uvx
if (!(Get-Command uvx -ErrorAction SilentlyContinue)) {
    Print-Error "uvx not found. Please install uv first."
    exit 1
}
Print-Success "uvx found"

# Step 1: Convert models to ONNX
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Step 1: Converting PyTorch models to ONNX" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if (!(Test-Path "lambda/models") -or ((Get-ChildItem "lambda/models" -ErrorAction SilentlyContinue).Count -eq 0)) {
    Write-Host "Converting models..."
    Push-Location data_preparation
    uvx --with torch --with torchvision --with transformers --with pillow --with numpy --with onnxruntime --with onnxscript python convert_to_onnx.py
    Pop-Location
    Print-Success "Models converted to ONNX"
} else {
    Print-Warning "ONNX models already exist, skipping conversion"
}

# Step 2: Install CDK dependencies
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Step 2: Installing CDK dependencies" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Push-Location infrastructure/cdk
if (!(Test-Path "node_modules")) {
    npm install
    Print-Success "CDK dependencies installed"
} else {
    Print-Warning "CDK dependencies already installed"
}

# Step 3: Bootstrap CDK (if needed)
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Step 3: Bootstrapping CDK (if needed)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$AccountId = (aws sts get-caller-identity --query Account --output text)
$Region = (aws configure get region)
if ([string]::IsNullOrEmpty($Region)) {
    $Region = "us-east-1"
}

try {
    aws cloudformation describe-stacks --stack-name CDKToolkit --region $Region | Out-Null
    Print-Warning "CDK already bootstrapped"
} catch {
    Write-Host "Bootstrapping CDK..."
    cdk bootstrap "aws://$AccountId/$Region"
    Print-Success "CDK bootstrapped"
}

# Step 4: Deploy CDK stack
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Step 4: Deploying CDK stack" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host "Deploying infrastructure..."
cdk deploy --require-approval never

# Get outputs
$ApiUrl = (aws cloudformation describe-stacks --stack-name NuScenesSearchStack --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" --output text)
$DistributionUrl = (aws cloudformation describe-stacks --stack-name NuScenesSearchStack --query "Stacks[0].Outputs[?OutputKey=='DistributionUrl'].OutputValue" --output text)

Print-Success "CDK stack deployed"
Write-Host "  API URL: $ApiUrl"
Write-Host "  CloudFront URL: $DistributionUrl"

Pop-Location

# Step 5: Build and deploy frontend
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Step 5: Building and deploying frontend" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Push-Location integ-app/frontend

# Install frontend dependencies
if (!(Test-Path "node_modules")) {
    Write-Host "Installing frontend dependencies..."
    npm install
    Print-Success "Frontend dependencies installed"
}

# Build frontend
Write-Host "Building frontend with API URL: $ApiUrl"
$env:NEXT_PUBLIC_API_URL = $ApiUrl
npm run build

if (Test-Path "out") {
    Print-Success "Frontend built successfully"
    
    # Deploy to S3
    $FrontendBucket = (aws cloudformation describe-stacks --stack-name NuScenesSearchStack --query "Stacks[0].Outputs[?OutputKey=='FrontendBucketName'].OutputValue" --output text)
    Write-Host "Deploying to S3 bucket: $FrontendBucket"
    aws s3 sync out/ "s3://$FrontendBucket/" --delete
    Print-Success "Frontend deployed to S3"
    
    # Invalidate CloudFront cache
    $DistributionId = (aws cloudformation describe-stacks --stack-name NuScenesSearchStack --query "Stacks[0].Outputs[?OutputKey=='DistributionId'].OutputValue" --output text)
    Write-Host "Invalidating CloudFront cache..."
    aws cloudfront create-invalidation --distribution-id $DistributionId --paths "/*" | Out-Null
    Print-Success "CloudFront cache invalidated"
} else {
    Print-Error "Frontend build failed - out/ directory not found"
    exit 1
}

Pop-Location

# Step 6: Summary
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Deployment Complete!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Your application is now deployed:"
Write-Host "  API Gateway URL:  $ApiUrl"
Write-Host "  CloudFront URL:   $DistributionUrl"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Visit the CloudFront URL to access the application"
Write-Host "  2. Test text search and image search"
Write-Host "  3. Check CloudWatch Logs for any issues"
Write-Host ""
Write-Host "To clean up resources:"
Write-Host "  cd infrastructure/cdk; cdk destroy"
Write-Host ""
Print-Success "All done!"
