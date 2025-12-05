import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as s3deploy from 'aws-cdk-lib/aws-s3-deployment';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as apigateway from 'aws-cdk-lib/aws-apigatewayv2';
import * as apigatewayIntegrations from 'aws-cdk-lib/aws-apigatewayv2-integrations';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import { Construct } from 'constructs';
import * as path from 'path';
import * as fs from 'fs';

export class NuScenesSearchStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // ========================================
    // S3 Bucket: Data Storage (Models, Vector DB, Images)
    // ========================================
    const dataBucket = new s3.Bucket(this, 'DataBucket', {
      bucketName: `nuscenes-search-data-${this.account}`,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: cdk.RemovalPolicy.RETAIN, // 本番環境では保持
      autoDeleteObjects: false,
      cors: [
        {
          allowedMethods: [
            s3.HttpMethods.GET,
            s3.HttpMethods.HEAD,
          ],
          allowedOrigins: ['*'],
          allowedHeaders: ['*'],
          maxAge: 3600,
        },
      ],
    });

    // ========================================
    // S3 Bucket: Frontend Static Hosting
    // ========================================
    const frontendBucket = new s3.Bucket(this, 'FrontendBucket', {
      bucketName: `nuscenes-search-frontend-${this.account}`,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    // ========================================
    // Lambda Function: Search API Container
    // ========================================
    // CDKが自動的に以下を実行：
    // 1. Dockerイメージをビルド
    // 2. ECRリポジトリを作成（自動管理）
    // 3. イメージをECRにプッシュ
    // 4. Lambda関数を作成
    // 5. cdk destroyでECRも自動削除
    const searchFunction = new lambda.DockerImageFunction(this, 'SearchFunction', {
      functionName: 'nuScenes-search',
      code: lambda.DockerImageCode.fromImageAsset('../../lambda', {
        file: 'Dockerfile',
      }),
      memorySize: 512,
      timeout: cdk.Duration.seconds(30),
      environment: {
        DATA_BUCKET: dataBucket.bucketName,
        VECTOR_DB_KEY: 'vector_db.json',
        TEXT_MODEL_KEY: 'models/text_projector.pt',
        IMAGE_MODEL_KEY: 'models/image_projector.pt',
        // S3 Vectors configuration (set USE_S3_VECTORS=true to enable)
        USE_S3_VECTORS: 'false',  // Change to 'true' when S3 Vectors is available
        VECTOR_BUCKET_NAME: 'mcap-search-vectors',
        TEXT_INDEX_NAME: 'scene-text-embeddings',
        IMAGE_INDEX_NAME: 'scene-image-embeddings',
        METADATA_KEY: 'metadata/scenes_metadata.json',
      },
      logRetention: logs.RetentionDays.ONE_WEEK,
    });

    // S3読み取り権限を付与
    dataBucket.grantRead(searchFunction);
    
    // S3 Vectors権限を付与（USE_S3_VECTORS=trueの場合に必要）
    // 注意: 現在の環境ではSCPによりS3 Vectorsアクセスが制限されている可能性があります
    searchFunction.addToRolePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        's3vectors:QueryVectors',
        's3vectors:GetVectors',
        's3vectors:ListVectors',
      ],
      resources: [
        `arn:aws:s3vectors:${this.region}:${this.account}:bucket/mcap-search-vectors/*`,
      ],
    }));

    // ========================================
    // S3 Bucket Deployment: Models and Data
    // ========================================
    // Deploy ONNX models to S3
    const modelsDeployment = new s3deploy.BucketDeployment(this, 'ModelsDeployment', {
      sources: [s3deploy.Source.asset(path.join(__dirname, '../../../lambda/models'))],
      destinationBucket: dataBucket,
      destinationKeyPrefix: 'models/',
      prune: false, // Don't delete existing files
    });

    // Deploy vector database and metadata
    const dataDeployment = new s3deploy.BucketDeployment(this, 'DataDeployment', {
      sources: [s3deploy.Source.asset(path.join(__dirname, '../../../integ-app/backend/app/model'), {
        exclude: ['*.pt'], // Exclude PyTorch models, only deploy JSON files
      })],
      destinationBucket: dataBucket,
      destinationKeyPrefix: 'data/',
      prune: false,
    });

    // Deploy scene images
    const imagesDeployment = new s3deploy.BucketDeployment(this, 'ImagesDeployment', {
      sources: [s3deploy.Source.asset(path.join(__dirname, '../../../data_preparation/extracted_data/images'))],
      destinationBucket: dataBucket,
      destinationKeyPrefix: 'images/',
      prune: false,
    });

    // Lambda function depends on data being deployed
    searchFunction.node.addDependency(modelsDeployment);
    searchFunction.node.addDependency(dataDeployment);

    // ========================================
    // API Gateway HTTP API
    // ========================================
    const httpApi = new apigateway.HttpApi(this, 'HttpApi', {
      apiName: 'nuscenes-search-api',
      description: 'nuScenes Multimodal Search API',
      corsPreflight: {
        allowOrigins: ['*'],
        allowMethods: [
          apigateway.CorsHttpMethod.GET,
          apigateway.CorsHttpMethod.POST,
          apigateway.CorsHttpMethod.OPTIONS,
        ],
        allowHeaders: ['Content-Type', 'Authorization'],
        maxAge: cdk.Duration.hours(1),
      },
    });

    // Lambda統合
    const lambdaIntegration = new apigatewayIntegrations.HttpLambdaIntegration(
      'LambdaIntegration',
      searchFunction
    );

    // ルート設定 - /search/text
    httpApi.addRoutes({
      path: '/search/text',
      methods: [apigateway.HttpMethod.POST],
      integration: lambdaIntegration,
    });

    // ルート設定 - /search/image
    httpApi.addRoutes({
      path: '/search/image',
      methods: [apigateway.HttpMethod.POST],
      integration: lambdaIntegration,
    });

    // ========================================
    // CloudFront Distribution
    // ========================================
    // Origin Access Identity for S3
    const originAccessIdentity = new cloudfront.OriginAccessIdentity(this, 'OAI', {
      comment: 'OAI for nuScenes Search Frontend',
    });

    // Grant CloudFront read access to frontend bucket
    frontendBucket.grantRead(originAccessIdentity);

    // ========================================
    // Frontend Deployment
    // ========================================
    // Note: Frontend must be built before CDK deployment
    // Run: cd integ-app/frontend && API_URL=<api-url> node build-for-cdk.js
    // The 'out' directory will be deployed to S3
    const frontendPath = path.join(__dirname, '../../../integ-app/frontend/out');
    
    let frontendDeployment;
    if (fs.existsSync(frontendPath)) {
      frontendDeployment = new s3deploy.BucketDeployment(this, 'FrontendDeployment', {
        sources: [s3deploy.Source.asset(frontendPath)],
        destinationBucket: frontendBucket,
        // distribution: distribution, // Will be set after distribution is created
        // distributionPaths: ['/*'], // Invalidate all paths
      });
      console.log('✓ Frontend deployment configured');
    } else {
      console.warn('⚠ Frontend build not found at:', frontendPath);
      console.warn('  Run: cd integ-app/frontend && API_URL=<api-url> node build-for-cdk.js');
    }

    // CloudFront Distribution
    const distribution = new cloudfront.Distribution(this, 'Distribution', {
      comment: 'nuScenes Search Frontend',
      defaultBehavior: {
        origin: new origins.S3Origin(frontendBucket, {
          originAccessIdentity: originAccessIdentity,
        }),
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        cachePolicy: new cloudfront.CachePolicy(this, 'FrontendCachePolicy', {
          defaultTtl: cdk.Duration.hours(24),
          maxTtl: cdk.Duration.days(7),
          minTtl: cdk.Duration.seconds(0),
          queryStringBehavior: cloudfront.CacheQueryStringBehavior.none(),
        }),
      },
      additionalBehaviors: {
        '/api/*': {
          origin: new origins.HttpOrigin(
            `${httpApi.httpApiId}.execute-api.${this.region}.amazonaws.com`
          ),
          cachePolicy: cloudfront.CachePolicy.CACHING_DISABLED,
          viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.HTTPS_ONLY,
          allowedMethods: cloudfront.AllowedMethods.ALLOW_ALL,
        },
      },
      defaultRootObject: 'index.html',
      errorResponses: [
        {
          httpStatus: 404,
          responseHttpStatus: 200,
          responsePagePath: '/index.html',
          ttl: cdk.Duration.minutes(5),
        },
      ],
    });

    // Configure CloudFront invalidation for frontend deployment
    if (frontendDeployment) {
      frontendDeployment.node.addDependency(distribution);
      // Note: BucketDeployment automatically invalidates CloudFront when distribution is provided
      // But we need to set it after distribution is created
      const cfnDeployment = frontendDeployment.node.defaultChild as cdk.CfnResource;
      cfnDeployment.addPropertyOverride('DistributionId', distribution.distributionId);
      cfnDeployment.addPropertyOverride('DistributionPaths', ['/*']);
    }

    // ========================================
    // Outputs
    // ========================================
    new cdk.CfnOutput(this, 'DataBucketName', {
      value: dataBucket.bucketName,
      description: 'S3 bucket for models and data',
      exportName: 'NuScenesSearchDataBucket',
    });

    new cdk.CfnOutput(this, 'FrontendBucketName', {
      value: frontendBucket.bucketName,
      description: 'S3 bucket for frontend static assets',
      exportName: 'NuScenesSearchFrontendBucket',
    });

    new cdk.CfnOutput(this, 'FunctionName', {
      value: searchFunction.functionName,
      description: 'Lambda Function Name',
      exportName: 'NuScenesSearchFunctionName',
    });

    new cdk.CfnOutput(this, 'FunctionArn', {
      value: searchFunction.functionArn,
      description: 'Lambda Function ARN',
      exportName: 'NuScenesSearchFunctionArn',
    });

    new cdk.CfnOutput(this, 'ApiUrl', {
      value: httpApi.url!,
      description: 'API Gateway URL',
      exportName: 'NuScenesSearchApiUrl',
    });

    new cdk.CfnOutput(this, 'DistributionUrl', {
      value: `https://${distribution.distributionDomainName}`,
      description: 'CloudFront Distribution URL',
      exportName: 'NuScenesSearchDistributionUrl',
    });

    new cdk.CfnOutput(this, 'DistributionId', {
      value: distribution.distributionId,
      description: 'CloudFront Distribution ID',
      exportName: 'NuScenesSearchDistributionId',
    });
  }
}
