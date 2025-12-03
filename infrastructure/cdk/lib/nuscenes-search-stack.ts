import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as logs from 'aws-cdk-lib/aws-logs';
// import * as apigateway from 'aws-cdk-lib/aws-apigatewayv2';
// import * as apigatewayIntegrations from 'aws-cdk-lib/aws-apigatewayv2-integrations';
// import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
// import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import { Construct } from 'constructs';

export class NuScenesSearchStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // ========================================
    // S3 Bucket: Raw Data Storage
    // ========================================
    const dataBucket = new s3.Bucket(this, 'DataBucket', {
      bucketName: `nuscenes-search-data-${this.account}`,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: cdk.RemovalPolicy.RETAIN, // 本番環境では保持
      autoDeleteObjects: false,
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
      },
      logRetention: logs.RetentionDays.ONE_WEEK,
    });

    // S3読み取り権限を付与
    dataBucket.grantRead(searchFunction);

    // ========================================
    // API Gateway HTTP API（コメントアウト）
    // ========================================
    // 注意: API Gatewayが制限されている環境では、Lambda Function URLを使用してください
    // 
    // const httpApi = new apigateway.HttpApi(this, 'HttpApi', {
    //   apiName: 'nuscenes-search-api',
    //   description: 'nuScenes Multimodal Search API',
    //   corsPreflight: {
    //     allowOrigins: ['*'],
    //     allowMethods: [
    //       apigateway.CorsHttpMethod.GET,
    //       apigateway.CorsHttpMethod.POST,
    //       apigateway.CorsHttpMethod.OPTIONS,
    //     ],
    //     allowHeaders: ['Content-Type', 'Authorization'],
    //     maxAge: cdk.Duration.hours(1),
    //   },
    // });
    //
    // // Lambda統合
    // const lambdaIntegration = new apigatewayIntegrations.HttpLambdaIntegration(
    //   'LambdaIntegration',
    //   searchFunction
    // );
    //
    // // ルート設定
    // httpApi.addRoutes({
    //   path: '/{proxy+}',
    //   methods: [apigateway.HttpMethod.ANY],
    //   integration: lambdaIntegration,
    // });

    // ========================================
    // Lambda Function URL（API Gatewayの代替）
    // ========================================
    // 注意: 現在の環境ではSCPによりLambda Function URLの作成が制限されています
    // 本番環境では以下のコメントを解除してください
    // const functionUrl = searchFunction.addFunctionUrl({
    //   authType: lambda.FunctionUrlAuthType.NONE,
    //   cors: {
    //     allowedOrigins: ['*'],
    //     allowedMethods: [lambda.HttpMethod.ALL],
    //     allowedHeaders: ['*'],
    //     maxAge: cdk.Duration.hours(1),
    //   },
    // });

    // ========================================
    // S3 Bucket: フロントエンド静的ホスティング（オプション）
    // ========================================
    // 注意: 本番環境では有効化してください
    // const frontendBucket = new s3.Bucket(this, 'FrontendBucket', {
    //   bucketName: `nuscenes-search-frontend-${this.account}`,
    //   encryption: s3.BucketEncryption.S3_MANAGED,
    //   blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
    //   removalPolicy: cdk.RemovalPolicy.DESTROY,
    //   autoDeleteObjects: true,
    // });

    // ========================================
    // CloudFront Distribution（オプション）
    // ========================================
    // 注意: CloudFrontへのアクセスが制限されている環境ではコメントアウト
    // 本番環境では有効化してください
    // const distribution = new cloudfront.Distribution(this, 'Distribution', {
    //   comment: 'nuScenes Search Frontend',
    //   defaultBehavior: {
    //     origin: new origins.S3Origin(frontendBucket),
    //     viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
    //     cachePolicy: new cloudfront.CachePolicy(this, 'FrontendCachePolicy', {
    //       defaultTtl: cdk.Duration.hours(24),
    //       maxTtl: cdk.Duration.days(7),
    //       minTtl: cdk.Duration.seconds(0),
    //       queryStringBehavior: cloudfront.CacheQueryStringBehavior.none(),
    //     }),
    //   },
    //   defaultRootObject: 'index.html',
    //   errorResponses: [
    //     {
    //       httpStatus: 404,
    //       responseHttpStatus: 200,
    //       responsePagePath: '/index.html',
    //       ttl: cdk.Duration.minutes(5),
    //     },
    //   ],
    // });

    // ========================================
    // Outputs
    // ========================================
    new cdk.CfnOutput(this, 'DataBucketName', {
      value: dataBucket.bucketName,
      description: 'S3 bucket for models and data',
      exportName: 'NuScenesSearchDataBucket',
    });


    new cdk.CfnOutput(this, 'FunctionName', {
      value: searchFunction.functionName,
      description: 'Lambda Function Name (invoke directly using AWS CLI or SDK)',
      exportName: 'NuScenesSearchFunctionName',
    });

    new cdk.CfnOutput(this, 'FunctionArn', {
      value: searchFunction.functionArn,
      description: 'Lambda Function ARN',
      exportName: 'NuScenesSearchFunctionArn',
    });

    // Lambda Function URL（現在の環境では制限されています）
    // new cdk.CfnOutput(this, 'FunctionUrl', {
    //   value: functionUrl.url,
    //   description: 'Lambda Function URL',
    //   exportName: 'NuScenesSearchFunctionUrl',
    // });

    // API Gateway使用時のみ有効化
    // new cdk.CfnOutput(this, 'ApiUrl', {
    //   value: httpApi.url!,
    //   description: 'API Gateway URL',
    //   exportName: 'NuScenesSearchApiUrl',
    // });

    // CloudFront有効時のみ使用
    // new cdk.CfnOutput(this, 'DistributionUrl', {
    //   value: `https://${distribution.distributionDomainName}`,
    //   description: 'CloudFront Distribution URL',
    //   exportName: 'NuScenesSearchDistributionUrl',
    // });

    // new cdk.CfnOutput(this, 'DistributionId', {
    //   value: distribution.distributionId,
    //   description: 'CloudFront Distribution ID',
    //   exportName: 'NuScenesSearchDistributionId',
    // });
  }
}
