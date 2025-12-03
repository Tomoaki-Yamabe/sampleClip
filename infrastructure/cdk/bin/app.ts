#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { NuScenesSearchStack } from '../lib/nuscenes-search-stack';

const app = new cdk.App();

new NuScenesSearchStack(app, 'NuScenesSearchStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: 'us-west-2', 
  },
  description: 'nuScenes Multimodal Search Infrastructure',
});
