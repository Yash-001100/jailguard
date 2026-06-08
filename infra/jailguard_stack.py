from aws_cdk import (
    Stack, Duration, RemovalPolicy, CfnOutput,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_elasticache as elasticache,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_dynamodb as dynamodb,
    aws_logs as logs,
    aws_secretsmanager as secretsmanager,
    aws_cloudwatch as cw,
    aws_cloudwatch_actions as cw_actions,
    aws_sns as sns,
)
from constructs import Construct


class JailGuardStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── VPC (2 AZs, 1 NAT gateway to save cost) ───────────────────────────
        vpc = ec2.Vpc(self, "Vpc",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
            ],
        )

        # ── Security groups ────────────────────────────────────────────────────
        ecs_sg = ec2.SecurityGroup(self, "EcsSG",
            vpc=vpc,
            description="JailGuard ECS tasks",
        )
        redis_sg = ec2.SecurityGroup(self, "RedisSG",
            vpc=vpc,
            description="JailGuard Redis",
        )
        redis_sg.add_ingress_rule(ecs_sg, ec2.Port.tcp(6379), "ECS -> Redis")

        # ── ElastiCache Redis (t3.micro single node) ───────────────────────────
        redis_subnet_group = elasticache.CfnSubnetGroup(self, "RedisSubnetGroup",
            description="JailGuard Redis subnets",
            subnet_ids=[s.subnet_id for s in vpc.private_subnets],
        )
        redis = elasticache.CfnCacheCluster(self, "Redis",
            cache_node_type="cache.t3.micro",
            engine="redis",
            num_cache_nodes=1,
            vpc_security_group_ids=[redis_sg.security_group_id],
            cache_subnet_group_name=redis_subnet_group.ref,
        )

        # ── DynamoDB — API key registry + usage logs ───────────────────────────
        keys_table = dynamodb.Table(self, "ApiKeys",
            partition_key=dynamodb.Attribute(
                name="api_key", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
        )
        usage_table = dynamodb.Table(self, "UsageLogs",
            partition_key=dynamodb.Attribute(
                name="api_key", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.RETAIN,
        )

        # ── API key stored in Secrets Manager ─────────────────────────────────
        api_key_secret = secretsmanager.Secret(self, "ApiKeySecret",
            description="JailGuard master API key",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                exclude_punctuation=True,
                password_length=32,
            ),
        )

        # ── ECS Cluster ────────────────────────────────────────────────────────
        cluster = ecs.Cluster(self, "Cluster",
            vpc=vpc,
            container_insights=True,
        )

        # ── Fargate Service + ALB ──────────────────────────────────────────────
        redis_url = f"redis://{redis.attr_redis_endpoint_address}:{redis.attr_redis_endpoint_port}"

        fargate = ecs_patterns.ApplicationLoadBalancedFargateService(self, "Service",
            cluster=cluster,
            cpu=512,          # 0.5 vCPU
            memory_limit_mib=2048,  # 2 GB (model needs ~1.5 GB)
            desired_count=1,
            security_groups=[ecs_sg],
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_asset("../"),  # builds Dockerfile at project root
                container_port=8000,
                environment={
                    "REDIS_URL": redis_url,
                    "SESSION_TTL_SECONDS": "3600",
                    "CORS_ORIGINS": "*",
                },
                secrets={
                    "JAILGUARD_API_KEY": ecs.Secret.from_secrets_manager(api_key_secret),
                },
                log_driver=ecs.LogDrivers.aws_logs(
                    stream_prefix="jailguard",
                    log_retention=logs.RetentionDays.ONE_MONTH,
                ),
            ),
            public_load_balancer=True,
            health_check_grace_period=Duration.seconds(120),
        )

        fargate.target_group.configure_health_check(
            path="/health",
            healthy_http_codes="200",
            interval=Duration.seconds(30),
            timeout=Duration.seconds(10),
            healthy_threshold_count=2,
            unhealthy_threshold_count=3,
        )

        # Grant DynamoDB access to the task role
        keys_table.grant_read_write_data(fargate.task_definition.task_role)
        usage_table.grant_read_write_data(fargate.task_definition.task_role)

        # ── CloudFront (HTTPS + CDN) ───────────────────────────────────────────
        distribution = cloudfront.Distribution(self, "CDN",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.LoadBalancerV2Origin(
                    fargate.load_balancer,
                    protocol_policy=cloudfront.OriginProtocolPolicy.HTTP_ONLY,
                    connection_timeout=Duration.seconds(5),
                    read_timeout=Duration.seconds(30),
                ),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
            ),
        )

        # ── CloudWatch alarms ─────────────────────────────────────────────────
        alert_topic = sns.Topic(self, "Alerts", display_name="JailGuard Alerts")

        # High error rate alarm
        cw.Alarm(self, "HighErrorRate",
            metric=fargate.load_balancer.metric_http_code_elb(
                code=ecs_patterns.ApplicationLoadBalancedServiceRecordType.ALIAS,
            ) if False else cw.Metric(
                namespace="AWS/ApplicationELB",
                metric_name="HTTPCode_ELB_5XX_Count",
                dimensions_map={
                    "LoadBalancer": fargate.load_balancer.load_balancer_full_name
                },
                period=Duration.minutes(5),
                statistic="Sum",
            ),
            threshold=10,
            evaluation_periods=2,
            alarm_description="More than 10 5xx errors in 5 min",
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        ).add_alarm_action(cw_actions.SnsAction(alert_topic))

        # High latency alarm
        cw.Alarm(self, "HighLatency",
            metric=cw.Metric(
                namespace="AWS/ApplicationELB",
                metric_name="TargetResponseTime",
                dimensions_map={
                    "LoadBalancer": fargate.load_balancer.load_balancer_full_name
                },
                period=Duration.minutes(5),
                statistic="p95",
            ),
            threshold=0.5,  # 500ms
            evaluation_periods=3,
            alarm_description="p95 latency > 500ms",
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        ).add_alarm_action(cw_actions.SnsAction(alert_topic))

        # ── Outputs ────────────────────────────────────────────────────────────
        CfnOutput(self, "ApiUrl",
            value=f"https://{distribution.distribution_domain_name}",
            description="JailGuard API — use this URL in your apps",
        )
        CfnOutput(self, "AlbUrl",
            value=f"http://{fargate.load_balancer.load_balancer_dns_name}",
            description="Direct ALB URL (for debugging)",
        )
        CfnOutput(self, "ApiKeySecretArn",
            value=api_key_secret.secret_arn,
            description="Retrieve your API key: aws secretsmanager get-secret-value --secret-id <arn>",
        )
        CfnOutput(self, "AlertTopicArn",
            value=alert_topic.topic_arn,
            description="Subscribe your email: aws sns subscribe --topic-arn <arn> --protocol email --notification-endpoint your@email.com",
        )
