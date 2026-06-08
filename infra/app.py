#!/usr/bin/env python3
import aws_cdk as cdk
from jailguard_stack import JailGuardStack

app = cdk.App()
JailGuardStack(app, "JailGuardStack",
    env=cdk.Environment(region="us-east-1"),
)
app.synth()
