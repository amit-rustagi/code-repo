import boto3
import json
from typing import Dict, List

def discover_aws_services():
    """
    Scan and discover services used in an AWS account across multiple regions.
    
    Returns:
    Dict containing discovered services for each supported service type
    """
    # List of AWS services to check
    service_checks = {
        'EC2': check_ec2_resources,
        'RDS': check_rds_resources,
        'S3': check_s3_resources,
        'Lambda': check_lambda_resources,
        'ECS': check_ecs_resources,
        'EKS': check_eks_resources,
        'DynamoDB': check_dynamodb_resources,
        'CloudWatch': check_cloudwatch_resources
    }
    
    # Discover services across all supported regions
    discovered_services = {}
    
    try:
        # Get list of all regions
        ec2_client = boto3.client('ec2')
        regions = [region['RegionName'] for region in ec2_client.describe_regions()['Regions']]
        
        # Check services in each region
        for service_name, check_func in service_checks.items():
            service_resources = []
            for region in regions:
                try:
                    region_resources = check_func(region)
                    service_resources.extend(region_resources)
                except Exception as e:
                    print(f"Error checking {service_name} in {region}: {str(e)}")
            
            # Only add service if resources are found
            if service_resources:
                discovered_services[service_name] = service_resources
    
    except Exception as e:
        print(f"Error in service discovery: {str(e)}")
    
    return discovered_services

def check_ec2_resources(region: str) -> List[Dict]:
    """Check EC2 instances in a specific region"""
    ec2 = boto3.client('ec2', region_name=region)
    instances = ec2.describe_instances()
    return [
        {
            'instance_id': instance['InstanceId'],
            'type': instance['InstanceType'],
            'state': instance['State']['Name']
        } 
        for reservation in instances['Reservations'] 
        for instance in reservation['Instances']
    ]

def check_rds_resources(region: str) -> List[Dict]:
    """Check RDS instances in a specific region"""
    rds = boto3.client('rds', region_name=region)
    instances = rds.describe_db_instances()
    return [
        {
            'db_instance_id': db['DBInstanceIdentifier'],
            'engine': db['Engine'],
            'status': db['DBInstanceStatus']
        } 
        for db in instances['DBInstances']
    ]

def check_s3_resources(region: str = None) -> List[Dict]:
    """Check S3 buckets (not region-specific)"""
    s3 = boto3.client('s3')
    buckets = s3.list_buckets()
    return [
        {
            'bucket_name': bucket['Name']
        } 
        for bucket in buckets['Buckets']
    ]

def check_lambda_resources(region: str) -> List[Dict]:
    """Check Lambda functions in a specific region"""
    lambda_client = boto3.client('lambda', region_name=region)
    functions = lambda_client.list_functions()
    return [
        {
            'function_name': func['FunctionName'],
            'runtime': func['Runtime']
        } 
        for func in functions['Functions']
    ]

def check_ecs_resources(region: str) -> List[Dict]:
    """Check ECS clusters in a specific region"""
    ecs = boto3.client('ecs', region_name=region)
    clusters = ecs.list_clusters()
    return [
        {
            'cluster_arn': cluster
        } 
        for cluster in clusters['clusterArns']
    ]

def check_eks_resources(region: str) -> List[Dict]:
    """Check EKS clusters in a specific region"""
    eks = boto3.client('eks', region_name=region)
    clusters = eks.list_clusters()
    return [
        {
            'cluster_name': cluster
        } 
        for cluster in clusters['clusters']
    ]

def check_dynamodb_resources(region: str) -> List[Dict]:
    """Check DynamoDB tables in a specific region"""
    dynamodb = boto3.client('dynamodb', region_name=region)
    tables = dynamodb.list_tables()
    return [
        {
            'table_name': table
        } 
        for table in tables['TableNames']
    ]

def check_cloudwatch_resources(region: str) -> List[Dict]:
    """Check CloudWatch alarms in a specific region"""
    cloudwatch = boto3.client('cloudwatch', region_name=region)
    alarms = cloudwatch.describe_alarms()
    return [
        {
            'alarm_name': alarm['AlarmName'],
            'state': alarm['StateValue']
        } 
        for alarm in alarms['MetricAlarms']
    ]

def main():
    # Discover and print services
    discovered_services = discover_aws_services()
    
    # Pretty print the discovered services
    print(json.dumps(discovered_services, indent=2))

if __name__ == "__main__":
    main()
