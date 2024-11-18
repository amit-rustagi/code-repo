import os
import subprocess
from pyspark.sql import SparkSession
import boto3

# Configuration Settings
SNOWFLAKE_CONFIG = {
    "account": "your_account_name",
    "user": "your_username",
    "password": "your_password",
    "warehouse": "your_warehouse",
    "database": "your_database",
    "schema": "your_schema"
}

AWS_S3_CONFIG = {
    "bucket_name": "your-bucket-name",
    "export_path": "s3://your-bucket-name/snowflake_export/",
    "iceberg_warehouse_path": "s3://your-bucket-name/iceberg_warehouse/"
}

EXPORT_CHUNK_SIZE_MB = 100  # Maximum file size for Snowflake export

# Step 1: Extract Data from Snowflake
def extract_data_from_snowflake(table_name):
    """Export data from Snowflake to S3 using the COPY INTO command."""
    sql_command = f"""
    COPY INTO '{AWS_S3_CONFIG['export_path']}'
    FROM {table_name}
    FILE_FORMAT = (TYPE = PARQUET)
    MAX_FILE_SIZE = {EXPORT_CHUNK_SIZE_MB} MB
    HEADER = TRUE
    OVERWRITE = TRUE
    CREDENTIALS = (
        AWS_KEY_ID='your_aws_key'
        AWS_SECRET_KEY='your_aws_secret'
    );
    """
    print(f"Exporting data from Snowflake table: {table_name}")
    # Run Snowflake SQL command using Snowflake CLI (snowsql) or Snowflake Python connector
    # This is a simplified subprocess call assuming snowsql is configured
    with open("snowflake_export.sql", "w") as file:
        file.write(sql_command)
    subprocess.run(["snowsql", "-f", "snowflake_export.sql"])

# Step 2: Transform Data Using Spark
def transform_data_with_spark(table_name):
    """Transform data from S3 export to Iceberg-compatible format."""
    spark = SparkSession.builder \
        .appName("Snowflake to Iceberg Migration") \
        .config("spark.sql.catalog.my_catalog", "org.apache.iceberg.spark.SparkCatalog") \
        .config("spark.sql.catalog.my_catalog.type", "hadoop") \
        .config("spark.sql.catalog.my_catalog.warehouse", AWS_S3_CONFIG['iceberg_warehouse_path']) \
        .getOrCreate()

    print("Reading exported Parquet data from S3...")
    # Read data from S3
    input_path = os.path.join(AWS_S3_CONFIG['export_path'], table_name)
    df = spark.read.parquet(input_path)

    # Transform data (example: renaming columns, partitioning)
    print("Transforming data...")
    transformed_df = df.withColumnRenamed("old_column_name", "new_column_name") \
                       .repartition("partition_column")

    # Write transformed data to Iceberg
    print("Writing data to Iceberg table...")
    iceberg_table_path = f"my_catalog.{AWS_S3_CONFIG['bucket_name']}.{table_name}"
    transformed_df.write \
        .format("iceberg") \
        .mode("overwrite") \
        .save(iceberg_table_path)

    print(f"Transformation complete for table: {table_name}")

# Step 3: Validate Data
def validate_data(snowflake_table, iceberg_table, spark):
    """Compare row counts between Snowflake and Iceberg."""
    print("Validating data consistency...")
    # Query Snowflake for row count (use Snowflake Python connector for actual implementation)
    snowflake_row_count = 1000  # Replace with actual row count query
    iceberg_row_count = spark.read.format("iceberg").load(iceberg_table).count()

    if snowflake_row_count == iceberg_row_count:
        print(f"Validation successful for table {snowflake_table}: {snowflake_row_count} rows matched.")
    else:
        print(f"Validation failed for table {snowflake_table}: Snowflake({snowflake_row_count}) vs Iceberg({iceberg_row_count})")

# Step 4: Monitor Progress
def monitor_migration():
    """Track migration progress with AWS S3 metrics."""
    print("Monitoring migration progress...")
    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")
    result = paginator.paginate(Bucket=AWS_S3_CONFIG['bucket_name'], Prefix="iceberg_warehouse/")
    total_files = sum(1 for _ in result)
    print(f"Total files in Iceberg warehouse: {total_files}")

# Step 5: Main Migration Workflow
def main():
    tables_to_migrate = ["table1", "table2", "table3"]  # List your Snowflake tables here

    for table in tables_to_migrate:
        print(f"Starting migration for table: {table}")
        extract_data_from_snowflake(table)
        transform_data_with_spark(table)
        # Spark session is reused for validation
        spark = SparkSession.builder.getOrCreate()
        validate_data(table, f"my_catalog.{AWS_S3_CONFIG['bucket_name']}.{table}", spark)
        monitor_migration()

if __name__ == "__main__":
    main()

