import os
from typing import List, Dict, Any
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, 
    DoubleType, TimestampType, BooleanType
)
from delta import *

class SalesforceDeltaLakeIntegrator:
    def __init__(self, 
                 salesforce_client,
                 base_path: str = "/mnt/delta/salesforce"):
        """Initialize Salesforce Delta Lake integrator"""
        self.spark = (SparkSession.builder
            .appName("SalesforceDeltaLake")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .getOrCreate())
        
        self.salesforce_client = salesforce_client
        self.base_path = base_path
    
    def _get_salesforce_schema(self, object_name: str) -> StructType:
        """Generate dynamic schema for Salesforce objects"""
        schemas = {
            'Opportunity': StructType([
                StructField("Id", StringType(), False),
                StructField("Name", StringType(), True),
                StructField("Amount", DoubleType(), True),
                StructField("StageName", StringType(), True),
                StructField("CloseDate", TimestampType(), True),
                StructField("AccountId", StringType(), True)
            ]),
            'Quote': StructType([
                StructField("Id", StringType(), False),
                StructField("QuoteNumber", StringType(), True),
                StructField("OpportunityId", StringType(), True),
                StructField("TotalPrice", DoubleType(), True),
                StructField("Status", StringType(), True),
                StructField("ExpirationDate", TimestampType(), True),
                StructField("IsSyncing", BooleanType(), True)
            ]),
            'Order': StructType([
                StructField("Id", StringType(), False),
                StructField("OrderNumber", StringType(), True),
                StructField("AccountId", StringType(), True),
                StructField("OpportunityId", StringType(), True),
                StructField("TotalAmount", DoubleType(), True),
                StructField("Status", StringType(), True),
                StructField("EffectiveDate", TimestampType(), True)
            ]),
            'Account': StructType([
                StructField("Id", StringType(), False),
                StructField("Name", StringType(), True),
                StructField("Industry", StringType(), True),
                StructField("Type", StringType(), True),
                StructField("AnnualRevenue", DoubleType(), True)
            ]),
            'Contact': StructType([
                StructField("Id", StringType(), False),
                StructField("FirstName", StringType(), True),
                StructField("LastName", StringType(), True),
                StructField("Email", StringType(), True),
                StructField("AccountId", StringType(), True)
            ])
        }
        
        return schemas.get(object_name)
    
    def fetch_salesforce_data(self, object_name: str, query: str = None) -> List[Dict[str, Any]]:
        """Fetch data from Salesforce object"""
        if not query:
            query = f"SELECT Id, Name FROM {object_name}"
        
        return self.salesforce_client.query(query)['records']
    
    def ingest_to_delta_lake(self, object_name: str):
        """Ingest Salesforce object data to Delta Lake"""
        records = self.fetch_salesforce_data(object_name)
        
        df = self.spark.createDataFrame(
            records, 
            schema=self._get_salesforce_schema(object_name)
        )
        
        delta_path = os.path.join(self.base_path, object_name)
        
        (df.write
            .format("delta")
            .mode("overwrite")
            .save(delta_path))
    
    def merge_salesforce_data(self, object_name: str):
        """Perform incremental merge of Salesforce data"""
        records = self.fetch_salesforce_data(
            object_name, 
            query=f"SELECT Id, Name, LastModifiedDate FROM {object_name} WHERE LastModifiedDate > LAST_N_DAYS:N"
        )
        
        new_df = self.spark.createDataFrame(
            records, 
            schema=self._get_salesforce_schema(object_name)
        )
        
        delta_path = os.path.join(self.base_path, object_name)
        
        deltaTable = DeltaTable.forPath(self.spark, delta_path)
        
        (deltaTable.alias("existing")
            .merge(
                new_df.alias("updates"),
                "existing.Id = updates.Id"
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute())
    
    def query_sales_pipeline(self):
        """Comprehensive sales pipeline analysis"""
        # Load Delta tables
        opportunities = self.query_delta_table('Opportunity')
        quotes = self.query_delta_table('Quote')
        orders = self.query_delta_table('Order')
        
        # Register as temporary views for SQL analysis
        opportunities.createOrReplaceTempView("opportunities")
        quotes.createOrReplaceTempView("quotes")
        orders.createOrReplaceTempView("orders")
        
        # Complex pipeline analysis query
        pipeline_analysis = self.spark.sql("""
            SELECT 
                o.StageName,
                COUNT(DISTINCT o.Id) as total_opportunities,
                SUM(o.Amount) as total_opportunity_amount,
                COUNT(DISTINCT q.Id) as total_quotes,
                SUM(q.TotalPrice) as total_quote_value,
                COUNT(DISTINCT ord.Id) as total_orders,
                SUM(ord.TotalAmount) as total_order_revenue
            FROM 
                opportunities o
            LEFT JOIN 
                quotes q ON o.Id = q.OpportunityId
            LEFT JOIN 
                orders ord ON o.Id = ord.OpportunityId
            GROUP BY 
                o.StageName
        """)
        
        return pipeline_analysis

def main():
    # Example usage (mock Salesforce client)
    class MockSalesforceClient:
        def query(self, soql_query):
            return {
                'records': [
                    {'Id': '001', 'Name': 'Enterprise Deal', 
                     'Amount': 100000.0, 'StageName': 'Negotiation'},
                    {'Id': 'Q001', 'QuoteNumber': 'Q-2023-001', 
                     'OpportunityId': '001', 'TotalPrice': 95000.0},
                    {'Id': 'ORD001', 'OrderNumber': 'ORD-2023-001', 
                     'OpportunityId': '001', 'TotalAmount': 90000.0}
                ]
            }
    
    sf_client = MockSalesforceClient()
    delta_integrator = SalesforceDeltaLakeIntegrator(sf_client)
    
    # Ingest and merge multiple objects
    for obj in ['Opportunity', 'Quote', 'Order']:
        delta_integrator.ingest_to_delta_lake(obj)
        delta_integrator.merge_salesforce_data(obj)
    
    # Perform sales pipeline analysis
    pipeline_analysis = delta_integrator.query_sales_pipeline()
    pipeline_analysis.show()

if __name__ == "__main__":
    main()
