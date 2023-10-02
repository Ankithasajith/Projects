# Databricks notebook source
import pandas as pd
from pyspark.sql.functions import col

# COMMAND ----------

path = "/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/ma_model_config.json"
with open(path,"r") as f:
    config = json.loads(f.read())

file_name = config["input_data"]["input_data_file"]["input_file_name"]
file_path = config["input_data"]["input_data_file"]["path"]
print(f"The folder name is {file_path}/{file_name}.csv")
final_training_data = pd.read_csv((f"{file_path}/{file_name}.csv"))

# COMMAND ----------

anumaan_forecast = spark.sql(
    """
        select sum(a.forecast) as anumaan_forecast ,coalesce(n.planning_linkage,n.item_linkage) as planning_linkage from dsapps_staging.anumaanforecast a join dsapps_staging.dwniyojanplanninglinkage n on a.product = n.item_linkage
        and n.active = 1
        and a.active = 1
        and a.application = "fmcg_demand"
        and a.model_type = "ensemble"
        -- and a.location like "%FMCG%"
        and a.forecast_create_period = "2023-08-01"
        and a.forecast_target_period = "2023-08-01"
        group by coalesce(n.planning_linkage,n.item_linkage) 
    """

)

anumaan_forecast.display()

# COMMAND ----------

snop_data = spark.sql(
    """

        select sum(

        case when nstms.final_target = "" then 0
        else cast(nstms.final_target as double) 
        end

        ) as niyojan_target_sum,
        nstms.planning_linkage from dsapps_staging.dwniyojansalestargetmonthlysnapshot nstms 
        group by nstms.planning_linkage

    """
)

snop_data.display()

# COMMAND ----------

comparison_df = snop_data.join(anumaan_forecast , on="planning_linkage" , how="inner")
comparison_df.display()


# COMMAND ----------

columns = ["niyojan_target_sum","anumaan_forecast"]

sum_df = comparison_df.groupBy().sum()


# COMMAND ----------

sum_df.display()

# COMMAND ----------

actual_data = spark.sql(

    select * from dsapps_stra


)