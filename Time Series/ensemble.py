# Databricks notebook source
import pandas as pd
import numpy as np
import json
import pytz
from datetime import datetime
utc_now = datetime.utcnow()

# Convert UTC timestamp to IST
ist = pytz.timezone('Asia/Kolkata')
ist_now = utc_now.astimezone(ist)

# COMMAND ----------

ml_forecast = """
select * from adhoc_dev.anumaanmlforecast
"""

df_ml = spark.sql(ml_forecast)
df_ml = df_ml.toPandas()



# COMMAND ----------

ma_forecast = """
select * from adhoc_dev.anumaanmaforecast
"""

df_ma = spark.sql(ma_forecast)
df_ma =df_ma.toPandas()

# COMMAND ----------

df_ma

# COMMAND ----------

df_ml

# COMMAND ----------

print(df_ml.shape)
print(df_ma.shape)

# COMMAND ----------

df_ma[['location','product']] = df_ma.id.str.split('|',n=1,expand=True)
df_ml[['location','product']] = df_ml.id.str.split('|',n=1,expand=True)

# COMMAND ----------

df_ma = df_ma[['location','product','forecast_create_period','forecast_target_period','forecast']]
df_ml = df_ml[['location','product','forecast_create_period','forecast_target_period','forecast']]

# COMMAND ----------


cred_config = "/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/credentials.json"
with open(cred_config, 'r') as openfile:
        configuration = json.load(openfile)

# COMMAND ----------

def create_forecast_df(data,application,model):
    df = data
    df['model_type'] = f"{model}"
    df['model_version'] = configuration[f"inference_ensemble_{application}"][f"{model}"][f"{model}_experiment_name"]
    df['level'] = "Warehouse Item Linkage"
    df['frequency'] = "Monthly"
    df['application'] = f"{application}"
    df['create_timestamp'] = ist_now
    df['modified'] = ist_now
    df['active'] = 1
    df = df[["frequency","forecast","model_version","product","application","location","model_type","forecast_create_period","forecast_target_period","create_timestamp","level","modified","active"]]
    return df

# COMMAND ----------

application = dbutils.widgets.get("application")
# application = "other_commodity"
ml = "ml"
df_ml1 = create_forecast_df(df_ml,application,ml)
df_ma1 = create_forecast_df(df_ma,application,"ma")

# COMMAND ----------

df_ma1

# COMMAND ----------

df_ml1

# COMMAND ----------

df_ma.rename(columns={'forecast':"ma"+'_prediction'},inplace=True)
df_ml.rename(columns={'forecast':"ml"+'_prediction'},inplace=True)

# COMMAND ----------

df_ma

# COMMAND ----------

df_ml

# COMMAND ----------

final_df = df_ma.merge(df_ml,how='inner', on=['location','product','forecast_create_period','forecast_target_period'])

# COMMAND ----------

final_df.shape

# COMMAND ----------

final_df

# COMMAND ----------

final_df['id'] = final_df['location'] + '|' + final_df['product']
merged_inference = final_df

# COMMAND ----------

merged_inference.groupby(['forecast_target_period']).agg(MA = ('ma_prediction','sum'),ML = ('ml_prediction','sum'),Record_count = ('location','count')).reset_index()

# COMMAND ----------

# MAGIC %sh ls /tmp
# MAGIC
# MAGIC

# COMMAND ----------

with open("/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/ensemble_model_config.json","r") as file:
    config = json.load(file)
with open("/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/credentials.json") as file:
    cred_config = json.load(file)
column_lst = merged_inference.columns.to_list()
preds = list(filter(lambda k: '_prediction' in k, column_lst))
merged_inference =  merged_inference.fillna(0)
path = config["output"]["write_to_infernce_file"]["path"]
# file_name = cred_config["inference_ensemble_fmcg_demand"]["ensemble"]["ensemble_experiment_name"] + "/model_weights.csv"
file_name = cred_config[f"inference_ensemble_{application}"]["ensemble"]["ensemble_experiment_name"] + "/model_weights.csv"
model_weights = pd.read_csv(f"{path}/{file_name}")

# COMMAND ----------

model_weights

# COMMAND ----------

model_weights.shape

# COMMAND ----------

## Merging the ML-MA forecast dataframe with the model weights
merged_inference = merged_inference.merge(model_weights,how='left',on = ['id']) #.drop(columns=['sum_weights'])

# COMMAND ----------

merged_inference.shape

# COMMAND ----------

# merged_inference["weight_ma_prediction"].fillna
## Filling the empty weights with mean value

merged_inference['weight_ma_prediction'].fillna((merged_inference['weight_ma_prediction'].mean()), inplace=True)
merged_inference['weight_ml_prediction'].fillna((merged_inference['weight_ml_prediction'].mean()), inplace=True)

merged_inference.head()


# COMMAND ----------

merged_inference['weight_ma_prediction'] = merged_inference['weight_ma_prediction'].fillna(1)
merged_inference = merged_inference.fillna(0)

# COMMAND ----------

merged_inference['forecast'] = 0
merged_inference['sum_weight'] = 0
for pred in preds:
    print(pred)
    if f'weight_{pred}' in merged_inference.columns:
        merged_inference['forecast'] = merged_inference['forecast'] + merged_inference[f'{pred}'] * merged_inference[f'weight_{pred}']
        merged_inference['sum_weight'] = merged_inference['sum_weight'] + merged_inference[f'weight_{pred}']

# COMMAND ----------

## Adding the required columns before pushing it into the DB

# merged_inference['model_type'] = "ensemble"
# merged_inference['model_version'] = configuration[f"inference_ensemble_{dbutils.widgets.get('application)}"]["ensemble"]["ensemble_experiment_name"]
# # merged_inference['model_version'] = configuration["inference_ensemble_fmcg_demand"]["ensemble"]["ensemble_experiment_name"]
# merged_inference['level'] = "Warehouse Item Linkage"
# merged_inference['frequency'] = "Monthly"
# merged_inference['application'] = dbutils.widgets.get("application")
# # merged_inference['application'] = "fmcg_demand"
# merged_inference['create_timestamp'] = ist_now
# merged_inference['modified'] = ist_now


merged_inference = create_forecast_df(merged_inference,application,"ensemble")


# COMMAND ----------

## Creating the Final dataframe for in the pushing in the DB 

merged_inference = merged_inference[['frequency','forecast','model_version','product','application','location','model_type','forecast_create_period','forecast_target_period','create_timestamp','level','modified','active']]

# COMMAND ----------

## checking the final forecast before pushing it into the table

merged_inference

# COMMAND ----------

## checking the records

merged_inference.groupby(['forecast_target_period']).agg(forecast = ('forecast','sum'),Record_count = ('location','count')).reset_index()

# COMMAND ----------

logs_table = f'''CREATE TABLE IF NOT EXISTS dsapps_staging.anumaanforecast (
            frequency STRING,
            forecast DECIMAL(18,6),
            model_version STRING,
            product STRING,
            application STRING,
            location STRING,
            model_type STRING,
            forecast_create_period DATE,
            forecast_target_period DATE,
            create_timestamp TIMESTAMP,
            level STRING,
            modified TIMESTAMP,
            active INT
            );
            '''
spark.sql(logs_table)

# COMMAND ----------

# Write to the adhoc table
from pyspark.sql.types import DecimalType,IntegerType
from pyspark.sql import functions as F
from pyspark.sql.functions import col
# spark_ensemble = spark.createDataFrame(merged_inference)
# spark_ma = spark.createDataFrame(df_ma1)
# spark_ml = spark.createDataFrame(df_ml1)
def change_cols(df):
    df = spark.createDataFrame(df)
    df = df.withColumn("forecast",df["forecast"].cast(DecimalType(18,6)))
    df = df.withColumn("forecast_target_period", F.to_date("forecast_target_period"))
    df = df.withColumn("forecast_create_period", F.to_date("forecast_create_period"))
    # df = df.withColumn("create_timestamp", F.to_date("create_timestamp"))
    # df = df.withColumn("modified", F.to_date("modified"))
    df = df.withColumn("active",col("active").cast("Int"))


    df.write.mode("append").saveAsTable(f"dsapps_staging.anumaanforecast")
    return []
change_cols(merged_inference)
change_cols(df_ma1)
change_cols(df_ml1)

# COMMAND ----------

merged_inference.groupby(['forecast_target_period']).agg(forecast = ('forecast','sum'),series_count = ("frequency","count")).reset_index()

# COMMAND ----------

table_name = ["adhoc_dev.anumaanmaforecast","adhoc_dev.anumaanmlforecast"] 
for table in table_name :
    if table:
        spark.sql(f"TRUNCATE TABLE {table}")

        # Confirm data deletion
        if spark.sql(f"SELECT COUNT(*) FROM {table}").first()[0] == 0:
            print(f"Data in table '{table}' successfully deleted.")
        else:
            print(f"Failed to delete data in table '{table}'.")

# COMMAND ----------

fcp = f"""
    select forecast_create_period from dsapps_staging.anumaanforecast group by forecast_create_period
"""

fcp_df = spark.sql(fcp)

fcp_df.display()



# COMMAND ----------

list_fcp  = fcp_df.select("forecast_create_period").rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

for create_period in list_fcp:

    create_period = datetime(year=int(create_period[0:4]), month=int(create_period[4:6]), day=int(create_period[6:8]))


# COMMAND ----------

from datetime import datetime, timedelta


for create_period in list_fcp:

    create_period = datetime(year=int(create_period[0:4]), month=int(create_period[4:6]), day=int(create_period[6:8]))
    
    sql = f"""
    update dsapps_staging.anumaanforecast 
    set active =1 where create_timestamp = (select max(create_timestamp) from dsapps_staging.anumaanforecast 
    where 
    forecast_create_period = {create_period}
    and application = "fmcg_demand") 
    """

    spark.sql(sql)


    print(f"print updated the table for create period {create_period} and application fmcg_demand")



# update dsapps_staging.anumaanforecast set active =0 where create_timestamp != 
# (select max(create_timestamp) from dsapps_staging.anumaanforecast where application = "{application}") 

# COMMAND ----------

