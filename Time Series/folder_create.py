# Databricks notebook source
from datetime import datetime, timedelta
from dateutil.relativedelta import *
import pytz
import json
utc_now = datetime.utcnow()
import os
import pyspark.pandas as pd

ml_file_path = "/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/ml_model_config.json"
ma_file_path = "/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/ma_model_config.json"
ensemble_file_path = "/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/ensemble_model_config.json"
with open(ml_file_path, 'r') as json_file:
    ml_config = json.load(json_file)
with open(ma_file_path, 'r') as json_file:
    ma_config = json.load(json_file)
with open(ensemble_file_path, 'r') as json_file:
    ensemble_config = json.load(json_file)
# Convert UTC timestamp to IST
ma_model_type = ma_config['model']['type']
ma_model_version = ma_config['model']['type']
ensemble_model_type = ensemble_config['model']['type']
ensemble_model_version = ensemble_config['model']['version']
timestamp_val = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")#str(datetime.now())
curr_date = datetime.now()
curr_date = curr_date - relativedelta(months=+3)
CUR_DATE = curr_date.strftime("%Y-%m") + "-01"
print(CUR_DATE)
ensemble_config["model"]["metrics_start_date"] = CUR_DATE
ml_config["model"]["metrics_start_date"] = CUR_DATE
ma_config["model"]["metrics_start_date"] = CUR_DATE
print(f"Added Metrics start date {CUR_DATE} to all config files")

# COMMAND ----------

directory_path = f"/tmp/anumaan_model_folder_{CUR_DATE}"

# COMMAND ----------

if directory_path and not os.path.exists(directory_path):
    os.makedirs(directory_path)

# COMMAND ----------

il = dbutils.widgets.get("application")
# il = "other_commodity"
ml_config["output"]["write_to_infernce_file"]["path"] = directory_path
ml_config["output"]["metrics_file"]["path"] = directory_path
ml_config["output"]["backtest_file"]["path"] = directory_path
ml_config["output"]["params_file"]["path"] = directory_path
ml_config["output"]["models_file"]["path"] = directory_path
ml_config["input_data"]["train_file"]["path"] = directory_path
ml_config["input_data"]["input_data_file"]["input_file_name"] = f"{il}_train_data"
ml_config["input_data"]["input_data_file"]["path"] = directory_path
ma_config["output"]["write_to_infernce_file"]["path"] = directory_path
ma_config["output"]["metrics_file"]["path"] = directory_path
ma_config["output"]["backtest_file"]["path"] = directory_path
ma_config["output"]["params_file"]["path"] = directory_path
ma_config["output"]["models_file"]["path"] = directory_path
ma_config["input_data"]["train_file"]["path"] = directory_path
ma_config["input_data"]["input_data_file"]["path"] = directory_path
ma_config["input_data"]["input_data_file"]["input_file_name"] = f"{il}_train_data"
ensemble_config["output"]["write_to_infernce_file"]["path"] = directory_path
ensemble_config["output"]["metrics_file"]["path"] = directory_path
ensemble_config["output"]["backtest_file"]["path"] = directory_path
ensemble_config["output"]["params_file"]["path"] = directory_path
ensemble_config["output"]["models_file"]["path"] = directory_path
ensemble_config["input_data"]["train_file"]["path"] = directory_path
ensemble_config["input_data"]["input_data_file"]["path"] = directory_path
ensemble_config["input_data"]["forecasts_data_files"]["ma"]["path"] = directory_path
ensemble_config["input_data"]["forecasts_data_files"]["ml"]["path"] = directory_path
ensemble_config["input_data"]["input_data_file"]["input_file_name"] = f"{il}_train_data"



# COMMAND ----------

os.remove(ensemble_file_path)
os.remove(ma_file_path)
os.remove(ml_file_path)
# Write the modified data back to the JSON file
with open (ml_file_path,'w') as json_file:
    json.dump(ml_config,json_file, indent=4)
with open (ma_file_path,'w') as json_file:
    json.dump(ma_config,json_file, indent=4)
with open(ensemble_file_path, 'w') as json_file:
    json.dump(ensemble_config, json_file, indent=4)

# COMMAND ----------

# %sh ls /tmp/anumaan_model_folder_2023-04-01

# COMMAND ----------

logs_table = f'''CREATE TABLE IF NOT EXISTS adhoc_dev.anumaanlogger (
            experiment_name STRING ,
            experiment_type STRING , 
            start_time STRING,
            end_time STRING,
            Run_Date STRING,
            Application STRING
            );
            '''
spark.sql(logs_table)

# COMMAND ----------

