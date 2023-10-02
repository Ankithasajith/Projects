# Databricks notebook source
from ma import ma
import json
import pandas as pd
import os
from datetime import datetime

import metrics
from data_preprocessing import data_preprocessing
import json
import pyarrow.feather as feather

# COMMAND ----------

mode = dbutils.widgets.get("mode")
# mode = "train"
application = dbutils.widgets.get("application")
# application = "other_commodity"

# COMMAND ----------

def get_config(config_name):
    config_file = config_name
    with open(config_file, 'r') as openfile:
    # Reading from json file
        configuration = json.load(openfile)
    return configuration

ma_config = get_config('/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/ma_model_config.json')
credential_config = get_config('/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/credentials.json')
ensemble_config = get_config('/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/ensemble_model_config.json')



# COMMAND ----------

def get_sales_data(config):

    configuration = config
    output_path = configuration['output']['models_file']['path']
    input_data_path = configuration['input_data']['input_data_file']['path']
    input_file_name = configuration['input_data']['input_data_file']['input_file_name']
    sales_data = pd.read_csv(f'{input_data_path}/{input_file_name}.csv')
    input_preprocess = data_preprocessing(ma_config,credential_config)
    preprocessed_sales_data = input_preprocess.default_clean_impute_data(sales_data)
    sales_data_final = preprocessed_sales_data['sales_data_final']
    df_metadata = preprocessed_sales_data['metadata']
    return preprocessed_sales_data

preprocessed_sales_data = get_sales_data(ma_config)


# COMMAND ----------

def remove_column_if_exists(df, column_name):
    if column_name in df.columns:
        df.drop(column_name, axis=1, inplace=True)
        print(f"Column '{column_name}' removed.")
    else:
        print(f"Column '{column_name}' does not exist in the DataFrame.")


# COMMAND ----------

def train_runner(preprocessed_sales_data,application):
    runner = ma(ma_config,credential_config)

    ma_output = runner.tune(preprocessed_sales_data,application)

# train_runner(preprocessed_sales_data)

# COMMAND ----------

def inference_runner(preprocessed_sales_data,application):
        output_path = ma_config['output']['models_file']['path']
        experiment_name = credential_config[f"inference_ensemble_{application}"]["ma"]["ma_experiment_name"]
        # experiment_name = credential_config[f"inference_ensemble_fmcg_demand"]["ma"]["ma_experiment_name"]

        runner = ma(ma_config,credential_config)
        final_output_df = runner.predict(preprocessed_sales_data,application)
        column_name_to_remove = 'Unnamed: 0'
        remove_column_if_exists(final_output_df, column_name_to_remove)
        path = f'{output_path}/{experiment_name}'


        print(path)
        if os.path.exists(path) is False:
            os.mkdir(path)

        spark_df= spark.createDataFrame(final_output_df)
        table_name = f"adhoc_dev.anumaanmaforecast"  # Replace with your actual table name
        # Drop the table if it exists using SQL query
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")
        # Confirm table deletion
        if table_name not in spark.catalog.listTables():
            print(f"Table '{table_name}' successfully deleted or did not exist.")
        else:
            print(f"Failed to delete table '{table_name}'.")
        spark_df.write.mode("overwrite").saveAsTable("adhoc_dev.anumaanmaforecast")

# inference_runner(preprocessed_sales_data)

# COMMAND ----------

def main(application,mode):
    cred_file = "/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/credentials.json"
    ensemble_file = "/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/ensemble_model_config.json"
    timestamp_val = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")#str(datetime.now())
    start_time = timestamp_val
    if mode == "train":
        ma_model_type = ma_config['model']['type']
        ma_model_version = ma_config['model']['version']
        ma_name = ma_model_type + '_' + ma_model_version + '_' + start_time
        credential_config[f"inference_ensemble_{application}"]["ma"]["ma_experiment_name"] = ma_name
        credential_config[f"inference_ensemble_{application}"]["ma"]["ma_start_time"] = start_time
        ensemble_config["input_data"]["forecasts_data_files"]["ma"]["folder"] = ma_name
        os.remove(cred_file)
        os.remove(ensemble_file)
        with open(cred_file, 'w') as json_file:
            json.dump(credential_config, json_file, indent=4)
        with open(ensemble_file, 'w') as json_file:
            json.dump(ensemble_config, json_file, indent=4)
        train_runner(preprocessed_sales_data,application)
        inference_runner(preprocessed_sales_data,application)
        end_time = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")#str(datetime.now())
        credential_config[f"inference_ensemble_{application}"]["ma"]["ma_end_time"] = end_time
        os.remove(cred_file)
        with open(cred_file, 'w') as json_file:
            json.dump(credential_config, json_file, indent=4)


    elif mode == "inference":
        inference_runner(preprocessed_sales_data)
    

main(application,mode)
        


# COMMAND ----------

# %sh ls /tmp/anumaan_model_folder_2023-04-01/ma_v1_2023-07-30-15_18_43

# COMMAND ----------

def best_window(application):    
    path = ma_config["input_data"]["input_data_file"]["path"]+"/"+credential_config[f"inference_ensemble_{application}"]["ma"]["ma_experiment_name"]+"/ma_filtered_metrics.csv" 
    path_config = "/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/ensemble_model_config.json"
    path_cred_config = "/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/credentials.json"
    data = pd.read_csv(path)
    data["sum"] = data["wmape"] + abs(data["bias_pct"])
    selected_window = data[data["sum"]== data["sum"].min()].reset_index(drop=True)
    # selected_window = selected_window["ma_window"].reset_index(drop=True)
    ma_window = selected_window['ma_window'].iloc[0]
    print(selected_window)
    with open(path_config,"r") as f:
        config = json.loads(f.read())

    config["input_data"]["forecasts_data_files"]["ma"]["backtest_file_name"] = f"backtest_window_{ma_window}.ftr"
    credential_config[f"inference_ensemble_{application}"]["ma"]["backtest_file_name"] = f"backtest_window_{ma_window}.ftr"
    print(config["input_data"]["forecasts_data_files"]["ma"]["backtest_file_name"])
    os.remove(path_config)
    os.remove(path_cred_config)

    with open (path_config,'w') as json_file:
        json.dump(config,json_file, indent=4)
    with open (path_cred_config,'w') as json_file:
        json.dump(credential_config,json_file, indent=4)
    return ma_window


# COMMAND ----------

print("The best training window is ",best_window(application))

# COMMAND ----------

def logger(application):
    Date = datetime.now().strftime("%Y-%m-%d")

    add_data = f"""

    INSERT into adhoc_dev.anumaanlogger  
    Values('{credential_config[f"inference_ensemble_{application}"]["ma"]["ma_experiment_name"]}',
    '{ma_config["model"]["type"]}',
    '{credential_config[f"inference_ensemble_{application}"]["ma"]["ma_start_time"]}',
    '{credential_config[f"inference_ensemble_{application}"]["ma"]["ma_end_time"]}',
    '{Date}',
    '{application}');
    """
    spark.sql(add_data)
    print("Sucessfully inserted logs into table")
    return [] 
logger(application)

# COMMAND ----------

