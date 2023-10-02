# Databricks notebook source
# spark.conf.set("spark.driver.maxResultSize", "8g")
import json
from unittest import runner
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

#from anumaan.model.demand_forcast_ml_runner import ml_runner
from demand_forcast_ml_runner import ml_runner
from data_preprocessing import data_preprocessing
import pyarrow.feather as feather


# COMMAND ----------

mode = dbutils.widgets.get("mode")
application = dbutils.widgets.get("application")
# mode = "train"
# application = "other_commodity" 

# COMMAND ----------

def get_config(config_name):
    config_file = config_name
    print(config_name)
    with open(config_file, 'r') as openfile:
    # Reading from json file
        configuration = json.load(openfile)
    return configuration

ml_config = get_config('/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/ml_model_config.json')
credential_config = get_config('/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/credentials.json')
ensemble_config = get_config('/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/ensemble_model_config.json')



# COMMAND ----------

""" collect data """
def get_sales_data(config):

    configuration = config
    # output_path = configuration['output']['models_file']['path']

    input_data_path = configuration['input_data']['input_data_file']['path']
    input_file_name = configuration['input_data']['input_data_file']['input_file_name']

    # sales_data = pd.read_csv(f'{input_data_path}/model_input/{input_file_name}')
    print(f"file_name - {input_data_path}/{input_file_name}.csv")
    sales_data = pd.read_csv(f'{input_data_path}/{input_file_name}.csv')
    return sales_data

sales_data = get_sales_data(ml_config)
sales_data.head()





# COMMAND ----------

def get_preprocessed_data(sales_data):
    output_path = ml_config['input_data']['train_file']['path']
    timestamp_val = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    print(f"output path - {output_path}")
    generate_training_data = ml_config['input_data']['train_file']['generate_training_data']
    if generate_training_data is True:
        #sales_data = pd.read_csv('item_linkage_city_data_3_wh.csv')
        print(sales_data.shape)
        input_preprocess = data_preprocessing(ml_config,credential_config)
        preprocessed_sales_data = input_preprocess.default_clean_impute_data(sales_data)
        sales_data_final = preprocessed_sales_data['sales_data_final']
        df_metadata = preprocessed_sales_data['metadata']
        print(" *********** saved preprocessed data **********")
        # feather.write_feather(sales_data_final, f'{output_path}/training_{self.config["model"]["application"]}_sales_data_{timestamp_val}.ftr')
        # feather.write_feather(df_metadata, f'{output_path}/training_{self.config["model"]["application"]}_metadata_{timestamp_val}.ftr')

    else:
        #read data file with features to be used for training
        sales_data_final = ml_config['input_data']['train_file']['sales_data_filter']
        df_metadata = ml_config['input_data']['train_file']['meta_data_filter']

        print(" *********** reading preprocessed data **********")
        sales_data_final = feather.read_feather(f'{output_path}/{sales_data_final}')
        df_metadata = feather.read_feather(f'{output_path}/{df_metadata}')

        preprocessed_sales_data = {'sales_data_final':sales_data_final, 'metadata':df_metadata}

    return preprocessed_sales_data
    
preprocessed_sales_data = get_preprocessed_data(sales_data)

# COMMAND ----------

def remove_column_if_exists(df, column_name):
    if column_name in df.columns:
        df.drop(column_name, axis=1, inplace=True)
        print(f"Column '{column_name}' removed.")
    else:
        print(f"Column '{column_name}' does not exist in the DataFrame.")



# COMMAND ----------

def train_runner(application,preprocessed_sales_data):
    # mode = dbutils.widgets.get("mode")
    runner = ml_runner(ml_config,credential_config)
    print("This is the preprocessed sales data ----------------------\n",preprocessed_sales_data)
    runner.tune(preprocessed_sales_data,application)
    return 0



# COMMAND ----------

def inference_runner(application,preprocessed_sales_data):
    runner = ml_runner(ml_config,credential_config)
    output_path = ml_config['output']['models_file']['path']
    experiment_name = credential_config[f"inference_ensemble_{application}"]["ml"]['ml_experiment_name']
    # experiment_name = credential_config[f"inference_ensemble_other_commodity"]["ml"]['ml_experiment_name']
    final_output = runner.predictiction(application,preprocessed_sales_data)
    column_name_to_remove = 'Unnamed: 0'
    remove_column_if_exists(final_output, column_name_to_remove)
    path = f'{output_path}/{experiment_name}'
    print(path)
    if os.path.exists(path) is False:
        os.mkdir(path)
    # print(final_output_df.columns)
    spark_df= spark.createDataFrame(final_output)
    # Define the table name to be deleted
    table_name = f"adhoc_dev.anumaanmlforecast"  # Replace with your actual table name
    # Drop the table if it exists using SQL query
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")
    # Confirm table deletion
    if table_name not in spark.catalog.listTables():
        print(f"Table '{table_name}' successfully deleted or did not exist.")
    else:
        print(f"Failed to delete table '{table_name}'.")
    spark_df.write.mode("overwrite").saveAsTable("adhoc_dev.anumaanmlforecast")
    # final_output.to_csv(path+'/ml_forecast.csv',index=False)
    print(final_output)
    # final_output.to_csv(path+"/"+"ml_prediction_file.csv")
    return 0



# COMMAND ----------

def main(application,mode):
    cred_file = "/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/credentials.json"
    ensemble_file = "/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/ensemble_model_config.json"
    timestamp_val = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")#str(datetime.now())
    start_time = timestamp_val
    if mode == "train":
        ml_model_type = ml_config['model']['type']
        ml_model_version = ml_config['model']['version']
        ml_name = ml_model_type + '_' + ml_model_version + '_' + start_time
        credential_config[f"inference_ensemble_{application}"]["ml"]["ml_experiment_name"] = ml_name
        # credential_config[f"inference_ensemble_other_commodity"]["ml"]["ml_experiment_name"] = ml_name

        ensemble_config["input_data"]["forecasts_data_files"]["ml"]["folder"] = ml_name
        credential_config[f"inference_ensemble_{application}"]["ml"]["ml_start_time"] = start_time
        # credential_config[f"inference_ensemble_other_commodity"]["ml"]["ml_start_time"] = start_time

        os.remove(ensemble_file)
        os.remove(cred_file)
        with open(cred_file, 'w') as json_file:
            json.dump(credential_config, json_file, indent=4)
        with open(ensemble_file, 'w') as json_file:
            json.dump(ensemble_config, json_file, indent=4)
        train_runner(application,preprocessed_sales_data)
        inference_runner(application,preprocessed_sales_data)
        end_time = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")#str(datetime.now())
        credential_config[f"inference_ensemble_{application}"]["ml"]["ml_end_time"] = end_time
        # credential_config[f"inference_ensemble_other_commodity"]["ml"]["ml_end_time"] = end_time
        os.remove(cred_file)
        with open(cred_file, 'w') as json_file:
            json.dump(credential_config, json_file, indent=4)


    elif mode == "inference":
        inference_runner(preprocessed_sales_data)
    

main(application,mode)
        

# COMMAND ----------

def logger(application):    # Push to logger table
    Date = datetime.now().strftime("%Y-%m-%d")

    add_data = f"""

    INSERT into adhoc_dev.anumaanlogger 
    Values('{credential_config[f"inference_ensemble_{application}"]["ml"]["ml_experiment_name"]}',
    '{ml_config["model"]["type"]}',
    '{credential_config[f"inference_ensemble_{application}"]["ml"]["ml_start_time"]}',
    '{credential_config[f"inference_ensemble_{application}"]["ml"]["ml_end_time"]}',
    '{Date}',
    '{application}');
    """

    print(add_data)
    spark.sql(add_data)
    print("Sucessfully inserted logs into table")
    return []
logger(application)

# COMMAND ----------

