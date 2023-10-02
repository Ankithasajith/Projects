# Databricks notebook source
import pandas as pd
import pyarrow.feather as feather
import json
from datetime import datetime
import os
from ensemble import ensemble
from data_preprocessing import data_preprocessing

def get_config(config_name):
    config_file = config_name
    with open(config_file, 'r') as openfile:
    # Reading from json file
        configuration = json.load(openfile)
    return configuration

config = get_config('/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/ensemble_model_config.json')
cred_config = get_config('/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/credentials.json')


# COMMAND ----------

# application= "other_commodity"
application = dbutils.widgets.get("application")
# mode = "train"
mode = dbutils.widgets.get("mode")


# COMMAND ----------

def get_sales_data():
        # # input_path = self.config['input_data']['train_file']['path']
        # # data_file_name =  self.config['input_data']['train_file']['file_name']
        # input_data_path = self.config['input_data']['input_data_file']['path']
        # input_file_name = self.config['input_data']['input_data_file']['input_file_name']

        # sales_data = pd.read_csv(f'{input_data_path}/model_input/{input_file_name}')

        # print("*******************",sales_data.columns)
        sales = f"""
            select * from dsapps_staging.anumaantraindata
        """

        sales_data = spark.sql(sales)
        sales_data = sales_data.toPandas()
        return sales_data




# COMMAND ----------

def main(mode,application):
    try :
        print(f" WE are running the ensemble model for --- > {application}")


        if mode == "train":
            print(" Starting the run bois :> ")
            runner = ensemble(config,cred_config)

            if mode == "train":
                cred_file = "/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/credentials.json"
                timestamp_val = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")#str(datetime.now())
                start_time = timestamp_val
                if mode == "train":
                    ensemble_model_type = config['model']['type']
                    ensemble_model_version = config['model']['version']
                    ensemble_name = ensemble_model_type + '_' + ensemble_model_version + '_' + start_time
                    cred_config[f"inference_ensemble_{application}"]["ensemble"]["ensemble_experiment_name"] = ensemble_name
                    cred_config[f"inference_ensemble_{application}"]["ensemble"]["ensemble_start_time"] = start_time
                    os.remove(cred_file)
                    with open(cred_file, 'w') as json_file:
                        json.dump(cred_config, json_file, indent=4)
                    runner.tune(application)
                    end_time = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")#str(datetime.now())
                    cred_config[f"inference_ensemble_{application}"]["ensemble"]["ensemble_end_time"] = end_time
                    os.remove(cred_file)
                    with open(cred_file, 'w') as json_file:
                        json.dump(cred_config, json_file, indent=4)



        if mode == "predict":
            output_path = config['output']['write_to_infernce_file']['path']

            merged_inference = runner.predict("")

            merged_inference[config['model']['inference_columns']].to_csv(f'{output_path}/forecast.csv',index=False)


        return ""
        print(f"Me Happy because code ran successfully ;>")

    except Exception as e :
        print(f"Me sad because the code didnt run :< because of this {e} ")


main(mode,application)




# def get_config(config_name):
#         config_file = config_name
#         with open(config_file, 'r') as openfile:
#         # Reading from json file
#             configuration = json.load(openfile)
#         return configuration


# model_config = get_config('/Users/user/workspace/frappe-bench/apps/anumaan/anumaan/model/notebook_config/comex_ensemble_model_config.json')
# cred_config = get_config('/Users/user/workspace/frappe-bench/apps/anumaan/anumaan/model/notebook_config/credentials.json')


# runner = ensemble(model_config, cred_config)
# runner.read_forecasts_local("backtest_file_name")

# df = runner.predict(pd.DataFrame())
# df = runner.train_for_test(pd.DataFrame())
# df = runner.read_forecasts_local()

# metrics = pd.read_csv('/Users/user/Desktop/metrics/metrics.csv')
# backtest = pd.read_csv('/Users/user/Desktop/metrics/backtest.csv')

# runner.save_sktime_artifacts(metrics,backtest)

# df.to_csv('/Users/user/Desktop/training_files/predict_ensemble_final_dataframe.csv')

# COMMAND ----------

def logger(application):
    Date = datetime.now().strftime("%Y-%m-%d")

    add_data = f"""

    INSERT into adhoc_dev.anumaanlogger 
    Values('{cred_config[f"inference_ensemble_{application}"]["ensemble"]["ensemble_experiment_name"]}',
    '{config["model"]["type"]}',
    '{cred_config[f"inference_ensemble_{application}"]["ensemble"]["ensemble_start_time"]}',
    '{cred_config[f"inference_ensemble_{application}"]["ensemble"]["ensemble_end_time"]}',
    '{Date}',
    '{application}');
    """
    spark.sql(add_data)
    print("Sucessfully inserted logs into table")
    return []
logger(application)

# COMMAND ----------

# %sh ls /tmp/anumaan_model_folder_2023-04-01/ensemble_v1_2023-07-30-15_31_27

# COMMAND ----------

