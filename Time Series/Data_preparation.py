# Databricks notebook source
import pyspark.pandas as pd
import numpy as np
# import pyarrow.feather as feather
import numpy as np
import pandas as pd
import os
from datetime import datetime
import pickle
import json
from sklearn.preprocessing import LabelEncoder
from pyspark.sql.functions import sum, count, max, min, avg



# COMMAND ----------

# state = dbutils.widgets.get("state")
# state = "Maharashtra"
# print(state)
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# COMMAND ----------

anumaan_monthly_data = f"""
select * from dsapps_core.anumaanmonthlysalesforecasttraining 
"""
df_fmcg = spark.sql(anumaan_monthly_data)

df_fmcg.show()


# COMMAND ----------

from pyspark.sql.functions import col
convert_cols = ['order_sales', 'stock_qty', 'price', 'rlp', 'mrp', 'sum_stock_value_in_month', 'avg_stock_value_in_month', 'total_balance_qty_in_month', 'avg_balance_qty_in_month', 'reserved_qty', 'start_month_stock', 'end_month_stock']
df_fmcg.printSchema
for cols in convert_cols :
    print(cols)
    print(type(df_fmcg[cols].values))
    df_fmcg = df_fmcg.withColumn(cols, col(cols).cast('float'))

    


# COMMAND ----------

df_fmcg_pd = df_fmcg.toPandas()

# COMMAND ----------

df_fmcg_pd

# COMMAND ----------

import datetime
start_date = "2022-01-01"
current_timestamp = datetime.datetime.now()
# Add three months to the current timestamp
three_months_later = current_timestamp + datetime.timedelta(days=3 * 30)
end_date = three_months_later.strftime("%Y-%m-%d")


# COMMAND ----------

exp = """ 

with ilv_month_start_dates as (
select distinct start_of_month,end_of_month, 1 as dummy from core.dimdate
where date >= '{0}' and date <= '{1}'
),
wh_data as (
 select
 concat(dw.city,'~',TRIM(SUBSTRING_INDEX(SUBSTRING_INDEX(dw.name, '-', -2), '-', 1 ) )) wh_city,
 dw.name as warehouse_name,global_id,city,state,
 actual_start_date,active,warehouse_type
 from 
 core.dimwarehouse dw
where dw.active = 1
    and disabled = 0
    and lower(name) like "%fmcg%"
    and lower(name)  not like "%PUNE%"
    and lower(name)  not like "%mdt4%"
    and lower(name)  not like "% urban%"
    and lower(name)  not like "%kirun%"
    and lower(name)  not like "%comex%" 
    and lower(name) not like "%Homerun%"
    and global_id is not null 
    and global_id <> 'NULL'
    and warehouse_type = 'Main' 
 ),
ilv_promo as (
SELECT distinct
    IFNULL(pbr.brand, pricitem.brand) as brand
    , pspd.warehouse
     ,ps.apply_on
    ,if((ps.apply_on = "Brand" and (ifnull(ps.er_category_to_apply_on,"") != "")), 'sub_cateogry', ps.apply_on) as `apply_on_custom`
    ,ps.er_category_to_apply_on
    ,if((ps.apply_on = "Brand" and (ifnull(ps.er_category_to_apply_on,"") != "" or ps.er_category_to_apply_on = '')),pbr.brand ,'') as `promo_brand`
    ,if((ps.apply_on = "Brand" and (ifnull(ps.er_category_to_apply_on,"") != "")),ps.er_category_to_apply_on ,'') as  `promo_item_sub_category`
    ,if(ps.apply_on != "Brand",pricitem.item_code ,'') as  `promo_item_code`
    -- , case when ps.apply_on = "Brand" and (ps.er_category_to_apply_on is NULL or ps.er_category_to_apply_on = '') then item.item_code
    --       when ps.apply_on = "Brand" and ps.er_category_to_apply_on is not NULL and  ps.er_category_to_apply_on != '' and ps.er_category_to_apply_on = item.item_sub_category then item.item_code
    --       when ps.apply_on != "Brand" then pricitem.item_code
    --       else null end as item_code
    -- -- , pspd.priority 
    , ps.name 
    -- , ps.disable
    -- , ps.workflow_state
    -- , ps.selling
    -- , ps.buying
    -- , ps.apply_on
    -- , ps.er_category_to_apply_on
    , ps.valid_from
    , ps.valid_upto
    , pspd.rule_description
    , pspd.discount_percentage 
    , pspd.discount_amount
    -- , pspd.min_qty 
    -- , pspd.max_qty 
    -- , pspd.min_amount
    -- , pspd.max_amount
    -- , pspd.apply_multiple_pricing_rules
    -- , pspd.threshold_percentage
    , pspd.rate_or_discount
    , 1 as dummy
    -- , pspd.price_list
    -- , IFNULL(pbr.uom, pricitem.uom) as uom
FROM
    staging.tabpromotionalscheme AS ps 
JOIN 
    staging.tabpromotionalschemepricediscount pspd 
ON
    pspd.parent = ps.name and pspd.parenttype = "Promotional Scheme"
LEFT JOIN 
    staging.dwpricingrulebrand pbr 
ON 
    pbr.parent = ps.name AND pbr.active = 1 and pbr.parenttype = "Promotional Scheme"
LEFT JOIN 
    staging.dwpricingruleitemcode pricitem
ON 
    pricitem.parent = ps.name AND pricitem.active = 1 and pricitem.parenttype = "Promotional Scheme"
-- LEFT JOIN 
--     core.dimitem item 
-- ON 
--     item.brand = pbr.brand AND item.active = TRUE
where
 (ps.valid_upto >= '{0}' OR ps.valid_upto IS NULL)
),
-- select * from 
-- ilv_promo
ilv_promo_final as (
select  
    warehouse promo_warehouse
    ,apply_on
    ,apply_on_custom
    ,er_category_to_apply_on
    ,promo_brand
    ,promo_item_sub_category
    ,promo_item_code
    -- ,item_code
    , (A.start_of_month) as month 
    ,count(distinct if(A.start_of_month between date_add(LAST_DAY(B.valid_from - INTERVAL 1 DAY - INTERVAL 1 month),1) and B.valid_upto, B.name, null)) as schemes
    ,avg(if(A.start_of_month between date_add(LAST_DAY(B.valid_from - INTERVAL 1 DAY - INTERVAL 1 month),1) and B.valid_upto, B.discount_amount, null)) as avg_discount_amount
    ,avg(distinct if(A.start_of_month between date_add(LAST_DAY(B.valid_from - INTERVAL 1 DAY - INTERVAL 1 month),1) and B.valid_upto, B.discount_percentage, null)) as avg_discount_percentage
    ,max(if(A.start_of_month between date_add(LAST_DAY(B.valid_from - INTERVAL 1 DAY - INTERVAL 1 month),1)  and B.valid_upto,DATEDIFF(least(A.end_of_month , B.valid_upto),greatest(A.start_of_month , B.valid_from ))+1,0)) as days_of_promo
from ilv_month_start_dates A
inner join ilv_promo B on A.dummy = B.dummy 
group by month, warehouse
,apply_on
,apply_on_custom
    ,er_category_to_apply_on
    ,promo_brand
    ,promo_item_sub_category
    ,promo_item_code
)

select 
-- pf.*, di.item_linkage item_linkage 
    month
    ,pf.promo_warehouse
    -- ,apply_on
    ,apply_on_custom apply_on
    -- ,er_category_to_apply_on
    ,promo_brand
    ,promo_item_sub_category
    ,di.item_linkage item_linkage
    ,avg(schemes) schemes
    ,avg(avg_discount_amount) avg_discount_amount
    ,avg(avg_discount_percentage) avg_discount_percentage
    ,max(days_of_promo) max_days_of_promo
    ,avg(days_of_promo) avg_days_of_promo
from 
ilv_promo_final pf
left join core.dimitem di on pf.promo_item_code = di.item_code and di.active = 1
where 
1 = 1 
and days_of_promo != 0
-- and pf.promo_warehouse = 'FMCG West Bengal - ER'
group by promo_warehouse
,apply_on
,apply_on_custom
    ,er_category_to_apply_on
    ,promo_brand
    ,promo_item_sub_category
    ,di.item_linkage
    ,month
""".format(str(start_date),str(end_date))




df_exp = spark.sql(exp)







# COMMAND ----------

df_exp_pd = df_exp.toPandas()

# COMMAND ----------

df_exp_pd

# COMMAND ----------

# df_exp_pd.groupby(["month"]).agg({'month': [np.min,np.max]})

# COMMAND ----------

# df_exp_pd[df_exp_pd["promo_warehouse"]=="FMCG Maharashtra - ER"].shape

# COMMAND ----------

hier = """ 

with ilv_whprofile as
(
select * from 
(select warehouse,replenishment_center,
row_number()over(partition by warehouse order by modified desc) as rnk
from 
staging.dwWarehouseProfile
where active = 1 ) whp
where rnk = 1 
),

ilv_warehouse_with_profile as (
select 
dwh.name
,dwh.reporting_warehouse
,dwh.city
,dwh.state
,dwh.co_located
,dwh.global_id
,dwh.warehouse_type
-- ,coalesce(whp.replenishment_center, 'No-RC') as replenishment_center_1
, case when dwh.warehouse_type = 'Sector'
            then coalesce(whp_sector.replenishment_center,'No-RC') 
    else 
        coalesce(whp.replenishment_center, 'No-RC') 
    end as replenishment_center
,case when IFNULL(wh_crc.replenishment_center,'') != "" then 1 else 0 end as is_crc
,dwh.physical_warehouse
,dwh.active
,dwh.disabled
,dwh.inactive
,dwh.actual_start_date
,dwh.modified
from core.dimwarehouse dwh
 left join ilv_whprofile whp on whp.warehouse = dwh.name 
 left join ilv_whprofile whp_sector on whp_sector.warehouse = dwh.physical_warehouse
 left join (select distinct replenishment_center from ilv_whprofile) wh_crc on wh_crc.replenishment_center = dwh.name 
where dwh.active = 1 
-- and disabled = 0 
and dwh.name not like '%% - D'
    -- and dwh.global_id is not null 
    and IFNULL(dwh.global_id,"") != ""
    and dwh.global_id != 'NULL'
    and lower(dwh.name) not like '%dark%' 
    and upper(dwh.name) not like '%R-URBAN%'
      and upper(dwh.name) not like '%-KIRUN%'
        and upper(dwh.name) not like '%HOMERUN-%'
        and upper(dwh.name) not like '% URBAN %'
     --  and upper(wh.name) not like '%-RURBAN%'
    and lower(dwh.name) not like '%cac%'
    and dwh.warehouse_type in ('Main','Sector','Transit')
),

ilv_all_wh as (
select global_id,replenishment_center,city,state,name warehouse,modified from (
    select global_id,replenishment_center,city,state,name,modified,
    row_number()over(partition by city,global_id,replenishment_center order by modified desc) as rnk
    from ilv_warehouse_with_profile
    where disabled = 0 and active = 1 and inactive = 0 
    and upper(name) not like '%MDT4%'
        and name like '%FMCG%'
        and upper(name) not like '%R-URBAN%'
        and upper(name) not like '%-KIRUN%'
        and upper(name) not like '%HOMERUN-%'
        and upper(name) not like '% URBAN %'
         -- and upper(name) not like '%-RURBAN%'
        and name not like '%% - D'
    and lower(name) not like '%dark%' 
    and lower(name) not like '%cac%'
    and lower(warehouse_type)='main'
        -- and city = 'Tiruvannamalai'
        ) X where rnk = 1
),
wh_hierarchy as (
select ref_parent higher_hier_warehouse,ref_docname warehouse
from staging.tabnestedsetparent nsp
where nsp.ref_doctype = 'Warehouse' 
)
select 
A.higher_hier_warehouse,B.* 
from 
wh_hierarchy A 
inner join ilv_all_wh B 
on A.warehouse = B.warehouse
"""
df_hier = spark.sql(hier)
df_hier_pd = df_hier.toPandas()

# COMMAND ----------

# df_hier_pd

# COMMAND ----------

def removeDuplication(df):
    df_g = df.groupby(['id','month_start_date']).agg(rec_count=('id','count')).reset_index()
    # df_g = df.groupBy(['id','month_start_date']).agg(count('id').alias('rec_count'))
    df_ggg = df_g[df_g['rec_count']>1]
    print(df_ggg)

    df_final = pd.merge(df, df_ggg, on=['id','month_start_date'], how='outer', indicator=True)\
       .query("_merge != 'both'")\
       .drop('_merge', axis=1)\
       .reset_index(drop=True)

    # df_g = df_final.groupby(['id','month_start_date']).agg(rec_count=('id','count')).reset_index()

    # print(df_g.shape)

    return df_final

# COMMAND ----------

il = """

with ilv_dimitem as
(
    select item_sub_category,er_item_type,planning_linkage as item_linkage,brand from
    (
        select 
             row_number()over(partition by coalesce(di.item_linkage,item_code) order by di.modified desc) as rnk
             ,coalesce(di.item_linkage,item_code) as planning_linkage
            ,item_sub_category
            ,derived_er_item_type as er_item_type
            ,case when derived_er_item_type = 'FMCG brands' then brand else derived_er_item_type end as brand
        from core.dimitem di
        -- left join staging.dwniyojanplanninglinkage npl
           --  on di.item_linkage = npl.item_linkage
            -- and npl.active=1
        where 1=1
        and di.active=1
        and di.end_date > current_date
    )a
    where rnk=1
    and planning_linkage is not null
)
select * from ilv_dimitem

"""
df_il = spark.sql(il)
df_il_pd = df_il.toPandas()
# df_il_pd.head()

# COMMAND ----------

# df_il_pd

# COMMAND ----------

df_fmcg_pd['month_start_date'] = df_fmcg_pd['month_start_date'].astype(str)
df_fmcg_pd = removeDuplication(df_fmcg_pd)


# COMMAND ----------

df_item = df_il_pd[['item_linkage','er_item_type']]
df_item.head()
print(df_item.er_item_type.unique())
df_with_item = df_fmcg_pd.merge(df_item,how='inner',on=['item_linkage'])
# item_type = dbutils.widgets.get("item_type")
item_type = 'FMCG brands'
df_with_item_fmcg = df_with_item[df_with_item['er_item_type']==item_type]
df_with_item_fmcg.drop(['er_item_type'],axis=1,inplace=True)
df_with_item_fmcg.head()

# COMMAND ----------

# df_with_item_fmcg['month_start_date'] = df_with_item_fmcg['month_start_date'].astype(str)
df_exp_pd['month'] = df_exp_pd['month'].astype(str)
df_hier_pd['modified'] = df_hier_pd['modified'].astype(str)
df_exp_pd['month'] = df_exp_pd['month'].str.slice(0,10)
df_hier_pd['modified'] = df_hier_pd['modified'].str.slice(0,10)

# COMMAND ----------

type(df_with_item_fmcg['month_start_date'].iloc[0])

# COMMAND ----------

df_with_item_fmcg = df_with_item_fmcg.dropna(subset=['id'])

# COMMAND ----------

df_with_item_fmcg.shape

# COMMAND ----------

promo_df_final = df_exp_pd.merge(df_hier_pd,how='left',left_on="promo_warehouse",right_on="higher_hier_warehouse")

# COMMAND ----------

promo_df_final['location'] =promo_df_final['city']+'~'+ promo_df_final['global_id']+'~'+promo_df_final['replenishment_center']
print(promo_df_final.shape)
promo_df_final = promo_df_final[~promo_df_final['location'].isnull()]
print(promo_df_final.shape)

# COMMAND ----------

df_temp = promo_df_final.groupby(['location','month','promo_item_sub_category','item_linkage','promo_brand','apply_on'],dropna=False)\
.agg(schemes=('schemes','sum'),avg_discount_amount=('avg_discount_amount','mean'),avg_discount_percentage=('avg_discount_percentage','mean'),
max_days_of_promo=('max_days_of_promo','max'), avg_days_of_promo=('avg_days_of_promo','mean')).reset_index()


# COMMAND ----------

df_temp['promo_brand']

# COMMAND ----------

promo_df_final = df_temp

# COMMAND ----------

df_fmcg_final = df_with_item_fmcg

print(df_fmcg_final.columns)
print(promo_df_final.columns)

df_fmcg_final['location'] =df_fmcg_final['city']+'~'+ df_fmcg_final['global_id']+'~'+df_fmcg_final['replenishment_center']

print(df_fmcg_final.shape)

# COMMAND ----------

promo_df_final['schemes']

target_columns  = ['schemes', 'avg_discount_amount','avg_discount_percentage', 'max_days_of_promo', 'avg_days_of_promo']
promo_df_final['key'] = promo_df_final['location']+'|'+promo_df_final['promo_item_sub_category']+'|'+promo_df_final['item_linkage']+'|'+promo_df_final['promo_brand']+'|'+promo_df_final['apply_on']
leads = [0,1,2]

promo_df_final.shape

# COMMAND ----------

key_column = 'key'

for target_column in target_columns:
            print(key_column,target_column)
            lead_cols = []
            lead_cols = [f"{target_column}_lead_{lead}" for lead in leads ]
            for lead, lead_col in zip(leads, lead_cols):
                promo_df_final[lead_col] = promo_df_final[[key_column,target_column]].groupby(key_column)[target_column].shift(-lead).fillna(0)
            dt = promo_df_final

# COMMAND ----------

promo_df_final.head(3)

# COMMAND ----------

df_brand = promo_df_final[promo_df_final['apply_on']=='Brand']
df_brand = df_brand[['location','promo_brand','month','schemes_lead_0', 'schemes_lead_1', 'schemes_lead_2',
       'avg_discount_amount_lead_0', 'avg_discount_amount_lead_1',
       'avg_discount_amount_lead_2', 'avg_discount_percentage_lead_0',
       'avg_discount_percentage_lead_1', 'avg_discount_percentage_lead_2',
       'max_days_of_promo_lead_0', 'max_days_of_promo_lead_1',
       'max_days_of_promo_lead_2', 'avg_days_of_promo_lead_0',
       'avg_days_of_promo_lead_1', 'avg_days_of_promo_lead_2']]

# COMMAND ----------

df_brand.head(3)

# COMMAND ----------

# df_brand = promo_df_final[promo_df_final['apply_on']=='Brand']
# df_brand = df_brand[['location','promo_brand','month','schemes_lead_0', 'schemes_lead_1', 'schemes_lead_2',
#        'avg_discount_amount_lead_0', 'avg_discount_amount_lead_1',
#        'avg_discount_amount_lead_2', 'avg_discount_percentage_lead_0',
#        'avg_discount_percentage_lead_1', 'avg_discount_percentage_lead_2',
#        'max_days_of_promo_lead_0', 'max_days_of_promo_lead_1',
#        'max_days_of_promo_lead_2', 'avg_days_of_promo_lead_0',
#        'avg_days_of_promo_lead_1', 'avg_days_of_promo_lead_2']]
print(df_brand.shape)
df_a = df_fmcg_final.merge(df_brand,how='left',left_on=['location','brand','month_start_date'],right_on=['location','promo_brand','month'])

# COMMAND ----------

pd.set_option('display.max_columns',None)

# COMMAND ----------

df_fmcg_final.shape

# COMMAND ----------

df_sub_category = promo_df_final[promo_df_final['apply_on']=='sub_cateogry']
df_b = df_fmcg_final.merge(df_sub_category,how='left',left_on=['location','item_sub_category','month_start_date'],right_on=['location','promo_item_sub_category','month'])

# COMMAND ----------

df_b.shape

# COMMAND ----------

df_b = df_b[['id','month_start_date','schemes_lead_0','schemes_lead_1','schemes_lead_2', 'avg_discount_amount_lead_0','avg_discount_amount_lead_1','avg_discount_amount_lead_2','avg_discount_percentage_lead_0','avg_discount_percentage_lead_1','avg_discount_percentage_lead_2', 'max_days_of_promo_lead_0','max_days_of_promo_lead_1','max_days_of_promo_lead_2', 'avg_days_of_promo_lead_0','avg_days_of_promo_lead_1','avg_days_of_promo_lead_2']]

# COMMAND ----------

df_item_code = promo_df_final[promo_df_final['apply_on']=='Item Code']
df_c = df_fmcg_final.merge(df_item_code,how='left',left_on=['location','item_linkage','month_start_date'],right_on=['location','item_linkage','month'])

# COMMAND ----------

df_c.shape

# COMMAND ----------

df_c = df_c[['id','month_start_date','schemes_lead_0','schemes_lead_1','schemes_lead_2', 'avg_discount_amount_lead_0','avg_discount_amount_lead_1','avg_discount_amount_lead_2','avg_discount_percentage_lead_0','avg_discount_percentage_lead_1','avg_discount_percentage_lead_2', 'max_days_of_promo_lead_0','max_days_of_promo_lead_1','max_days_of_promo_lead_2', 'avg_days_of_promo_lead_0','avg_days_of_promo_lead_1','avg_days_of_promo_lead_2']]

# COMMAND ----------

print(df_a.shape)
print(df_b.shape)
print(df_c.shape)

# COMMAND ----------

df_temp = df_a.merge(df_b,how='inner',on=['id','month_start_date'],suffixes=('_brand','_category'))

# COMMAND ----------

df_temp_t = df_temp.merge(df_c,how='inner',on=['id','month_start_date'],suffixes=('','_code'))

# COMMAND ----------

df_temp_t.shape

# COMMAND ----------

df_temp_t.drop(['month','rec_count'],axis=1,inplace=True)

# COMMAND ----------

df_temp_t.groupby(['month_start_date']).agg(sum=('stock_qty','sum')).reset_index()

# COMMAND ----------

#####.  SEASONALITY COLUMNS .######

# COMMAND ----------

df_temp_t.rename(columns = {"item_linkage":"Item_linkage"},inplace=True)
df_temp_t['Item_linkage']

# COMMAND ----------

er_variant = """ 
with ilv_unique_linkage AS (
SELECT  Item_linkage
       ,MAX(COALESCE(Item_Classification,Item_Sub_Category)) ER_Variant
       ,AVG(ti.shelf_life_in_days) shelf_life_in_days
       ,MAX(ti.Brand) as Brand
       ,MAX(COALESCE(Brand_type, "NA")) as Brand_Type
       ,MAX(Item_Sub_Category) as Item_Sub_Category 
FROM staging.tabitem ti 
LEFT JOIN staging.tabbrand br 
ON ti.brand = br.brand 
WHERE lower(ti.item_code) not like "%_discount%"
AND lower(ti.item_code) not like "%_supersaver%"
GROUP BY  item_linkage 
)
select * from ilv_unique_linkage
"""


df_er_variant = spark.sql(er_variant)
df_er_variant_pd = df_er_variant.toPandas()
df_ER_variant1 = df_er_variant_pd[df_er_variant.columns[0:2]]
df_ER_variant1.head()





# COMMAND ----------

df_er_variant1 = pd.merge(df_temp_t,df_ER_variant1,on="Item_linkage",how="left")

# COMMAND ----------

df_er_variant1['ER_Variant'] = np.where(df_er_variant1['ER_Variant'].isna(), df_er_variant1['item_sub_category'], df_er_variant1['ER_Variant'])

# COMMAND ----------

df_er_variant2 = df_er_variant1.dropna(subset=['item_sub_category'])

# COMMAND ----------

df_er_variant2.head()

# COMMAND ----------

threshold = 0.5

# COMMAND ----------

seasonality = f"""
 with ilv_sales as (
 SELECT  fsii.parent
        ,dwar.state 
        ,dwar.city
        ,dwar.name as Warehouse 
        ,ditem.item_code 
        ,ditem.derived_er_item_type
        -- ,ditem.brand 
        ,ddate.date shippingstatusdate
        ,ti.item_sub_category as Item_sub_category 
        ,IFNULL(if(ti.Item_Classification='',null,ti.Item_Classification) , ti.item_sub_category) as ER_Variant
	    -- ,ditem.item_linkage 
	    ,coalesce(npl.planning_linkage,ditem.item_linkage) as item_linkage
        -- ,IFNULL(wp.replenishment_center, dwar.name) as CRC  
        ,wp.replenishment_center as CRC 
        ,fsii.rate  
        ,fsii.amount
        ,fsii.amount - IF(dwar.name like 'MD%%' , fsii.sle_valuation_amount /1.01 , fsii.sle_valuation_amount ) as margin 
        ,fsii.amount_with_tax  
        -- ,ditem.item_linkage
        ,fsii.amount_inclusive_all_taxes
        ,dcu.name as customer
        ,fsii.name  as sii
        , coalesce(doa_gt_1000,0) doa_gt_1000
        ,coalesce(30_day_sales,0)  30_day_sales
      , coalesce(30_day_average_stock) 30_day_average_stock
      , coalesce(current_stock) current_stock
      ,CAST(snop.m_0_target AS DECIMAL(18,2)) * 1.1 as Target
       ,snop.target_basis snop_target_basis
       -- hard-coding to 1 to easily swap between use stock_factor
       -- ,IF(doa_gt_1000 = 0 , 1 , 30/doa_gt_1000) as stock_factor 
       ,1 as stock_factor
     , ddate.month_name
     , ddate.year
     , ddate.date
FROM core.`factsalesinvoiceitem` fsii

INNER JOIN core.`dimdate` ddate
ON fsii.`shippingstatusdatekey` = ddate.`dimkey`

INNER JOIN core.`dimitem`  ditem
ON fsii.`itemkey` = ditem.`dimkey`

INNER JOIN staging.tabitem ti 
ON ti.item_code = ditem.item_code

INNER JOIN core.`dimwarehouse` dwar
ON fsii.warehousekey = dwar.dimkey

INNER JOIN core.`dimcustomer`  dcu
ON fsii.`customerkey` = dcu.`dimkey`

LEFT JOIN staging.dwwarehouseprofile wp 
ON wp.warehouse = dwar.name 
AND wp.active = 1 

LEFT JOIN dsapps_staging.dwniyojanplanninglinkage npl
ON ditem.item_linkage = npl.item_linkage
AND npl.active=1

 LEFT JOIN dsapps_core.snopplanningmaster snop
ON snapshotdate = (select max(snapshotdate) from dsapps_core.snopplanningmaster)
and snop.planning_linkage = npl.item_linkage 
and snop.warehouse =dwar.name
and snop.is_crc != 1

WHERE fsii.si_shipping_status = 'Delivered'
AND dcu.name <> 'ElasticRun Distributor'
and dwar.name not like "%MDT4%"
and upper(dwar.name) not like '%URBAN%'
and upper(dwar.name) not like '%KIRUN%'
and upper(dwar.name) not like '%DAMAGES%'
and upper(dwar.name) not like '%DISCARDED%'


AND ddate.date Between CURDATE() - Interval 730 DAY AND CURDATE()
AND nullif(ditem.derived_er_item_type,'') IS NOT NULL
AND nullif(dwar.md_warehouse,'') IS NULL

-- AND ditem.derived_er_item_type <> 'General Merchandise'
-- AND ditem.derived_er_item_type <> 'Oil'
-- AND ditem.derived_er_item_type <> 'Sugar'
AND lower(ditem.item_code) not like '%supersaver%'
AND ditem.item_code not like  '%_discount%'
AND wp.replenishment_center is not null 
),

ilv_grp1 as (

select
    state, 
    -- city, 
    CRC, 
    Item_sub_category, 
    ER_Variant, 
    -- item_linkage,
    SUM(stock_factor * amount) as total_sales,
    SUM(CASE WHEN month_name = 'January' THEN stock_factor * amount ELSE 0 END) as january_sales,
    SUM(CASE WHEN month_name = 'February' THEN stock_factor * amount ELSE 0 END) as february_sales,
    SUM(CASE WHEN month_name = 'March' THEN stock_factor * amount ELSE 0 END) as march_sales,
    SUM(CASE WHEN month_name = 'April' THEN stock_factor * amount ELSE 0 END) as april_sales,
    SUM(CASE WHEN month_name = 'May' THEN stock_factor * amount ELSE 0 END) as may_sales,
    SUM(CASE WHEN month_name = 'June' THEN stock_factor * amount ELSE 0 END) as june_sales,
    SUM(CASE WHEN month_name = 'July' THEN stock_factor * amount ELSE 0 END) as july_sales,
    SUM(CASE WHEN month_name = 'August' THEN stock_factor * amount ELSE 0 END) as august_sales,
    SUM(CASE WHEN month_name = 'September' THEN stock_factor * amount ELSE 0 END) as september_sales,
    SUM(CASE WHEN month_name = 'October' THEN stock_factor * amount ELSE 0 END) as october_sales,
    SUM(CASE WHEN month_name = 'November' THEN stock_factor * amount ELSE 0 END) as november_sales,
    SUM(CASE WHEN month_name = 'December' THEN stock_factor * amount ELSE 0 END) as december_sales,
    min(date) as introduction_date
from ilv_sales
group by state,  CRC, Item_sub_category , ER_Variant
-- , item_linkage
-- city,
),

ilv_grp2 as (

select
    ilv_grp1.*,
    count(distinct case when dimdate.date >= ilv_grp1.introduction_date THEN concat(dimdate.month, dimdate.year) else null end) as months_seen,
    count(distinct case when dimdate.date >= ilv_grp1.introduction_date and dimdate.month_name = 'January' THEN concat(dimdate.month, dimdate.year) else null end) as january_seen,
    count(distinct case when dimdate.date >= ilv_grp1.introduction_date and dimdate.month_name = 'February' THEN concat(dimdate.month, dimdate.year) else null end) as february_seen,
    count(distinct case when dimdate.date >= ilv_grp1.introduction_date and dimdate.month_name = 'March' THEN concat(dimdate.month, dimdate.year) else null end) as march_seen,
    count(distinct case when dimdate.date >= ilv_grp1.introduction_date and dimdate.month_name = 'April' THEN concat(dimdate.month, dimdate.year) else null end) as april_seen,
    count(distinct case when dimdate.date >= ilv_grp1.introduction_date and dimdate.month_name = 'May' THEN concat(dimdate.month, dimdate.year) else null end) as may_seen,
    count(distinct case when dimdate.date >= ilv_grp1.introduction_date and dimdate.month_name = 'June' THEN concat(dimdate.month, dimdate.year) else null end) as june_seen,
    count(distinct case when dimdate.date >= ilv_grp1.introduction_date and dimdate.month_name = 'July' THEN concat(dimdate.month, dimdate.year) else null end) as july_seen,
    count(distinct case when dimdate.date >= ilv_grp1.introduction_date and dimdate.month_name = 'August' THEN concat(dimdate.month, dimdate.year) else null end) as august_seen,
    count(distinct case when dimdate.date >= ilv_grp1.introduction_date and dimdate.month_name = 'September' THEN concat(dimdate.month, dimdate.year) else null end) as september_seen,
    count(distinct case when dimdate.date >= ilv_grp1.introduction_date and dimdate.month_name = 'October' THEN concat(dimdate.month, dimdate.year) else null end) as october_seen,
    count(distinct case when dimdate.date >= ilv_grp1.introduction_date and dimdate.month_name = 'November' THEN concat(dimdate.month, dimdate.year) else null end) as november_seen,
    count(distinct case when dimdate.date >= ilv_grp1.introduction_date and dimdate.month_name = 'December' THEN concat(dimdate.month, dimdate.year) else null end) as december_seen
from ilv_grp1
cross join core.dimdate
where dimdate.date > (select min(introduction_date) from ilv_grp1)
and dimdate.date <= CURDATE()
group by state,  CRC, Item_sub_category,ER_Variant, total_sales, january_sales, february_sales, march_sales, april_sales, may_sales, june_sales, july_sales, august_sales,
    september_sales, october_sales, november_sales, december_sales, introduction_date
-- city,
--   item_linkage,
),

ilv_der1 as (

select
    ilv_grp2.*,
    case when MONTH(introduction_date) = 1 THEN january_sales + (DAY(introduction_date) - 1) * january_sales / 31 * january_seen ELSE january_sales END as adjusted_january_sales,
    case when MONTH(introduction_date) = 2 THEN february_sales + (DAY(introduction_date) - 1) * february_sales / 28 * february_seen ELSE february_sales END as adjusted_february_sales,
    case when MONTH(introduction_date) = 3 THEN march_sales + (DAY(introduction_date) - 1) * march_sales / 31 * march_seen ELSE march_sales END as adjusted_march_sales,
    case when MONTH(introduction_date) = 4 THEN april_sales + (DAY(introduction_date) - 1) * april_sales / 30 * april_seen ELSE april_sales END as adjusted_april_sales,
    case when MONTH(introduction_date) = 5 THEN may_sales + (DAY(introduction_date) - 1) * may_sales / 31 * may_seen ELSE may_sales END as adjusted_may_sales,
    case when MONTH(introduction_date) = 6 THEN june_sales + (DAY(introduction_date) - 1) * june_sales / 30 * june_seen ELSE june_sales END as adjusted_june_sales,
    case when MONTH(introduction_date) = 7 THEN july_sales + (DAY(introduction_date) - 1) * july_sales / 31 * july_seen ELSE july_sales END as adjusted_july_sales,
    case when MONTH(introduction_date) = 8 THEN august_sales + (DAY(introduction_date) - 1) * august_sales / 31 * august_seen ELSE august_sales END as adjusted_august_sales,
    case when MONTH(introduction_date) = 9 THEN september_sales + (DAY(introduction_date) - 1) * september_sales / 30 * september_seen ELSE september_sales END as adjusted_september_sales,
    case when MONTH(introduction_date) = 10 THEN october_sales + (DAY(introduction_date) - 1) * october_sales / 31 * october_seen ELSE october_sales END as adjusted_october_sales,
    case when MONTH(introduction_date) = 11 THEN november_sales + (DAY(introduction_date) - 1) * november_sales / 30 * november_seen ELSE november_sales END as adjusted_november_sales,
    case when MONTH(introduction_date) = 12 THEN december_sales + (DAY(introduction_date) - 1) * december_sales / 31 * december_seen ELSE december_sales END as adjusted_december_sales,
    IF(months_seen = 0, 0, total_sales / months_seen) as monthly_average
from ilv_grp2
),

ilv_der2 as (

select
    ilv_der1.*,
    IF(january_seen = 0, 0, adjusted_january_sales / january_seen) as january_average,
    IF(february_seen = 0, 0, adjusted_february_sales / february_seen) as february_average,
    IF(march_seen = 0, 0, adjusted_march_sales / march_seen) as march_average,
    IF(april_seen = 0, 0, adjusted_april_sales / april_seen) as april_average,
    IF(may_seen = 0, 0, adjusted_may_sales / may_seen) as may_average,
    IF(june_seen = 0, 0, adjusted_june_sales / june_seen) as june_average,
    IF(july_seen = 0, 0, adjusted_july_sales / july_seen) as july_average,
    IF(august_seen = 0, 0, adjusted_august_sales / august_seen) as august_average,
    IF(september_seen = 0, 0, adjusted_september_sales / september_seen) as september_average,
    IF(october_seen = 0, 0, adjusted_october_sales / october_seen) as october_average,
    IF(november_seen = 0, 0, adjusted_november_sales / november_seen) as november_average,
    IF(december_seen = 0, 0, adjusted_december_sales / december_seen) as december_average
from ilv_der1
),

ilv_der3 as (

select 
    ilv_der2.*,
    IF(monthly_average = 0, 0, january_average / monthly_average) as january_seasonality_ratio,
    IF(monthly_average = 0, 0, february_average / monthly_average) as february_seasonality_ratio,
    IF(monthly_average = 0, 0, march_average / monthly_average) as march_seasonality_ratio,
    IF(monthly_average = 0, 0, april_average / monthly_average) as april_seasonality_ratio,
    IF(monthly_average = 0, 0, may_average / monthly_average) as may_seasonality_ratio,
    IF(monthly_average = 0, 0, june_average / monthly_average) as june_seasonality_ratio,
    IF(monthly_average = 0, 0, july_average / monthly_average) as july_seasonality_ratio,
    IF(monthly_average = 0, 0, august_average / monthly_average) as august_seasonality_ratio,
    IF(monthly_average = 0, 0, september_average / monthly_average) as september_seasonality_ratio,
    IF(monthly_average = 0, 0, october_average / monthly_average) as october_seasonality_ratio,
    IF(monthly_average = 0, 0, november_average / monthly_average) as november_seasonality_ratio,
    IF(monthly_average = 0, 0, december_average / monthly_average) as december_seasonality_ratio
from ilv_der2

),

ilv_final as (
select
    ilv_der3.*,
    CASE
        WHEN january_seen = 0 THEN NULL
        WHEN january_seasonality_ratio < 1 - {threshold} THEN 'Out of Season'
        WHEN january_seasonality_ratio > 1 + {threshold} THEN 'In Season'
        ELSE 'Normal'
    END AS january_seasonality_status,
    
    CASE
        WHEN february_seen = 0 THEN NULL
        WHEN february_seasonality_ratio < 1 - {threshold} THEN 'Out of Season'
        WHEN february_seasonality_ratio > 1 + {threshold} THEN 'In Season'
        ELSE 'Normal'
    END AS february_seasonality_status,
    
    CASE
        WHEN march_seen = 0 THEN NULL
        WHEN march_seasonality_ratio < 1 - {threshold} THEN 'Out of Season'
        WHEN march_seasonality_ratio > 1 + {threshold} THEN 'In Season'
        ELSE 'Normal'
    END AS march_seasonality_status,
    
    CASE
        WHEN april_seen = 0 THEN NULL
        WHEN april_seasonality_ratio < 1 - {threshold} THEN 'Out of Season'
        WHEN april_seasonality_ratio > 1 + {threshold} THEN 'In Season'
        ELSE 'Normal'
    END AS april_seasonality_status,
    
    CASE
        WHEN may_seen = 0 THEN NULL
        WHEN may_seasonality_ratio < 1 - {threshold} THEN 'Out of Season'
        WHEN may_seasonality_ratio > 1 + {threshold} THEN 'In Season'
        ELSE 'Normal'
    END AS may_seasonality_status,
    
    CASE
        WHEN june_seen = 0 THEN NULL
        WHEN june_seasonality_ratio < 1 - {threshold} THEN 'Out of Season'
        WHEN june_seasonality_ratio > 1 + {threshold} THEN 'In Season'
        ELSE 'Normal'
    END AS june_seasonality_status,
    
    CASE
        WHEN july_seen = 0 THEN NULL
        WHEN july_seasonality_ratio < 1 - {threshold} THEN 'Out of Season'
        WHEN july_seasonality_ratio > 1 + {threshold} THEN 'In Season'
        ELSE 'Normal'
    END AS july_seasonality_status,
    
    CASE
        WHEN august_seen = 0 THEN NULL
        WHEN august_seasonality_ratio < 1 - {threshold} THEN 'Out of Season'
        WHEN august_seasonality_ratio > 1 + {threshold} THEN 'In Season'
        ELSE 'Normal'
    END AS august_seasonality_status,
    
    CASE
        WHEN september_seen = 0 THEN NULL
        WHEN september_seasonality_ratio < 1 - {threshold} THEN 'Out of Season'
        WHEN september_seasonality_ratio > 1 + {threshold} THEN 'In Season'
        ELSE 'Normal'
    END AS september_seasonality_status,
    
    CASE
        WHEN october_seen = 0 THEN NULL
        WHEN october_seasonality_ratio < 1 - {threshold} THEN 'Out of Season'
        WHEN october_seasonality_ratio > 1 + {threshold} THEN 'In Season'
        ELSE 'Normal'
    END AS october_seasonality_status,
    
    CASE
        WHEN november_seen = 0 THEN NULL
        WHEN november_seasonality_ratio < 1 - {threshold} THEN 'Out of Season'
        WHEN november_seasonality_ratio > 1 + {threshold} THEN 'In Season'
        ELSE 'Normal'
    END AS november_seasonality_status,
    
    CASE
        WHEN december_seen = 0 THEN NULL
        WHEN december_seasonality_ratio < 1 - {threshold} THEN 'Out of Season'
        WHEN december_seasonality_ratio > 1 + {threshold} THEN 'In Season'
        ELSE 'Normal'
    END AS december_seasonality_status
  
  
fROM ilv_der3
where 1 = 1 
-- [[AND CRC in (Select name from dimwarehouse where {{CRC}})]]
-- and Item_sub_category = 'Cold Drinks'
) 
select 
-- state,
CRC,
Item_sub_category,
ER_Variant,
january_seasonality_status,
february_seasonality_status,	
march_seasonality_status,
april_seasonality_status,
may_seasonality_status,
june_seasonality_status,
july_seasonality_status,
august_seasonality_status,
september_seasonality_status,
october_seasonality_status,
november_seasonality_status,
december_seasonality_status
from ilv_final
"""

df_seasonality = spark.sql(seasonality)

# COMMAND ----------

df_seasonality_pd = df_seasonality.toPandas() 

# COMMAND ----------

# df_seasonality_pd

# COMMAND ----------

final_training_data =pd.merge(df_er_variant2, df_seasonality_pd, how='left', left_on=['ER_Variant','item_sub_category','replenishment_center'],right_on=['ER_Variant','Item_sub_category','CRC'])



# COMMAND ----------

# final_training_data.head()

# COMMAND ----------

params = {'january_seasonality_status', 'february_seasonality_status', 'march_seasonality_status', 'april_seasonality_status', 'may_seasonality_status', 'june_seasonality_status', 'july_seasonality_status', 'august_seasonality_status', 'september_seasonality_status', 'october_seasonality_status', 'november_seasonality_status', 'december_seasonality_status'}

# COMMAND ----------

def create_category_encoder_features(data,config):
        print("creating category encoder features")
        print(config)
        for cc in config:
            le = LabelEncoder()
            data[f'{cc}_cat'] = data[cc].copy()
            data[f'{cc}_cat'] = le.fit_transform(data[f'{cc}_cat'])
        return data

# COMMAND ----------

final_training_data = create_category_encoder_features(final_training_data,params)

# COMMAND ----------

final_training_data = final_training_data.drop(['Item_sub_category'],axis=1)

# COMMAND ----------

import json
path = "/Workspace/Users/nishad.chaoji@elasticrun.com/Anumaan/ma_model_config.json"
with open(path,"r") as f:
    config = json.loads(f.read())

file_name = config["input_data"]["input_data_file"]["input_file_name"]
file_path = config["input_data"]["input_data_file"]["path"]
print(f"The folder name is {file_path}/{file_name}.csv")
final_training_data.to_csv(f"{file_path}/{file_name}.csv")


# COMMAND ----------

# spark_df= spark.createDataFrame(final_training_data)
# spark_df.write.mode("overwrite").saveAsTable(f"dsapps_staging.anumaantraindata")
import gc
df_fmcg.unpersist()
df_exp.unpersist()
df_hier.unpersist()
df_seasonality.unpersist()
del df_ER_variant1
del df_a
del df_b
del df_brand
del df_c
del df_er_variant
del df_er_variant1
del df_er_variant2
del df_er_variant_pd
del df_exp
del df_exp_pd
del df_fmcg
del df_fmcg_final
del df_fmcg_pd
del df_hier
del df_hier_pd
del df_il
del df_il_pd
del df_item
del df_item_code
del df_seasonality
del df_seasonality_pd
del df_sub_category
del df_temp
del df_temp_t
del df_with_item
del df_with_item_fmcg
gc.collect()

# COMMAND ----------

