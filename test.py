import sys
import os
sys.path.append('./src')
os.chdir(sys.path[0])



# data Analysis 
import geopandas as gpd
import pandas as pd
import numpy as np



# plot
import matplotlib.pyplot as plt
import gc

# output
from output_image import write_output, analyze_model_result, output_to_csv


# multiprocessing
import multiprocessing
import brun


# model
from school_model import School


#config
import configparser
import warnings
warnings.filterwarnings("ignore")


map_path = "./test/testdata/school.shp"
schedule_path = "./test/testdata/small_schedule.csv" 
schedule_steps = 5 # full day_schedule steps should be 90

# two types of parameter setups available for batchrunner
# pre-setup for fixed/variable parameter dictionaries (consistant with mesa batchrunner)
######################
grade_N = 20
KG_N = 20
preschool_N = 20
special_education_N = 10
faculty_N = 10
seat_dist = 12
mask_prob = 0.516
days = 5
max_steps = days*schedule_steps
iterations = 1

school = School(map_path, schedule_path, grade_N, KG_N, preschool_N, special_education_N, 
                 faculty_N, seat_dist, init_patient=3, attend_rate=1, mask_prob=0.5, inclass_lunch=False, username="xyzabc")


while school.running and school.schedule.steps < 1:
    school.step()

params = "{'test': 0}" 
agent_df = school.datacollector.get_agent_vars_dataframe()
model_df = school.datacollector.get_model_vars_dataframe()
model_df.to_csv('./test/output/model_df.csv', index = False)
agent_df.to_csv('./test/output/agent_df.csv', index = False)

