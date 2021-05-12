import sys
import os

sys.path.append('./src')
os.chdir(sys.path[0])



# data Analysis 
import geopandas as gpd

# plot
import matplotlib.pyplot as plt
import seaborn as sns



# model
from school_model import School


#config
import configparser
import warnings
import transmission_rate as trans_rate
warnings.filterwarnings("ignore")
import os



map_path = "/home/geoact/schoollayout1/schoollayout1_processed.shp"#"/Users/kaushikramganapathy/Downloads/temp_project/layouts/schoollayout1/schoollayout1_processed.shp" #"/home/geoact/layouts/schoollayout1.shp" #"./test/testdata/schoollayout1.shp"
schedule_path = "/home/geoact/schedule_data/day_schedule.csv"#"/Users/kaushikramganapathy/testschoolmodel/0527291751/schedule_data/day_schedule.csv" 
schedule_steps = 90 # full day_schedule steps should be 90

# Trying to get this out of config files
parser_schoolparams = configparser.ConfigParser()
parser_schoolparams.read('./config/schoolparams.ini')
school_population = parser_schoolparams['SCHOOL_POPULATION']
interventions = parser_schoolparams['INTERVENTION']
time = parser_schoolparams['TIME']


# User Input Parameters
######################
grade_N = int(school_population['grade_N'])
KG_N = int(school_population['KG_N'])
preschool_N = int(school_population['preschool_N'])

special_education_N = int(school_population['special_education_N'])
faculty_N = int(school_population['faculty_N'])

mask_prob = float(interventions['mask_prob'])/100

days = int(time['days'])

attend_rate = float(interventions['attend_rate'])/100
inclass_lunch = eval(interventions['inclass_lunch'])

student_vaccine_prob = float(interventions['student_vaccine_prob'])/100

teacher_vaccine_prob = float(interventions['teacher_vaccine_prob'])/100

student_testing_freq = int(interventions['student_testing_freq'])
teacher_testing_freq = int(interventions['teacher_testing_freq'])
teacher_mask = interventions['teacher_mask']
factor = attend_rate * (grade_N + KG_N + preschool_N + special_education_N)
#####################

seat_dist = 5



max_steps = days*schedule_steps

school = School(map_path, schedule_path, grade_N, KG_N, preschool_N, special_education_N, 
                 faculty_N, seat_dist, init_patient=3, attend_rate=attend_rate, mask_prob=mask_prob, inclass_lunch=inclass_lunch, student_vaccine_prob=student_vaccine_prob,
                 teacher_vaccine_prob=teacher_vaccine_prob, student_testing_freq=student_testing_freq, teacher_testing_freq=teacher_testing_freq, teacher_mask=teacher_mask)


while school.running and school.schedule.steps < max_steps:
    school.step()

params = "{'test': 0}" 
agent_df = school.datacollector.get_agent_vars_dataframe()
model_df = school.datacollector.get_model_vars_dataframe()
model_df.to_csv('./test/output/model_df.csv', index = False)
agent_df.to_csv('./test/output/agent_df.csv', index = False)


plt.plot(model_df['day'].values, model_df['cov_positive'].values/factor, 'r*-')
plt.title(f'Proportion Infected After {days} Days')
plt.xlabel('Day')
plt.ylabel('Proportion of Infected')
plt.savefig(f'./test/output/lineplot.png', dpi=200)
plt.clf()

# We add this here because our analysis.py worklow cannot be used with simplistic datasets
# As this is not the intended usecase for the codebase. 

dataframe_exposed_loc =agent_df
dataframe_exposed_loc = dataframe_exposed_loc[dataframe_exposed_loc['health_status'] == 'exposed']
exposed_locations = dataframe_exposed_loc.groupby(["unique_id"])["x", "y"].first()
school_geometry = gpd.read_file(map_path)
      
ax = school_geometry.plot(color="white", edgecolor='black')
sns.kdeplot(exposed_locations["x"], exposed_locations["y"], ax=ax, fill=True, cmap='YlOrBr', alpha=0.45)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_title(f'Heatmap of Exposed Locations', fontdict={'size':20})
plt.savefig(f'./test/output/heatmap.png', dpi=400)