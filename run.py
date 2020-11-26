import os
import sys
sys.path.append('./src')
os.chdir(sys.path[0])

import output_image



arguments = list(sys.argv)

target = arguments[1]

# open notebook for complete model run
if target == 'run-notebook':
    os.system('echo launching notebook ...')
    os.system('jupyter notebook ./notebooks/run_notebook.ipynb')

# when file_path to model output is provided, generate analysis graph ala notebook
if target == 'analyze-results':
    filepath = arguments[2]
    output_image.analyze_model_result(model_out_path=filepath)
    
if target == 'test':
    os.system('python test.py')


    