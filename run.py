import os
import vvn.prep as vp

# define global variables
PATH_TO_DATA = os.path.join(os.getcwd(), 'data', 'ZoomOut', 'test')
PATH_TO_MODELS = os.path.join(os.getcwd(), 'models')

results = vp.get_correct_samples(PATH_TO_MODELS, PATH_TO_DATA)
print(results)
