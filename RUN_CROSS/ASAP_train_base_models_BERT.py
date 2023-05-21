import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from learning_curve.train_asap_base_models import train_base_models_bert

train_base_models_bert()