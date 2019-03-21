import sys
path =  '/home/erobb/NAS-CloserLookFewShot'
sys.path.insert(0, path)

import settings

usage = "Usage: python main.py <modelfile> <architecture> <genotype> <num_layers> <cell#>\ne.g.: python main.py evolved/best_model.tar genotype EvoResNetIsh1S10E10P573 4 3"
if len(sys.argv) != 6:
    print(usage)
    quit()

settings.MODEL_FILE = sys.argv[1]
settings.MODEL = sys.argv[2]
settings.GENE_NAME = sys.argv[3]
settings.LAYERS = int(sys.argv[4])
settings.LAYER = int(sys.argv[5])
settings.init(settings.MODEL)
print(settings.OUTPUT_FOLDER)

from loader.model_loader import loadmodel
from feature_operation import hook_feature,FeatureOperator
from visualize.report import generate_html_summary
from util.clean import clean


fo = FeatureOperator()
model = loadmodel(hook_feature)

############ STEP 1: feature extraction ###############
features, maxfeature = fo.feature_extraction(model=model)

for layer_id,layer in enumerate(settings.FEATURE_NAMES):
############ STEP 2: calculating threshold ############
    thresholds = fo.quantile_threshold(features[layer_id],savepath="quantile.npy")

############ STEP 3: calculating IoU scores ###########
    tally_result = fo.tally(features[layer_id],thresholds,savepath="tally.csv")

############ STEP 4: generating results ###############
    generate_html_summary(fo.data, layer,
                          tally_result=tally_result,
                          maxfeature=maxfeature[layer_id],
                          features=features[layer_id],
                          thresholds=thresholds)
    if settings.CLEAN:
        clean()
