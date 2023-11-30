from traintools import *
from utils.args import *
import warnings
warnings.filterwarnings("ignore")

def main(parse, config: ConfigParser):
    
    if parse.traintools == 'train_synthesized':
        train_synthesized(parse, config)
    elif parse.traintools == 'train_realworld':
        train_realworld(parse, config)
    else:
        raise NotImplemented
        
if __name__ == '__main__':
    config, parse = parse_args()

    ### TRAINING ###
    main(parse, config)
