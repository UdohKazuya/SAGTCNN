from models import GTCNN
from omegaconf import OmegaConf
import omegaconf
from torchinfo import summary

cli_conf = {
    'experiment':{
        'color':1,
    },
    'model':{
        'depth':1,
    },
}

conf = OmegaConf.load('confs/GTCNN.yaml')
OmegaConf.set_struct(conf, True)
conf = OmegaConf.merge(conf, cli_conf)

model = GTCNN(conf.model)
summary(model)
#print(model)
