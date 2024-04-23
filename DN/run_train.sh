

# Gray 50 GTCNND1
python main.py confs/GTCNN experiment.color=1 dataset.test_set=[Set12,BSD68,Urban100] experiment.device_ids=[1] 
# Gray 50 GTCNND3
python main.py confs/GTCNN experiment.color=1 model.depth=3 experiment.batchsize=30 dataset.test_set=[Set12,BSD68,Urban100]
python main.py confs/GTCNN experiment.comment=GTCNND3_sofmtax  dataset.test_set=[Set12,Set68,Urban100] experiment.color=1  experiment.name=GTCNND3_sofmtax experiment.device_ids=[1,2] model.depth=3 experiment.imgSize=192 model.name=GTCNN experiment.save_checkpoint=True experiment.batchsize=30  experiment.opt_level=O1 model.use_bnorm=True experiment.epoch_num=200
# Gray 50 GTCNND6
python main.py confs/GTCNN experiment.color=1 model.depth=6 experiment.batchsize=12 model.GTL_stage_option=outconv_slim  dataset.test_set=[Set12,BSD68,Urban100]  experiment.device_ids=[1] 


###########
# GTCNN-D1
###########
# Gray 30
python main.py confs/GTCNN experiment.sigma=30 experiment.color=1  dataset.test_set=[Set12,BSD68,Urban100]
# Gray 70
python main.py confs/GTCNN experiment.sigma=70 experiment.color=1  dataset.test_set=[Set12,BSD68,Urban100]

# Color 50 
python main.py confs/GTCNN   
# Color 30
python main.py confs/GTCNN experiment.sigma=30   
# Color 70
python main.py confs/GTCNN experiment.sigma=70  