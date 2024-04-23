# Gray 50 GTCNND1
# python main.py confs/GTCNN experiment.color=1 experiment.test_only=True experiment.best_model=checkpoint_RetrainGTCNND1_545 dataset.test_set=[Set12,Set68,Urban100] experiment.name=16pad_GTCNND1_gray50 experiment.device_ids=[2]
# # Gray 50 GTCNND3
# python main.py confs/GTCNN experiment.color=1 model.depth=3 experiment.test_only=True experiment.best_model=Gray50_GTCNN_D3 dataset.test_set=[Set12,Set68,Urban100] experiment.name=16pad_GTCNND3_gray50
# # Gray 50 GTCNND6
# python main_trace.py confs/GTCNN experiment.best_model=checkpoint_RetrainGTCNND6 experiment.color=1 model.depth=6 model.GTL_stage_option=outconv_slim experiment.test_only=True dataset.test_set=[Set12,Set68,Urban100] experiment.name=retrain_GTCNND6_gray50
python main_trace.py confs/GTCNN experiment.color=1 model.depth=3 experiment.test_only=True experiment.best_model=checkpoint_RetrainGTCNND3_restart150_285 dataset.test_set=[Set12,Set68,Urban100] experiment.name=16pad_GTCNND3_gray50 
