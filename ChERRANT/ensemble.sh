#single model post edit
python rule_ensemble.py\
--result_path /home/predict/Stage2_FAM_minimal.tgtunk.m2\
--output_path /home/predict/Stage2_FAM_minimal.srctgt.m2\
--threshold 1

#three models ensemble and post edit
python rule_ensemble.py\
--result_path /home/predict/Stage2_FAM_minimal.tgtunk.m2\
/home/predict/Stage2_Min_minimal.tgtunk.m2\
/home/predict/Stage2_Flu_minimal.tgtunk.m2\
--output_path /home/predict/Stage2_FAM_minimal.srctgt.m2\
--threshold 3