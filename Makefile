sanity_check:
	mim train mmseg configs/sen1floods11_config_prompt_tuning.py

sanity_check_infer:
	mim test mmseg configs/sen1floods11_config_prompt_tuning.py \
		--checkpoint  maiannh_exp_01/epoch_2.pth\
		--eval mIoU \
		--show-dir work_dirs/sen1floods11_config_prompt_tuning/inference
