sanity_check:
	mim train mmseg configs/sen1floods11_config_prompt_tuning.py

infer:
	mim test mmseg configs/sen1floods11_config_prompt_tuning.py \
		--checkpoint maianh_exp_01/epoch_2.pth \
		--eval mIoU mDice \
		--launcher none \ 
		--show-dir work_dirs/sen1floods11_config_prompt_tuning/inference \
		--options show_dir=work_dirs/sen1floods11_config_prompt_tuning/inference
