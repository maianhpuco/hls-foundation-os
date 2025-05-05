sanity_check:
	mim train mmseg configs/sen1floods11_config_prompt_tuning.py

infer:
	mim test mmseg configs/sen1floods11_config_prompt_tuning.py \
		--checkpoint maianh_exp_01/epoch_2.pth \
		--eval mIoU \
		--launcher none

infer_08: 
	mim test mmseg configs/sen1floods11_config_prompt_tuning_08.py \
		--checkpoint nprompt_08/latest.pth \
		--eval mIoU mDice \
		--launcher none  >> infer_08.txt 

infer_16: 
	mim test mmseg configs/sen1floods11_config_prompt_tuning_16.py \
		--checkpoint nprompt_16/latest.pth \
		--eval mIoU mDice \
		--launcher none >> infer_16.txt  

infer_32: 
	mim test mmseg configs/sen1floods11_config_prompt_tuning_32.py \
		--checkpoint nprompt_32/latest.pth \
		--eval mIoU mDice \
		--launcher none \  

infer_32_10: 
	mim test mmseg configs/sen1floods11_config_prompt_tuning_32_train_10.py \
		--checkpoint nprompt_32_10/latest.pth \
		--eval mIoU mDice \
		--launcher none \   

infer_32_25: 
	mim test mmseg configs/sen1floods11_config_prompt_tuning_32_train_25.py \
		--checkpoint nprompt_32_25/latest.pth \
		--eval mIoU mDice \
		--launcher none \   

infer_32_50: 
	mim test mmseg configs/sen1floods11_config_prompt_tuning_32_train_50.py \
		--checkpoint nprompt_32_50/latest.pth \
		--eval mIoU mDice \
		--launcher none \     

infer_32_75: 
	mim test mmseg configs/sen1floods11_config_prompt_tuning_32_train_75.py \
		--checkpoint nprompt_32_75/latest.pth \
		--eval mIoU mDice \
		--launcher none \     

infer_10: 
	mim test mmseg configs/sen1flood11_config_10.py \
		--checkpoint full_10/latest.pth \
		--eval mIoU mDice \
		--launcher none \    

infer_10: 
	mim test mmseg configs/sen1flood11_config_10.py \
		--checkpoint full_10/latest.pth \
		--eval mIoU mDice \
		--launcher none \
