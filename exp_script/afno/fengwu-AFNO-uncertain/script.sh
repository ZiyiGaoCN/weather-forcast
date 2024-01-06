deepspeed --include=localhost:6 --master_port=29566 \
    fengwu_deepspeed.py --cfg_path ./cfgs/finetune/finetune_fengwu.yaml\
    --cfg_new ./exp_script/afno/fengwu-AFNO-uncertain/finetune_afno_new.yaml
    