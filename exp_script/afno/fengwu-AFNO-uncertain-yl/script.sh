deepspeed --include=localhost:6 --master_port=29567 \
    fengwu_deepspeed.py --cfg_path ./cfgs/finetune/finetune_fengwu.yaml\
    --cfg_new ./exp_script/afno/fengwu-AFNO-uncertain-yl/finetune_afno_new.yaml
    