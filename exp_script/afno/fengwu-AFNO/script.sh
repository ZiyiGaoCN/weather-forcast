deepspeed --include=localhost:5 --master_port=29555 \
    fengwu_deepspeed.py --cfg_path ./cfgs/finetune/finetune_fengwu.yaml\
    --cfg_new ./exp_script/afno/fengwu-AFNO/finetune_afno_new.yaml
    