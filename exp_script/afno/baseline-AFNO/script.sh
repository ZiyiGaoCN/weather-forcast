deepspeed --include=localhost:1 --master_port=29511 \
    fourcastnet_deepspeed.py --cfg_path ./cfgs/finetune/finetune_afno.yaml\
    --cfg_new ./exp_script/afno/baseline-AFNO/finetune_afno_new.yaml
    