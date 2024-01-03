deepspeed --include=localhost:7 --master_port=29577 \
    fourcastnet_deepspeed.py --cfg_path ./cfgs/finetune/finetune_afno.yaml\
    --cfg_new ./exp_script/afno/baseline-nocertain/finetune_afno_new.yaml
    