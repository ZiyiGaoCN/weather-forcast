deepspeed --include=localhost:0 --master_port=29499 \
    fourcastnet_deepspeed.py --cfg_path ./cfgs/finetune/finetune_afno.yaml\
    --cfg_new ./exp_script/afno/climax-nouncertain/finetune_afno_new.yaml
    