deepspeed --include=localhost:2 --master_port=29522 \
    fourcastnet_deepspeed.py --cfg_path ./cfgs/finetune/finetune_afno.yaml\
    --cfg_new ./exp_script/afno/baseline-AFNO/finetune_afno_new.yaml
    