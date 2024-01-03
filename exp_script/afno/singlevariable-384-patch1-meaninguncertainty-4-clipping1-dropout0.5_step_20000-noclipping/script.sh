deepspeed --include=localhost:7 --master_port=29577 \
    fourcastnet_deepspeed.py --cfg_path ./cfgs/finetune/finetune_afno.yaml\
    --cfg_new ./exp_script/afno/singlevariable-384-patch1-meaninguncertainty-4-clipping1-dropout0.5_step_20000-noclipping/finetune_afno_new.yaml
    