deepspeed --include=localhost:3 --master_port=29533 \
    fourcastnet_deepspeed.py --cfg_path ./cfgs/finetune/finetune_afno.yaml\
    --cfg_new ./exp_script/afno/singlevariable-384-patch1-meaninguncertainty-4-clipping1-dropout0.5_step_20000-noclipping-lr0.00005/finetune_afno_new.yaml
    