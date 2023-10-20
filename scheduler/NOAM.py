from torch.optim.lr_scheduler import LambdaLR

def NOAMLR(optimizer , warmup_steps=1000 , scale=1.0,model_size=192):
    return LambdaLR(optimizer, lambda s: (1000 ** -0.5) * min( ( max(s,1) ** -0.5), s * warmup_steps ** -1.5) * scale)