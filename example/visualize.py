import pandas
import fire
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

def draws(name="",path=None,variables = None,step = 1):
    indices = [10,-1,-2,-3,45]

    days = [0,3,11,19] if step == 20 else [0]

    rmses = np.load(f'evaluation/{path}/{name}rmses.npy')
    accs = np.load(f'evaluation/{path}/{name}accs.npy')
    
    if len(rmses.shape) == 1:
        rmses = rmses.reshape(1,-1)
        accs = accs.reshape(1,-1)
    
    np.set_printoptions(suppress=True)
    
    if step == 20:
        x = np.arange(0.25, rmses.shape[0]*0.25+0.25, 0.25)
        fig, axes = plt.subplots(2, len(indices), figsize=(20, 10)) 
        for i, idx in enumerate(indices):
            # RMSE plot
            axes[0, i].plot(x,rmses[:,idx], label='RMSE')
            axes[0, i].set_title(f'RMSE for {variables[i]}')
            axes[0, i].set_xlabel('Measurement Index')
            axes[0, i].set_ylabel('RMSE')
            axes[0, i].legend()

            # Accuracy plot
            axes[1, i].plot(x,accs[:,idx], label='Accuracy')
            axes[1, i].set_title(f'Accuracy for {variables[i]}')
            axes[1, i].set_xlabel('Measurement Index')
            axes[1, i].set_ylabel('Accuracy')
            axes[1, i].legend()

        plt.tight_layout()
        plt.savefig(f'evaluation/{path}/{name}artifacts.png')
        plt.close()
    
    dict = {}
    dict['model'] = path

    for i, idx in enumerate(indices):
        # temp = [rmses[0,idx],rmses[3,idx],rmses[11,idx],rmses[19,idx]]
        temp = [rmses[day,idx] for day in days]
        # 3 digit precision
        temp = [round(x, 3) for x in temp]
        output = '/'.join([str(x) for x in temp])
        
        dict[variables[i]] = output
    
    dict_acc ={}
    
    for i, idx in enumerate(indices):
        temp = [accs[day,idx] for day in days]
        # 3 digit precision
        temp = [round(x, 3) for x in temp]
        output = '/'.join([str(x) for x in temp])
        
        dict_acc[variables[i]] = output
    
    

    logger.info(f'RMSE for {name} is {dict}')
    logger.info(f'Accuracy for {name} is {dict_acc}')
    df = pandas.DataFrame(dict, index=[0], columns=['model','t850', 't2m', '10v', '10u', 'z500'])
    df.to_csv(f'evaluation/{path}/{name}rmse.csv', index=False)

def artifacts (
    path='multivarible-swin-6token-normal-finetune',
    variables = [ 't850', 't2m', '10v', '10u', 'z500'] ):
    
    draw_names = ['trainone_','validone_','train_','']

    steps = [1,1,20,20]

    for i,name in enumerate(draw_names):
        try:
            draws(name=name,path=path,variables=variables,step=steps[i])
        except Exception as e:
            logger.warning(f'{name} not found')
            logger.warning(e)
    # draws(name="validone_",path=path,variables=variables)
    # draws(name="train_",path=path,variables=variables)
    # draws(name="",path=path,variables=variables)
    
    
    
import fire

if __name__ == '__main__':
    fire.Fire(artifacts)