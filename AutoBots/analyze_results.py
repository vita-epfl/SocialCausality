import numpy as np
import matplotlib.pyplot as plt

step_shift = 100
horizon = 500
#horizon_ls = [100, 200, 300, 400, 500, 1000]
low_data_ls = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
# dset_ls = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
dset_ls = ['hotel']
path = './ckps_eval/'

for dset in dset_ls:
 

    for i, low_data in enumerate(low_data_ls):
            
        ADE_ls = np.load(path+dset+'_'+str(low_data)+'_ADE.npy')
        FDE_ls = np.load(path+dset+'_'+str(low_data)+'_FDE.npy')
        start_step = 0
        horizon_steps = int(horizon / 10)
        ADE_sub = np.zeros((1,1))
        count = 0


        while(1):
            ADE_sub_ls = ADE_ls[start_step: start_step+horizon_steps]
            FDE_sub_ls = FDE_ls[start_step: start_step+horizon_steps]

            # get avg and std
            ADE_mean = np.mean(ADE_sub_ls, axis=0)
            FDE_mean = np.mean(FDE_sub_ls, axis=0)
            ADE_std = np.std(ADE_sub_ls, axis=0)
            FDE_std = np.std(FDE_sub_ls, axis=0)
            # append sub
            if count != 0:
                ADE_append = np.array([[ADE_mean]])
                ADE_sub = np.concatenate((ADE_sub, ADE_append),axis=0)
            else:
                ADE_sub[count, 0] = ADE_mean
       
            # shift horizon
            start_step = start_step + int(step_shift/10)
            if start_step+horizon_steps > ADE_ls.shape[0]:
                break
            count += 1
        
        # append all
        if i == 0:
            ADE_all = ADE_sub
        else:
            ADE_all = np.concatenate((ADE_all, ADE_sub), axis=1)

    # plot

    low_data_ls = np.array(low_data_ls)
    for i in range(ADE_all.shape[0]):
        plt.plot(low_data_ls, ADE_all[i], label=str(i))

    plt.xlabel('Percentage of training data')
    plt.ylabel('ADE')
    title = 'Step shift:' + str(step_shift) + ' Horizon:' + str(horizon)
    plt.title(title)
    plt.show()
    plt.savefig('plot.png')

        



