import numpy as np

p_ls = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
pw = str(0.75)

for p in p_ls:
    # ADE = np.load('/scratch/izar/luan/ckps_eval/s2r_baseline_hotel_'+str(p)+'_pw_'+pw+'_ADE.npy')
    # FDE = np.load('/scratch/izar/luan/ckps_eval/s2r_baseline_hotel_'+str(p)+'_pw_'+pw+'_FDE.npy')
    ADE = np.load('/scratch/izar/luan/ckps_eval/s2r_hotel_v1_w_1000_b_64_p_'+str(p)+'_ADE.npy')
    FDE = np.load('/scratch/izar/luan/ckps_eval/s2r_hotel_v1_w_1000_b_64_p_'+str(p)+'_FDE.npy')
    mean_ADE = np.mean(ADE)
    mean_FDE = np.mean(FDE)
    print(round(mean_ADE, 3),'/',round(mean_FDE, 3))
    #print(round(min(ADE), 3), '/', round(min(FDE), 3))
