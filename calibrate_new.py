import numpy as np

# Initialize variables
vcasn = [10,11,12]
bgr = [6,7,8]

v_ind = 0
b_ind = 0
v=vcasn[0]
b=bgr[0]


def calibrate(vcasn, bgr):
   for sec in range(0, 16):
        t.chip.write_gcrpar('BIAS%d_VCASN' % sec, vcasn)
        t.chip.write_gcrpar('BIAS%d_BGR_MEAN' % sec, bgr)



columns = ['vcasn', 'vbgr'] + [f'a[{i}]' for i in range(16)]
df_results = pd.DataFrame(columns=columns)


def splitsave()

    # Split the array into 16 blocks of 32 columns each
    blocks = np.split(res_incr, 16, axis=1)
    
    # Calculate mean for each block
    atot = np.array([block.mean() for block in blocks])


   



    
    # Reset res_incr
    res_incr = np.zeros_like(res_incr)
    
    print('cycle done')

    
    
    res.res_incr = np.full((512, 512), 0.0)


    # Creare una nuova riga con i dati
    new_row = pd.DataFrame({
        'vcasn': [vcasn],
        'vbgr': [bgr],
        **{f'a[{i}]': [atot[i]] for i in range(16)}
    })
    
    # Appendere la riga al DataFrame
    df_results = pd.concat([df_results, new_row], ignore_index=True)
    print('Calibration done')
    
    return df_results








    
       
    




# In the while true


if (active_cal==True):

#condizione sui giri di while true

    if v_ind<len(vcasn):
        calibrate(v,b)
        splitsave()
        v_ind += 1
        v=vcasn[v_ind]
    elif b_ind<len(bgr):
        v_ind = 0
        v=vcasn[v_ind]
        b_ind +=1
        b = bgr[b_ind]
        calibrate(v,b)
        splitsave()
    else:
        df_results.to_pickle("calibration.pkl")
        print('Calibration complete')


        
        