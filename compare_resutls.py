import numpy as np
import tensorflow as tf

def eval_haps(hap_pred,hap_real,hap_len,batch_size=100):
    hap_pred = tf.reshape(hap_pred,[batch_size,hap_len])
    hap_accuracy= tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(hap_pred,hap_real),dtype=tf.float32),-1)*tf.divide(1,hap_len),-1)*100
    _, auc = tf.metrics.auc(labels=hap_real,predictions=hap_pred)
    return 100-hap_accuracy, auc


def compare_PHASE(phase_out_file,true_haps1,true_haps2):    
    try:
        file_object = open(phase_out_file,"r")
    except:
        print("Output file \""+phase_out_file+"\" not found. Please point to an actual PHASE_i_out_pairs file.")
        return
    
    #read in all of file
    lines = [line.rstrip('\n') for line in file_object]
    
    #save most likely haplotypes for each sample 
    id = -1
    phase_hap1 = {}
    phase_hap2 = {}
    hap_prob = {}
    for line in lines:
        if line.startswith("IND"): #new sample
            id += 1
            hap_prob[id] = 0.0
        else:
            temp_hap1,temp_hap2,temp_hap_prob = [x.strip() for x in line.split(',')]
            
            if float(temp_hap_prob) > hap_prob[id]:
                hap_prob[id] = float(temp_hap_prob)
                phase_hap1[id] = np.fromstring(temp_hap1,'u1') - ord('0')
                phase_hap2[id] = np.fromstring(temp_hap2,'u1') - ord('0')

    hap_len = len(phase_hap1[0])
    samples = len(hap_prob)
    phase_hap1_list = []
    phase_hap2_list = []
    #redefine dict as list
    for i in range(len(hap_prob)):         
        phase_hap1_list.append(phase_hap1[i])
        phase_hap2_list.append(phase_hap2[i])
    
    # cast stuff as tensors so we can compare 
    phase_hap1_list_t = tf.cast(phase_hap1_list,dtype=tf.float32)
    phase_hap2_list_t = tf.cast(phase_hap2_list,dtype=tf.float32)
    true_haps1_t = tf.cast(true_haps1,dtype=tf.float32)
    true_haps2_t = tf.cast(true_haps1,dtype=tf.float32)
    
    print(phase_hap1_list_t.shape)
    print(true_haps1_t.shape)
    
    #haps 1 and 2 maybe be swapped, so for each sample, choose the best order for haps1 and haps2
    error_h11 = eval_haps(phase_hap1_list_t,true_haps1_t,hap_len)
    error_h22 = eval_haps(phase_hap2_list_t,true_haps2_t,hap_len)
    error_h12 = eval_haps(phase_hap1_list_t,true_haps2_t,hap_len)
    error_h12 = eval_haps(phase_hap2_list_t,true_haps1_t,hap_len)
        
    #with tf.Session() as sess:
    #	sess.run(tf.global_variables_initializer())
    #	o=sess.run([error_h11])

    
    
if __name__ =='__main__':

    
    # .npz file made with:
    # number_of_batches = 1000,sample_size=100,sim_hap_length=150,random_seed=1234
    # data_sim.get_data(number_of_batches,sample_size,sim_hap_length,random_seed,phase=range(900,1000),save_file=True)
    
    data = np.load('data/hap_mat_data.npz')

    haps_mat = data['train']
    haps1 = data['hap1_ground']
    haps2 = data['hap1_ground']
    max_hap_len = data['max_hap_len']
    
    batch_num = 999
    
    compare_PHASE("/Users/Jocelyn/Documents/school/ubc/cpsc532R/phase/phase/src/phase_testfiles/PHASE_"+str(batch_num)+"_out_pairs", haps1[batch_num],haps2[batch_num])
