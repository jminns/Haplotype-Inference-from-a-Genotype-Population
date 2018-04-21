import msprime
import numpy as np
import random 

def make_hap_mat(genotypes,sample_size,var_len):
    haps_mat = np.zeros([sample_size,var_len,3])
    for k,i in enumerate(genotypes):
        for l,j in enumerate(i):
            haps_mat[k,l,j]=1
    return haps_mat

def simulate(number_of_batches,sample_size,length,desired_hap_length,random_seed):
    # Read in the recombination map using the read_hapmap method,
    #infile = "hapmap/genetic_map_GRCh37_chr22.txt"
    #recomb_map = msprime.RecombinationMap.read_hapmap(infile)
 
    # Next, get number_of_batches sets of data 
    
    # Run the actual simulations
    ts = msprime.simulate(
        sample_size=sample_size*2*number_of_batches, #number of haplotypes for all number_of_batches sets of data
        length = length,
        Ne=1e6,
        recombination_rate=2.3*10**-8,
        #recombination_map=recomb_map,
        mutation_rate=2e-8,
        random_seed=random_seed  
    )
    
    if desired_hap_length == None or ts.num_sites < desired_hap_length:
        desired_hap_length = ts.num_sites
    
    genotypes = np.zeros(shape=(sample_size*number_of_batches,desired_hap_length)).astype(int)
    haplotypes = np.zeros(shape=(sample_size*2*number_of_batches,desired_hap_length)).astype(int)
    
    #create N genotypes given 2N haplotypes
    for i, hap in enumerate(ts.haplotypes()):
        genotypes[i%(sample_size*number_of_batches)] += np.array(map(int, hap[0:desired_hap_length]))
        haplotypes[i] = np.array(map(int, hap[0:desired_hap_length]))

    #var_len = haplotypes.shape[1]
    #print(var_len)
    
    haps_mats = np.zeros([number_of_batches,sample_size,desired_hap_length,3])
    for i in range(number_of_batches):
        haps_mats[i,:,:,:] = make_hap_mat(genotypes[i*sample_size:(i+1)*sample_size],sample_size,desired_hap_length)

    # break ground truth data into batches 
    hap1_ground = np.zeros(shape=(number_of_batches,sample_size,desired_hap_length)).astype(int)
    hap2_ground = np.zeros(shape=(number_of_batches,sample_size,desired_hap_length)).astype(int)
    for i in range(number_of_batches):
        hap1_ground[i,:,:] = haplotypes[i*sample_size:(i+1)*sample_size]
        hap2_ground[i,:,:] = haplotypes[(i+number_of_batches)*sample_size:(i+1+number_of_batches)*sample_size]

    
    return haps_mats, hap1_ground, hap2_ground, desired_hap_length, ts
            
        
def save_phase_file(hap1,haps2, sample_size, ts,desired_hap_length,filename="PHASE.txt"):

    if desired_hap_length == None or ts.num_sites < desired_hap_length:
        desired_hap_length = ts.num_sites
        
    #swap some haps sites (we don't want this to be too easy)
    for i in range(sample_size):
        for j in range(desired_hap_length):
            if hap1[i,j] + haps2[i,j] == 1 and random.randint(1, 2) == 2:
                temp = hap1[i,j]
                hap1[i,j] = haps2[i,j]
                haps2[i,j] = temp
    
    haps=np.concatenate((hap1,haps2))
    f = open(filename,'w')
    f.write(str(sample_size)+'\n')
    f.write(str(desired_hap_length)+'\n')
    f.write('P ')
    for i, variant in enumerate(ts.variants()):
        if i < desired_hap_length:
            f.write(str(variant.site.position) + ' ')
    f.write('\n')
    f.write("S" * desired_hap_length)
    f.write('\n')

    for i in range(sample_size):
        f.write("#"+str(i)+"\n")
        f.write(''.join(map(str,haps[i][0:desired_hap_length]))+'\n')
        f.write(''.join(map(str,haps[i+sample_size][0:desired_hap_length]))+'\n')
    f.close() 

def get_data(number_of_batches,sample_size,sim_hap_length,random_seed,desired_hap_length=None,save_file=False,phase=None):

    # still need to determine what the correct format for the ground truth data is
    train,hap1_ground,hap2_ground,max_hap_len,ts = simulate(number_of_batches,sample_size,sim_hap_length,desired_hap_length,random_seed)
    
    if phase != None:
        if desired_hap_length == None:
            desired_hap_length = max_hap_len
        for i in phase:
	        save_phase_file(hap1_ground[i],hap2_ground[i],sample_size,ts,desired_hap_length,"phase_data/PHASE_"+str(i)+".txt")
	        
    
    if save_file:
       np.savez('hap_mat_data', train=train, hap1_ground=hap1_ground,hap2_ground=hap2_ground,max_hap_len=max_hap_len)
    else:
       return train,hap1_ground,hap2_ground,max_hap_len
