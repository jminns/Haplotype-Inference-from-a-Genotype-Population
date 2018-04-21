from sys import argv
import logging
import tensorflow as tf
import numpy as np
from encoder_haps import hap_inf

def eval_haps(hap_pred, hap_real,hap_len,batch_size=100):
    hap_pred = tf.reshape(hap_pred,[batch_size,hap_len])
    hap_accuracy= tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.rint(hap_pred),hap_real),dtype=tf.float32),-1),-1)*tf.divide(100,hap_len)
    _, auc = tf.metrics.auc(labels=hap_real,predictions=hap_pred)
    full_hap_accuracy = tf.reduce_mean(tf.cast(tf.reduce_all((tf.equal(tf.rint(hap_pred),hap_real)),-1),dtype=tf.float32),-1)*100
    return 100-hap_accuracy, auc, 100-full_hap_accuracy


def train(**kwargs):

    max_hap_len = kwargs['max_hap_len']
    kl = kwargs['KL']
    units_bilstm = kwargs['units_bilstm']
    units_bilstm2 = kwargs['units_bilstm2']
    units_firstlstm = kwargs['units_firstlstm']
    dense_units = kwargs['dense_units']
    batch_size = kwargs['batch_size']
    dim_ancestors = kwargs['dim_ancestors']
    num_ancs = kwargs['num_ancs']
    num_epochs = kwargs['num_epochs']
    temperature = kwargs['post_temp']
    temperature_prior = kwargs['prior_temp']
    ort_constr = kwargs['ort_constr']
    classes = 3
    log_dir = kwargs['log_dir']
    log_file=kwargs['log_file']
    single_sample = kwargs['single_sample']
    lmbda = kwargs['lambda']

    #misc
    global_step = tf.Variable(0, name='global_step', trainable=False)
    #training placholders
    true_hap1_pl_train=tf.placeholder(shape=[batch_size,max_hap_len],dtype=tf.float32,name='true_hap1_pl_train')
    true_hap2_pl_train=tf.placeholder(shape=[batch_size,max_hap_len],dtype=tf.float32,name='true_hap2_pl_train')
    batch_a_haps_train_pl = tf.placeholder(shape=[batch_size,max_hap_len,classes],dtype=tf.float32,name='train_pl_haps')

    hap_lens_pl = tf.placeholder(shape=[batch_size],dtype=tf.int32,name='lengths_of_each_hap')

    #testing placeholders
    true_hap1_pl_test=tf.placeholder(shape=[batch_size,max_hap_len],dtype=tf.float32,name='true_hap1_pl_test')
    true_hap2_pl_test=tf.placeholder(shape=[batch_size,max_hap_len],dtype=tf.float32,name='true_hap2_pl_test')
    batch_a_haps_test_pl = tf.placeholder(shape=[batch_size,max_hap_len,classes],dtype=tf.float32, name='test_pl_haps')

    # training input
        # this doesn't change in our input
    lens = np.repeat(max_hap_len,batch_size)
    data = np.load('/home/zalperst/PycharmProjects/vae_proj/hap_data/hap_mat_data_train.npz')

    haps_mat = data['train']
    hap1_true = data['hap1_ground']
    hap2_true = data['hap2_ground']
    hap1_true_train = hap1_true[0:900]
    hap2_true_train = hap2_true[0:900]
    hap1_true_test = hap1_true[900:1000]
    hap2_true_test = hap1_true[900:1000]
    #test_data = np.load('/home/zalperst/PycharmProjects/vae_proj/hap_data/hap_mat_data_test.npz')
    #test_haps_mat = test_data['train']

    test_haps_mat = haps_mat[900:1000]
    haps_mat = haps_mat[0:900]
    print('Input shape {}'.format(np.shape(haps_mat)))
    # testing input



    net = hap_inf(lmbda=lmbda,single_sample=single_sample,max_hap_len=max_hap_len, units_bilstm2=units_bilstm2, units_bilstm=units_bilstm, dense_units=dense_units, units_firstlstm=units_firstlstm,
                        hap_lens=hap_lens_pl, len_ancs=num_ancs, temperature=temperature, batch_size=batch_size, dim_ancs=dim_ancestors,
                       temperature_prior=temperature_prior)
    net.kl=kl

    #train_cost, reconstruction, full_kl,kl_p1_h1, kl_p2_h1,kl_h1,kl_h2,pred_h1_train,lat_h1_train,params_h1_train,params_h2_train,param_prior,kl_p1_h1_p1,kl_p1_h1_p2,kl_p1_h1_p2_2,kl_p1_h1_p2_3,kl_p2_h1_log = net.run_network(hap_lens =hap_lens_pl ,batch_a_haps=batch_a_haps_train_pl)
    train_cost, reconstruction, full_kl,log_prob_h1,log_prob_pr_h1,lat_h1,pred_h1_train, pred_h2_train = net.run_network(ort_constr=ort_constr,hap_lens =hap_lens_pl,batch_a_haps=batch_a_haps_train_pl)

    pred_h1_test, pred_h2_test, lat_h1t, lat_h2t, param_h1, param_h2 = net.test_network(hap_lens=hap_lens_pl,batch_a_haps=batch_a_haps_test_pl)

    test_cost,kl_h1_t, kl_h2_t = net.compute_test_loss(pred_h1=pred_h1_test, pred_h2=pred_h2_test, param_h1=param_h1, param_h2=param_h2,input=batch_a_haps_test_pl)

    ###train eval haps
    error_rate_h11_train, auc_h11_train,haperror_rate_h11_train = eval_haps(hap_pred=pred_h1_train, hap_real=true_hap1_pl_train,
                                                  hap_len=max_hap_len)
    error_rate_h22_train, auc_h22_train,haperror_rate_h22_train = eval_haps(hap_pred=pred_h2_train, hap_real=true_hap2_pl_train,
                                                  hap_len=max_hap_len)
    error_rate_h21_train, auc_h21_train,haperror_rate_h21_train = eval_haps(hap_pred=pred_h2_train, hap_real=true_hap1_pl_train,
                                                  hap_len=max_hap_len)
    error_rate_h12_train, auc_h12_train,haperror_rate_h12_train = eval_haps(hap_pred=pred_h1_train, hap_real=true_hap2_pl_train,
                                                  hap_len=max_hap_len)
    sum_error_rate_h11_train = tf.summary.scalar(name='error_rate_h11_train', tensor=error_rate_h11_train)
    sum_error_rate_h22_train = tf.summary.scalar(name='error_rate_h22_train', tensor=error_rate_h22_train)
    sum_error_rate_h21_train = tf.summary.scalar(name='error_rate_h21_train', tensor=error_rate_h21_train)
    sum_error_rate_h12_train = tf.summary.scalar(name='error_rate_h12_train', tensor=error_rate_h12_train)
    sum_auc_h11_train = tf.summary.scalar(name='auc_h11_train', tensor=auc_h11_train)
    sum_auc_h22_train= tf.summary.scalar(name='auc_h22_train', tensor=auc_h22_train)
    sum_auc_h21_train = tf.summary.scalar(name='auc_h21_train', tensor=auc_h21_train)
    sum_auc_h12_train = tf.summary.scalar(name='auc_h12_train', tensor=auc_h12_train)
    sum_haperror_rate_h12_train = tf.summary.scalar(name='haperror_rate_h12_train',tensor=haperror_rate_h12_train)
    sum_haperror_rate_h22_train = tf.summary.scalar(name='haperror_rate_h22_train',tensor=haperror_rate_h22_train)
    sum_haperror_rate_h21_train = tf.summary.scalar(name='haperror_rate_h11_train',tensor=haperror_rate_h11_train)
    sum_haperror_rate_h11_train = tf.summary.scalar(name='haperror_rate_h21_train',tensor=haperror_rate_h21_train)
    ###test eval haps
    error_rate_h11_test,auc_h11_test,haperror_rate_h11_test = eval_haps(hap_pred=pred_h1_test, hap_real=true_hap1_pl_test, hap_len=max_hap_len)
    error_rate_h22_test, auc_h22_test,haperror_rate_h22_test = eval_haps(hap_pred=pred_h2_test, hap_real=true_hap2_pl_test, hap_len=max_hap_len)
    error_rate_h21_test,auc_h21_test,haperror_rate_h21_test = eval_haps(hap_pred=pred_h2_test, hap_real=true_hap1_pl_test, hap_len=max_hap_len)
    error_rate_h12_test, auc_h12_test,haperror_rate_h12_test = eval_haps(hap_pred=pred_h1_test, hap_real=true_hap2_pl_test, hap_len=max_hap_len)
    sum_haperror_rate_h12_test = tf.summary.scalar(name='haperror_rate_h12_test',tensor=haperror_rate_h12_test)
    sum_haperror_rate_h22_test = tf.summary.scalar(name='haperror_rate_h22_test',tensor=haperror_rate_h22_test)
    sum_haperror_rate_h21_test = tf.summary.scalar(name='haperror_rate_h11_test',tensor=haperror_rate_h11_test)
    sum_haperror_rate_h11_test = tf.summary.scalar(name='haperror_rate_h21_test',tensor=haperror_rate_h21_test)
    sum_error_rate_h11_test= tf.summary.scalar(name='error_rate_h11_test',tensor=error_rate_h11_test)
    sum_error_rate_h22_test = tf.summary.scalar(name='error_rate_h22_test', tensor=error_rate_h22_test)
    sum_error_rate_h21_test = tf.summary.scalar(name='error_rate_h21_test', tensor=error_rate_h21_test)
    sum_error_rate_h12_test = tf.summary.scalar(name='error_rate_h12_test', tensor=error_rate_h12_test)
    sum_auc_h11_test = tf.summary.scalar(name='auc_h11_test',tensor=auc_h11_test)
    sum_auc_h22_test = tf.summary.scalar(name='auc_h22_test',tensor=auc_h22_test)
    sum_auc_h21_test = tf.summary.scalar(name='auc_h21_test',tensor=auc_h21_test)
    sum_auc_h12_test = tf.summary.scalar(name='auc_h12_test',tensor=auc_h12_test)

    # Train Step
    # clipping gradients
    ######
    lr = 1e-3
    opt = tf.train.AdamOptimizer(lr)
    grads_t, vars_t = zip(*opt.compute_gradients(train_cost))
    clipped_grads_t, grad_norm_t = tf.clip_by_global_norm(grads_t, clip_norm=5.0)
    train_step = opt.apply_gradients(zip(clipped_grads_t, vars_t), global_step=global_step)
    grad_norm_sum = tf.summary.scalar(name='grad_norm',tensor=grad_norm_t)
##logging handle
    log_file = log_dir + log_file+'.txt'
    logger = logging.getLogger('hap_log')
    hdlr = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.DEBUG)

    #tensorflow session stuff
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    ######
    #tensorboard stuff

    if ort_constr:
        summary_inf_train = tf.summary.merge([sum_haperror_rate_h21_train,sum_haperror_rate_h11_train,sum_haperror_rate_h12_train,sum_haperror_rate_h22_train,grad_norm_sum,net.ort_sum,net.sum_kl_train,net.sum_elbo_train,net.sum_rec_train,sum_error_rate_h11_train,sum_error_rate_h22_train,sum_error_rate_h21_train,sum_error_rate_h12_train,sum_auc_h11_train,sum_auc_h22_train,sum_auc_h21_train,sum_auc_h12_train])

    else:
        summary_inf_train = tf.summary.merge([sum_haperror_rate_h21_train,sum_haperror_rate_h11_train,sum_haperror_rate_h12_train,sum_haperror_rate_h22_train,grad_norm_sum,net.sum_kl_train,net.sum_elbo_train,net.sum_rec_train,sum_error_rate_h11_train,sum_error_rate_h22_train,sum_error_rate_h21_train,sum_error_rate_h12_train,sum_auc_h11_train,sum_auc_h22_train,sum_auc_h21_train,sum_auc_h12_train])
    summary_inf_test = tf.summary.merge([sum_haperror_rate_h21_test,sum_haperror_rate_h11_test,sum_haperror_rate_h12_test,sum_haperror_rate_h22_test,net.sum_kl_test,net.sum_elbo_test,net.sum_rec_test,sum_error_rate_h11_test,sum_error_rate_h22_test,sum_error_rate_h21_test,sum_error_rate_h12_test,sum_auc_h11_test,sum_auc_h22_test,sum_auc_h21_test,sum_auc_h12_test])
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    n_batches_inds = range(np.shape(haps_mat)[0])
    for count in range(num_epochs):
        np.random.shuffle(n_batches_inds)
        for batch in n_batches_inds:
            sess.run(tf.local_variables_initializer())

            #kl_p2_h1_log_o,kl_p1_h1_p2_3_o,kl_p1_h1_p1_o, kl_p1_h1_p2_o, kl_p1_h1_p2_2_o,param_prior_o,params_h1_train_o, params_h2_train_o,cost_o,rec_o,kl_o,global_step_o,kl_p1_h1_o, kl_p2_h1_o,kl_h1_o,kl_h2_o,pred_h1_o,lat_h1_o =sess.run([kl_p2_h1_log,kl_p1_h1_p2_3,kl_p1_h1_p1,kl_p1_h1_p2,kl_p1_h1_p2_2,param_prior,params_h1_train,params_h2_train,train_cost, reconstruction, full_kl,global_step,kl_p1_h1, kl_p2_h1,kl_h1,kl_h2,pred_h1_train,lat_h1_train],feed_dict={hap_lens_pl:lens,batch_a_haps_train_pl:haps_mat[batch]})
            pred_h1_train_o,pred_h2_train_o,cost_o,rec_o,kl_o,global_step_o,_,summary_inf_train_o=sess.run([pred_h1_train,pred_h2_train,train_cost, reconstruction, full_kl,global_step,train_step,summary_inf_train],feed_dict={true_hap1_pl_train:hap1_true_train[batch],true_hap2_pl_train:hap2_true_train[batch],hap_lens_pl:lens,batch_a_haps_train_pl:haps_mat[batch]})
            print('h1 pred {} \n h2 pred {} \n true h1 {}\n true_h2 {}'.format(pred_h1_train_o[0][0:10],pred_h2_train_o[0][0:10],hap1_true_train[batch][0][0:10],hap2_true_train[batch][0][0:10]))
            summary_writer.add_summary(summary_inf_train_o, global_step_o)
            summary_writer.flush()
            if global_step_o % 1 == 0:
                sess.run(tf.local_variables_initializer())
                r_num = np.random.randint(low=0,high=np.shape(test_haps_mat)[0])
                kl_h1_ot, kl_h2_ot,test_cost_o, lat_h1_t, lat_h2_t, param_h1_t, param_h2_t,summary_inf_test_o = sess.run([kl_h1_t, kl_h2_t,test_cost, lat_h1t, lat_h2t, param_h1, param_h2,summary_inf_test],feed_dict={hap_lens_pl:lens,batch_a_haps_test_pl:test_haps_mat[r_num],true_hap1_pl_test:hap1_true_test[r_num],true_hap2_pl_test:hap2_true_test[r_num]})
                print('Test Cost {}'.format(test_cost_o))
                print('Train Cost: {}, Rec Cost: {}, KL: {} Global_step {}'.format(cost_o, rec_o, kl_o, global_step_o))
                print('kl 1 {} kl 2 {}'.format(kl_h1_ot, kl_h2_ot))
                summary_writer.add_summary(summary_inf_test_o, global_step_o)
                summary_writer.flush()


if __name__=='__main__':
    loc = argv[-1]
    dict ={'lambda':1,'single_sample':True,'KL':False,'ort_constr':False,'num_epochs':100,'num_ancs':10,'dim_ancestors':256,'dense_units':256,'units_firstlstm':256,'units_bilstm2':256,'units_bilstm':256,'max_hap_len':143,'batch_size':100,'prior_temp':0.5,'post_temp':0.5,'log_dir':loc,'log_file':'256x_prtmp01_potmp01_10ancs_nokl'}


    train(**dict)
