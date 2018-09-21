import pickle

from model import *

def test():
    time1 = time.time()
    input_path_S = pickle.load(open(a.input_dir+'style.txt','r'))
    input_path_C = pickle.load(open(a.input_dir+'content.txt','r'))
    target_path = pickle.load(open(a.input_dir+'target.txt','r'))
    print(time.time() - time1)

    ####################### network ################
    batch_inputsS_holder = tf.placeholder(tf.float32, [a.style_num*a.style_sample_n,80,80,1],name='inputsS')
    batch_inputsC_holder = tf.placeholder(tf.float32, [a.content_num*a.content_sample_n,80,80,1],name='inputsC')
    batch_targets_holder = tf.placeholder(tf.float32, [a.target_batch_size,80,80,1],name='targets')

    # compute the number of black pixels
    black = tf.greater(0.5, batch_targets_holder)
    as_ints = tf.cast(black, tf.int32)
    zero_n = tf.reduce_sum(as_ints,[1,2,3])+1

    # compute the mean of black pixels
    zeros = tf.zeros_like(batch_targets_holder)
    new_tensor = tf.where(black, batch_targets_holder, zeros)
    mean_pixel_value = tf.reduce_sum(new_tensor,[1,2,3])/tf.to_float(zero_n)

    # zero_n = tf.placeholder(tf.float32,[a.target_batch_size,1],name='zero_n')
    # mean_pixel_value = tf.placeholder(tf.float32,[a.target_batch_size,1],name='mean_pixel_value')

    with tf.variable_scope("generator"):
        pictures_decode, model_loss, model_mse = create_generator(batch_inputsS_holder, batch_inputsC_holder,
                                                                 batch_targets_holder, zero_n, mean_pixel_value)

    #########prepare data ###################################
    input_path_S_holder = tf.placeholder(tf.string)
    input_path_C_holder = tf.placeholder(tf.string)
    target_path_holder = tf.placeholder(tf.string)

    dataset1 = tf.data.Dataset.from_tensor_slices(input_path_S_holder)
    dataset1 = dataset1.map(process,num_parallel_calls=a.num_parallel_prefetch)
    dataset1 = dataset1.prefetch(a.style_sample_n*a.style_num * a.num_parallel_prefetch)
    dataset1 = dataset1.batch(a.style_sample_n*a.style_num).repeat(a.max_epochs)

    dataset2 = tf.data.Dataset.from_tensor_slices(input_path_C_holder)
    dataset2 = dataset2.map(process,num_parallel_calls=a.num_parallel_prefetch)
    dataset2 = dataset2.prefetch(a.content_sample_n*a.content_num * a.num_parallel_prefetch)
    dataset2 = dataset2.batch(a.content_sample_n*a.content_num).repeat(a.max_epochs)

    dataset3 = tf.data.Dataset.from_tensor_slices(target_path_holder)
    dataset3 = dataset3.map(process,num_parallel_calls=a.num_parallel_prefetch)
    dataset3 = dataset3.prefetch(a.target_batch_size * a.num_parallel_prefetch)
    dataset3 = dataset3.batch(a.target_batch_size).repeat(a.max_epochs)

    iterator1 = dataset1.make_initializable_iterator()
    one_element1 = tf.convert_to_tensor(iterator1.get_next())

    iterator2 = dataset2.make_initializable_iterator()
    one_element2 = tf.convert_to_tensor(iterator2.get_next())

    iterator3 = dataset3.make_initializable_iterator()
    one_element3 = tf.convert_to_tensor(iterator3.get_next())

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        model_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        # model_optim = tf.train.RMSPropOptimizer(a.rmsprop_lr)
        # learning_rate = tf.train.exponential_decay(a.adam_lr, global_step, a.decay_steps, a.decay_rate)
        model_optim = tf.train.AdamOptimizer(a.adam_lr)
        model_grads_and_vars = model_optim.compute_gradients(model_loss, var_list=model_tvars)
        model_train = model_optim.apply_gradients(model_grads_and_vars)

    variables = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]

    ###########################################################################

    saver = tf.train.Saver(max_to_keep=1, var_list=variables+bn_moving_vars)
    logdir = a.output_dir
    sv = tf.train.Supervisor(logdir=logdir, saver=None, summary_op=None)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with sv.managed_session(config=config) as sess:

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)
            print 'ok'

        sess.run(iterator1.initializer, feed_dict={input_path_S_holder: input_path_S})
        sess.run(iterator2.initializer, feed_dict={input_path_C_holder: input_path_C})
        sess.run(iterator3.initializer, feed_dict={target_path_holder: target_path})

        max_steps = len(target_path)/a.target_batch_size
        print max_steps

        for step in range(max_steps):
            print step

            size = [5,10]
            batch_inputsS = sess.run(one_element1)
            batch_inputsC = sess.run(one_element2)
            batch_targets = sess.run(one_element3)

            output_images = sess.run(pictures_decode,feed_dict={batch_inputsS_holder:batch_inputsS,
                                                            batch_inputsC_holder:batch_inputsC,
                                                            batch_targets_holder:batch_targets})

            save_images(output_images,step,size,'outputs')
            save_images(batch_targets,step,size,'targets')
            # save_images(batch_inputsC,step,[10,10],'inputsC')
            if sv.should_stop():
                break

