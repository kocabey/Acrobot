import numpy as np
import tensorflow as tf
from pprint import pprint

def initializer(std):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def update_avg_var_count(avg_a, var_a, count_a, avg_b, var_b, count_b):
    delta = avg_b - avg_a
    m_a = var_a * count_a
    m_b = var_b * count_b
    M2 = m_a + m_b + np.square(delta) * count_a * count_b / (count_a + count_b)
    count = count_a + count_b
    var = M2 / count
    avg = avg_a + delta * count_b / count
    return avg, var, count

class Logger(object):

    def __init__(self, filename, rate=30):
        self.index = 0
        self.rate = rate
        self.file = open("log/{}.txt".format(filename), "w")
    
    def log(self, value):
        self.index += 1
        self.file.write(str(value) + "\n")
        if self.index % self.rate == 0:
            self.file.flush()

class GaussianDist(object):

    def __init__(self, mean, std):
        self.std = std
        self.mean = mean

    def logp(self, x):
        return -tf.reduce_sum(tf.square(x - self.mean), axis=-1)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))
    
    def mode(self):
        return self.mean

class ContinuousPolicy(object):

    def __init__(
        self, 
        obv_, 
        num_hidden, 
        hidden_size, 
        act_space,
        epsilon=1e-4, 
        scope=""
    ):
        with tf.variable_scope(scope):
            shape = obv_.shape[1:]
            self._new_avg   = tf.placeholder(shape=shape, dtype=tf.float32)
            self._new_var   = tf.placeholder(shape=shape, dtype=tf.float32)
            self._new_count = tf.placeholder(shape=(),    dtype=tf.float32)
            self._avg = tf.get_variable(
                'rms/avg',
                dtype=tf.float32,
                initializer=np.zeros(shape, np.float32)
            )
            self._var = tf.get_variable(
                'rms/var',
                dtype=tf.float32,
                initializer=np.ones(shape, np.float32)
            )
            self._count = tf.get_variable(
                'rms/count',
                dtype=tf.float32,
                initializer=np.full((), epsilon, np.float32)
            )
            self.update_ops = tf.group([
                self._avg.assign(self._new_avg),
                self._var.assign(self._new_var),
                self._count.assign(self._new_count)
            ])

            obv_normalized_ = obv_ - self._avg
            obv_normalized_ = obv_normalized_ / tf.sqrt(tf.maximum(self._var, 1e-2))
            obv_normalized_ = tf.clip_by_value(obv_normalized_, -5.0, 5.0)

            pol_hidden = obv_normalized_
            val_hidden = obv_normalized_
            for i in range(num_hidden):
                pol_hidden = tf.layers.dense(
                    pol_hidden,
                    hidden_size,
                    name="pol_hidden{}".format(i+1),
                    kernel_initializer=initializer(1.0)
                )
                val_hidden = tf.layers.dense(
                    val_hidden,
                    hidden_size,
                    name="val_hidden{}".format(i+1),
                    kernel_initializer=initializer(1.0)
                )
                pol_hidden = tf.nn.tanh(pol_hidden)
                val_hidden = tf.nn.tanh(val_hidden)

            mean = tf.layers.dense(
                pol_hidden,
                act_space,
                name="mean",
                kernel_initializer=initializer(0.01)
            )
            value = tf.layers.dense(
                val_hidden,
                1,
                name="value",
                kernel_initializer=initializer(1.0)
            )
            self.std = tf.placeholder(
                shape=[],
                name="std", 
                dtype=tf.float32
            )
            self.obv_ = obv_
            self.value = value[:, 0]
            self.dist = GaussianDist(mean, self.std)
            self.stochastic_action = self.dist.sample()
            self.deterministic_action = self.dist.mode()
            self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)
            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.trainable_variables = [var for var in self.trainable_variables if "rms" not in var.name]

    def init_rms_params(self, sess):
        sess.run(tf.variables_initializer([
            self._avg, 
            self._var,
            self._count
        ]))
        self.avg, self.var, self.count = sess.run([
            self._avg, 
            self._var, 
            self._count
        ])

    def update_rms_params(self, batch, sess):
        batch_avg = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]

        avg, var, count = update_avg_var_count(
            self.avg, 
            self.var, 
            self.count, 
            batch_avg, 
            batch_var, 
            batch_count
        )

        sess.run(self.update_ops, feed_dict={
            self._new_avg: avg,
            self._new_var: var,
            self._new_count: count
        })

        self.avg, self.var, self.count = sess.run([
            self._avg, 
            self._var, 
            self._count
        ])

class Dataset(object):

    def __init__(self, seg, keys):
        self.seg = seg
        self.keys = keys
        self.length = len(seg[keys[0]])

    def iterate(self, batch_size):
        indices = np.arange(self.length)
        np.random.shuffle(indices)
        for key in self.keys:
            self.seg[key] = self.seg[key][indices]
        for i in range(self.length // batch_size):
            begin =   i   * batch_size
            end   = (i+1) * batch_size
            batch = {}
            for key in self.keys:
                batch[key] = self.seg[key][begin: end]
            yield batch

def add_advs_and_rets(seg, gam, lam):
    T = len(seg["rews"])
    rews = seg["rews"]
    news = np.append(seg["news"], 0)
    vals = np.append(seg["vals"], seg["nval"])
    seg["advs"] = gaelam = np.empty(T, np.float32)
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - news[t+1]
        delta = rews[t] + gam * vals[t+1] * nonterminal - vals[t]
        gaelam[t] = lastgaelam = delta + gam * lam * nonterminal * lastgaelam
    seg["rets"] = seg["advs"] + seg["vals"]
    seg["advs"] -= seg["advs"].mean()
    seg["advs"] /= seg["advs"].std()

def rollout(
        policy, 
        sess, 
        env, 
        horizon,
        min_std,
        max_std,
        std_iterations,
        stochastic=True
    ):
    t = 0
    new = True
    obv = env.reset()
    act = np.zeros(env.act_space(), np.float32)
    cur_std = max_std
    std_step = (max_std - min_std) / std_iterations

    obvs = np.array([obv for _ in range(horizon)])
    acts = np.array([act for _ in range(horizon)])

    rews = np.zeros(horizon, np.float32)
    vals = np.zeros(horizon, np.float32)
    news = np.zeros(horizon, np.int32)

    if stochastic: 
        action = policy.stochastic_action
    else: 
        action = policy.deterministic_action

    while True:
        act, val = sess.run(
            [action, policy.value],
            feed_dict = {
                policy.obv_: obv[None],
                policy.std: cur_std
            }
        )

        act = act[0]
        val = val[0]
        nval = val * (1 - new)

        if t > 0 and t % horizon == 0:
            yield {
                "obvs" : obvs,
                "rews" : rews,
                "news" : news,
                "acts" : acts,
                "vals" : vals,
                "nval" : nval,
                "cur_std" : cur_std
            }
            cur_std -= std_step
            cur_std = max(cur_std, min_std)

        i = t % horizon
        obvs[i] = obv
        news[i] = new
        acts[i] = act
        vals[i] = val

        obv, rew, new, _ = env.step(act)
        rews[i] = rew

        if new: obv = env.reset()

        t += 1

def learn(
        trainenv,
        gam=0.99,
        lam=0.95,
        clip=0.2,
        epochs=10,
        horizon=2048,
        num_hidden=2,
        hidden_size=64,
        batch_size=64,
        val_coeff=0.5,
        learning_rate=1e-4,
        max_grad_norm=0.5,
        max_std=1.0,
        min_std=0.01,
        std_iterations=10000,
        save_interval=100,
        max_iterations=100000000,
        restore_path=None,
        experiment_name="none"
    ):

    obv_space = trainenv.obv_space()
    act_space = trainenv.act_space()

    obv_ = tf.placeholder(tf.float32, shape=[None, obv_space])
    act_ = tf.placeholder(tf.float32, shape=[None, act_space])

    adv_ = tf.placeholder(tf.float32, shape=[None])
    ret_ = tf.placeholder(tf.float32, shape=[None])

    new_policy = ContinuousPolicy(obv_, num_hidden, hidden_size, act_space, scope="new")
    old_policy = ContinuousPolicy(obv_, num_hidden, hidden_size, act_space, scope="old")

    nlp = new_policy.dist.logp(act_)
    olp = old_policy.dist.logp(act_)

    pairs = zip(old_policy.variables, new_policy.variables)
    assign_old = [tf.assign(old, new) for old, new in pairs]
    saver = tf.train.Saver(
        var_list=new_policy.variables,
        max_to_keep=None
    )

    ratio = tf.exp(nlp - olp)
    pol_surr1 = ratio * adv_
    pol_surr2 = tf.clip_by_value(ratio, 1-clip, 1+clip) * adv_
    pol_loss = -tf.reduce_mean(tf.minimum(pol_surr1, pol_surr2))

    nv = new_policy.value
    ov = old_policy.value 
    nv_clipped = ov + tf.clip_by_value(nv - ov, -clip, clip)
    val_surr1 = tf.square(nv - ret_)
    val_surr2 = tf.square(nv_clipped - ret_)
    val_loss = 0.5 * tf.reduce_mean(tf.maximum(val_surr1, val_surr2))

    total_loss = pol_loss + val_coeff * val_loss
    trainer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-5)
    grads_and_var = trainer.compute_gradients(total_loss, new_policy.trainable_variables)
    grads, var = zip(*grads_and_var)
    grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
    grads_and_var = list(zip(grads, var))
    train_step = trainer.apply_gradients(grads_and_var)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    new_policy.init_rms_params(sess)
    if restore_path is not None:
        saver.restore(sess, restore_path)

    train_generator = rollout(
        new_policy, 
        sess, 
        trainenv, 
        horizon, 
        min_std,
        max_std,
        std_iterations,
        stochastic=True
    )

    rews_logger = Logger("{}_rews".format(experiment_name))
    mean_logger = Logger("{}_mean".format(experiment_name))
    std_logger = Logger("{}_std".format(experiment_name))
    ploss_logger = Logger("{}_ploss".format(experiment_name))
    vloss_logger = Logger("{}_vloss".format(experiment_name))
    
    print("Optimizing...")
    for i in range(max_iterations):
        seg = train_generator.__next__()
        reward = np.mean(seg["rews"]) 
        add_advs_and_rets(seg, gam, lam)
        new_policy.update_rms_params(seg["obvs"], sess)
        dataset = Dataset(seg, ["obvs", "acts", "advs", "rets"])
        sess.run(assign_old)
        count = 0
        mean_total = 0
        ploss_total = 0
        vloss_total = 0
        for epoch in range(epochs):
            for batch in dataset.iterate(batch_size):
                _, ploss, vloss, m = sess.run(
                    [train_step, 
                     pol_loss,
                     val_loss,
                     new_policy.dist.mean],
                    feed_dict = {
                        obv_ : batch["obvs"],
                        act_ : batch["acts"],
                        adv_ : batch["advs"],
                        ret_ : batch["rets"]
                    })
                mean_total += np.mean(m)
                ploss_total += np.mean(ploss)
                vloss_total += np.mean(vloss)
                count += 1

        mean_total /= count
        ploss_total /= count
        vloss_total /= count

        rews_logger.log(reward)
        mean_logger.log(mean_total)
        std_logger.log(seg["cur_std"])
        ploss_logger.log(ploss_total)
        vloss_logger.log(vloss_total)

        print(reward, mean_total, seg["cur_std"])

        if i % save_interval == 0 or i == max_iterations - 1:
            saver.save(sess, "models/{}_{}".format(experiment_name, i))
            print("Saved the model successfully!")

