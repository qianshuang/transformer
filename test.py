# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


beta = tf.Variable(tf.zeros(8))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(beta))
