# -*- coding: utf-8 -*-
from buffalo.misc import aux
import json
import time

import os
import sys
import time

from buffalo.algo import als, matrix, feature


class ALS:
    def __init__(self):
        self.logger = Aux.get_logger('ALS')

    def fetch_train_data(self):
        if not os.path.isfile('./res/main.mtx'):
            """
            [{"tags": ["락"], "id": 61281, 
            "plylst_title": "여행같은 음악", 
            "songs": [525514, 129701, 383374, 562083, 297861, 139541, 351214, 650298, 531057, 205238, 706183, 127099, 660493, 461973, 121455, 72552, 223955, 324992, 50104], 
            "like_cnt": 71, "updt_date": "2013-12-19 18:36:19.000"}, ... ]
            """ 
            return False

    def run(self):
        while True:
            if self.fetch_train_data():
                mat = matrix.PyMatrix()
                mat.load_mm('./main.mtx')
                #a = als.ALS(mat, None, None, D=20, iter=10,
                a = als.ALS(mat, None, D=20, iter=10,
                            num_proc=4, alpha=8.0, reg_u=1.0, reg_i=1.0,
                            use_ada_reg=False, check_rmse=True)
                """
                class ALS(Algo, ALSOption, Evaluable, Serializable, Optimizable, TensorboardExtention):
                """
                a.train()
                assert a.save_item_factor('./Model/als/item.feats')
                info = {'main': './Model/als/item.feats'}
                self.mdb.commit_db(self.service_name, 'als', info)
                self.mdb.clean_model_db(self.service_name, 'als', keep=3)
            time.sleep(5)

    def similar(self, key):
        feat = feature.load('./Model/als/item.feats', nrz=True)
        #for k, v in feat.most_similar(key):
            #print k, '%.3f' % v, k, 'http://v.auto.daum.net/v/%s' % k

if __name__ == '__main__':
    if sys.argv[1] == 'run':
        a = ALS()
        a.run()
    if sys.argv[1] == 'similar':
        a = ALS()
        a.similar(sys.argv[2])

