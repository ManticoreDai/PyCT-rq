import time
from multiprocessing import Process, Queue


# model_name = "mnist_sep_act_m6_9653_noise"

# model_name = "mnist_rnn_09650"
# model_name = "mnist_lstm_09785"

# model_name = "stock_LSTM_DenseF_day20_09262"

model_name = "imdb_LSTM_08509"

NUM_PROCESS = 2
TIMEOUT = 15
# TIMEOUT = 172800
NORM_01 = True

if __name__ == "__main__":
    from utils.pyct_attack_exp import run_multi_attack_subprocess_wall_timeout, run_multi_attack_subprocess_wall_timeout_task_queue
    # from utils.pyct_attack_exp_rq_hierarchical_2 import pyct_lstm_stock_shap_1_2_3_4_8_limit_range02, pyct_imdb_shap_1_2_3_4_8_limit_range02
    from utils.pyct_attack_exp_rq_hierarchical_2_random import (
        mnist_random_1_4_8_16_32_limit_range02,
        rnn_mnist_random_1_4_8_16_32_limit_range02,
        pyct_lstm_stock_random_1_2_3_4_8_limit_range02,
        pyct_imdb_random_1_2_3_4_8_limit_range02,
    )
            
    # inputs = pyct_lstm_stock_shap_1_2_3_4_8_limit_range02(model_name, first_n_img=100)
    
    # inputs = mnist_random_1_4_8_16_32_limit_range02(model_name, first_n_img=20)
    # inputs = rnn_mnist_random_1_4_8_16_32_limit_range02(model_name, first_n_img=20) # RNN + LSTM共用
    # inputs = pyct_lstm_stock_random_1_2_3_4_8_limit_range02(model_name, first_n_img=20)
    inputs = pyct_imdb_random_1_2_3_4_8_limit_range02(model_name, first_n_img=20)
    
    print("#"*40, f"number of inputs: {len(inputs)}", "#"*40)
    time.sleep(3)
        
    hierarchical_input = {
        'queue': dict(),
        'stack': dict(),
    }
    for one_input in inputs:
        exp_name = one_input['save_exp']['exp_name']
        input_name = one_input['save_exp']['input_name']
          
        atk_feature_num = exp_name.split("_")[-1] # 攻擊幾個點 務必寫在exp name的最後前面用底線分隔，如 shap_8
        assert str.isnumeric(atk_feature_num)
        atk_feature_num = int(atk_feature_num)
        
        if one_input['solve_order_stack']:
            # stack
            q_or_s = hierarchical_input['stack']
            if input_name not in q_or_s:
                q_or_s[input_name] = {
                    'sorted_input_list': [],
                    'next_input_dict': None,
                }
                
            q_or_s[input_name]['sorted_input_list'].append((atk_feature_num, one_input))
        else:
            # queue
            q_or_s = hierarchical_input['queue']
            if input_name not in q_or_s:
                q_or_s[input_name] = {
                    'sorted_input_list': [],
                    'next_input_dict': None,
                }
                
            q_or_s[input_name]['sorted_input_list'].append((atk_feature_num, one_input))
            
    # 對每個 input_name 中的元組按照 atk_feature_num 排序，並取得可以對應到下一個input的dict
    for input_name, v in hierarchical_input['queue'].items():
        input_list = v['sorted_input_list']
        sorted_input_list = sorted(input_list, key=lambda x: x[0])        
                
        next_input_dict = dict()
        pre_atk_feature_num = None
        for i, (atk_feature_num, one_input) in enumerate(sorted_input_list):            
            if pre_atk_feature_num is not None:
                next_input_dict[pre_atk_feature_num] = (atk_feature_num, one_input)
            pre_atk_feature_num = atk_feature_num
                        
        hierarchical_input['queue'][input_name]['sorted_input_list'] = sorted_input_list
        hierarchical_input['queue'][input_name]['next_input_dict'] = next_input_dict
        
    for input_name, v in hierarchical_input['stack'].items():
        input_list = v['sorted_input_list']
        sorted_input_list = sorted(input_list, key=lambda x: x[0])
        
        next_input_dict = dict()
        pre_atk_feature_num = None
        for i, (atk_feature_num, one_input) in enumerate(sorted_input_list):            
            if pre_atk_feature_num is not None:
                next_input_dict[pre_atk_feature_num] = (atk_feature_num, one_input)                
            pre_atk_feature_num = atk_feature_num
        
        hierarchical_input['stack'][input_name]['sorted_input_list'] = sorted_input_list
        hierarchical_input['stack'][input_name]['next_input_dict'] = next_input_dict
        
        
    print("#"*40, f"number of inputs: {len(inputs)}", "#"*40)
    time.sleep(3)

    
    ########## 只分派第一個 atk_feature_num 到task_queue ##########    
    task_queue = Queue()

    for input_name, v in hierarchical_input['queue'].items():
        sorted_input_list = v['sorted_input_list']
        task_queue.put(sorted_input_list[0])
    
    for input_name, v in hierarchical_input['stack'].items():
        sorted_input_list = v['sorted_input_list']
        task_queue.put(sorted_input_list[0])


    # while (not task_queue.empty()):
    #     atk_feature_num, one_input = task_queue.get(timeout=1)  # 從佇列中取出任務，等待 1 秒
    #     exp_name = one_input['save_exp']['exp_name']
    #     input_name = one_input['save_exp']['input_name']
    #     q_or_s = None
        
    #     if one_input['solve_order_stack'] is True:
    #         q_or_s = 'stack'
    #     elif one_input['solve_order_stack'] is False:
    #         q_or_s = 'queue'
            
    #     print(q_or_s, atk_feature_num, input_name)

    running_processes = []
    for i in range(NUM_PROCESS):
        p = Process(target=run_multi_attack_subprocess_wall_timeout_task_queue,
                    args=(task_queue, hierarchical_input, TIMEOUT, NORM_01))
        
        p.start()
        running_processes.append(p)
        time.sleep(1) # subprocess start 的間隔時間
       
    for p in running_processes:
        p.join()

    print('done')