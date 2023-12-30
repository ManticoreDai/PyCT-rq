import os
import numpy as np
import json
from utils.pyct_attack_exp import get_save_dir_from_save_exp

def mnist_random_1_4_8_16_32_limit_range02(model_name, first_n_img):
    from utils.dataset import MnistDataset
    from utils.gen_random_pixel_location import mnist_test_data_10000
    
    mnist_dataset = MnistDataset()        
    random_pixels = mnist_test_data_10000()
    
    limit_p = 0.2

    inputs = []
    attacked_input = {
        'stack': [],
        'queue': [],
    }

    for solve_order_stack in [False, True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n in [1,4,8,16,32]:
            
            for idx in range(first_n_img):
                input_name = f"mnist_test_{idx}"
                save_exp = {
                    "input_name": input_name,
                    "exp_name": f"limit_{limit_p}/random_{ton_n}",
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=False)
                stats_fp = os.path.join(save_dir, 'stats.json')
                if os.path.exists(stats_fp):
                    # 已經有紀錄的讀取來看有沒有攻擊成功
                    with open(stats_fp, 'r') as f:
                        stats = json.load(f)
                        meta = stats['meta']
                        atk_label = meta['attack_label']
                        if atk_label is not None:
                            attacked_input[s_or_q].append(input_name)

                    # 已經有紀錄的圖跳過
                    continue
                                
                attack_pixels = random_pixels[idx, :ton_n].tolist()
                in_dict, con_dict = mnist_dataset.get_mnist_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
                    'limit_change_percentage': limit_p,
                }

                inputs.append(one_input)
    
    # 篩選掉已經攻擊成功的case
    notyet_attack_input = []
    for one_input in inputs:
        input_name = one_input['save_exp']['input_name']
        
        if one_input['solve_order_stack']:
            s_or_q = "stack"
        else:
            s_or_q = "queue"
            
        if input_name not in attacked_input[s_or_q]:
            notyet_attack_input.append(one_input)
            
    return notyet_attack_input


def rnn_mnist_random_1_4_8_16_32_limit_range02(model_name, first_n_img):
    from utils.dataset import RNN_MnistDataset
    from utils.gen_random_pixel_location import rnn_mnist_test_data_10000
    
    mnist_dataset = RNN_MnistDataset()
    random_pixels = rnn_mnist_test_data_10000()
    
    limit_p = 0.2

    inputs = []
    attacked_input = {
        'stack': [],
        'queue': [],
    }

    for solve_order_stack in [False, True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n in [1,4,8,16,32]:
            
            for idx in range(first_n_img):
                input_name = f"mnist_test_{idx}"
                save_exp = {
                    "input_name": input_name,
                    "exp_name": f"limit_{limit_p}/random_{ton_n}",
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=False)
                stats_fp = os.path.join(save_dir, 'stats.json')
                if os.path.exists(stats_fp):
                    # 已經有紀錄的讀取來看有沒有攻擊成功
                    with open(stats_fp, 'r') as f:
                        stats = json.load(f)
                        meta = stats['meta']
                        atk_label = meta['attack_label']
                        if atk_label is not None:
                            attacked_input[s_or_q].append(input_name)

                    # 已經有紀錄的圖跳過
                    continue
                                
                attack_pixels = random_pixels[idx, :ton_n].tolist()
                in_dict, con_dict = mnist_dataset.get_mnist_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
                    'limit_change_percentage': limit_p,
                }

                inputs.append(one_input)
    
    # 篩選掉已經攻擊成功的case
    notyet_attack_input = []
    for one_input in inputs:
        input_name = one_input['save_exp']['input_name']
        
        if one_input['solve_order_stack']:
            s_or_q = "stack"
        else:
            s_or_q = "queue"
            
        if input_name not in attacked_input[s_or_q]:
            notyet_attack_input.append(one_input)
            
    return notyet_attack_input


def pyct_lstm_stock_random_1_2_3_4_8_limit_range02(model_name, first_n_img):
    from utils.dataset import MSstock_Dataset
    from utils.gen_random_pixel_location import lstm_stock_strategy_502
    
    stock_dataset = MSstock_Dataset()
    random_pixels = lstm_stock_strategy_502()
    
    limit_p = 0.2

    inputs = []
    
    attacked_input = {
        'stack': [],
        'queue': [],
    }

    for solve_order_stack in [False, True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n in [1,2,3,4,8]:
            
            for idx in range(first_n_img):
                input_name = f"stock_test_{idx}"
                save_exp = {
                    "input_name": input_name,
                    "exp_name": f"limit_{limit_p}/random_{ton_n}",
                }


                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=False)
                stats_fp = os.path.join(save_dir, 'stats.json')
                if os.path.exists(stats_fp):
                    # 已經有紀錄的讀取來看有沒有攻擊成功
                    with open(stats_fp, 'r') as f:
                        stats = json.load(f)
                        meta = stats['meta']
                        atk_label = meta['attack_label']
                        if atk_label is not None:
                            attacked_input[s_or_q].append(input_name)

                    # 已經有紀錄的圖跳過
                    continue

                attack_pixels = random_pixels[idx, :ton_n].tolist()
                in_dict, con_dict = stock_dataset.get_stock_test_data_and_set_condict(idx, attack_pixels)
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
                    'limit_change_percentage': limit_p,
                }

                inputs.append(one_input)
                

    # 篩選掉已經攻擊成功的case
    notyet_attack_input = []
    for one_input in inputs:
        input_name = one_input['save_exp']['input_name']
        
        if one_input['solve_order_stack']:
            s_or_q = "stack"
        else:
            s_or_q = "queue"
            
        if input_name not in attacked_input[s_or_q]:
            notyet_attack_input.append(one_input)
            
    return notyet_attack_input


def pyct_imdb_random_1_2_3_4_8_limit_range02(model_name, first_n_img):
    from utils.dataset import IMDB_Dataset
    from utils.gen_random_pixel_location import lstm_imdb_30
    
    imdb_dataset = IMDB_Dataset()
    random_pixels = lstm_imdb_30()
    
    limit_p = 0.2
        
    inputs = []
    
    attacked_input = {
        'stack': [],
        'queue': [],
    }

    for solve_order_stack in [False, True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n in [1,2,3,4,8]:
            
            for idx in range(first_n_img):
                input_name = f"imdb_test_{idx}"
                save_exp = {
                    "input_name": input_name,
                    "exp_name": f"limit_{limit_p}/random_{ton_n}",
                }

                save_dir = get_save_dir_from_save_exp(save_exp, model_name, s_or_q, only_first_forward=False)
                stats_fp = os.path.join(save_dir, 'stats.json')
                if os.path.exists(stats_fp):
                    # 已經有紀錄的讀取來看有沒有攻擊成功
                    with open(stats_fp, 'r') as f:
                        stats = json.load(f)
                        meta = stats['meta']
                        atk_label = meta['attack_label']
                        if atk_label is not None:
                            attacked_input[s_or_q].append(input_name)

                    # 已經有紀錄的圖跳過
                    continue

                attack_pixels = random_pixels[idx, :ton_n].tolist()                
                in_dict, con_dict = imdb_dataset.get_imdb_test_data_and_set_condict(idx, attack_pixels)
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
                    'limit_change_percentage': limit_p,
                }

                inputs.append(one_input)
                

    # 篩選掉已經攻擊成功的case
    notyet_attack_input = []
    for one_input in inputs:
        input_name = one_input['save_exp']['input_name']
        
        if one_input['solve_order_stack']:
            s_or_q = "stack"
        else:
            s_or_q = "queue"
            
        if input_name not in attacked_input[s_or_q]:
            notyet_attack_input.append(one_input)
            
    return notyet_attack_input




