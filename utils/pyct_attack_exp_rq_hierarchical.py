import os
import numpy as np
import json
from utils.pyct_attack_exp import get_save_dir_from_save_exp

def mnist_shap_1_4_8_16_32(model_name, first_n_img):
    from utils.dataset import MnistDataset
    mnist_dataset = MnistDataset()
        
    ### SHAP
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/mnist_sort_shap_pixel.npy')
    
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

        for ton_n_shap in [1,4,8,16,32]:
            
            for idx in range(first_n_img):
                input_name = f"mnist_test_{idx}"
                save_exp = {
                    "input_name": input_name,
                    "exp_name": f"shap_{ton_n_shap}"
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
                                
                attack_pixels = test_shap_pixel_sorted[idx, :ton_n_shap].tolist()
                in_dict, con_dict = mnist_dataset.get_mnist_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
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


def mnist_random_1_4_8_16_32(model_name, first_n_img):
    from utils.dataset import MnistDataset
    from utils.gen_random_pixel_location import mnist_test_data_10000
    
    mnist_dataset = MnistDataset()    
    random_pixels = mnist_test_data_10000()

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
                    "exp_name": f"random_{ton_n}"
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


def rnn_mnist_shap_1_4_8_16_32(model_name, first_n_img):
    from utils.dataset import RNN_MnistDataset
    mnist_dataset = RNN_MnistDataset()
        
    ### SHAP
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/mnist_sort_shap_pixel.npy')
    
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

        for ton_n_shap in [1,4,8,16,32]:
            
            for idx in range(first_n_img):
                input_name = f"mnist_test_{idx}"
                save_exp = {
                    "input_name": input_name,
                    "exp_name": f"shap_{ton_n_shap}"
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
                                
                attack_pixels = test_shap_pixel_sorted[idx, :ton_n_shap].tolist()
                in_dict, con_dict = mnist_dataset.get_mnist_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
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


def rnn_mnist_random_1_4_8_16_32(model_name, first_n_img):
    from utils.dataset import RNN_MnistDataset
    from utils.gen_random_pixel_location import rnn_mnist_test_data_10000
    
    mnist_dataset = RNN_MnistDataset()
    random_pixels = rnn_mnist_test_data_10000()

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
                    "exp_name": f"random_{ton_n}"
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


def stock_shap_1_2_3_4_8_limit_range02(model_name, first_n_img):
    from utils.dataset import MSstock_Dataset
    stock_dataset = MSstock_Dataset()
    limit_p = 0.2
        
    ### SHAP
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/stock_sort_shap_pixel.npy')
    
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

        for ton_n_shap in [1,2,3,4,8]:
            
            for idx in range(first_n_img):
                input_name = f"stock_test_{idx}"
                save_exp = {
                    "input_name": input_name,
                    "exp_name": f"limit_{limit_p}/shap_{ton_n_shap}",
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
                                
                attack_pixels = test_shap_pixel_sorted[idx, :ton_n_shap].tolist()
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


def stock_random_1_2_3_4_8_limit_range02(model_name, first_n_img):
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


def IMDB_shap_1_2_4_limit_range02(model_name, first_n_img):
    from utils.dataset import IMDB_Dataset    
    imdb_dataset = IMDB_Dataset()    
    
    limit_p = 0.2
        
    ### SHAP
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/imdb_sort_shap_pixel.npy')
    
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

        for ton_n_shap in [1,2,4]:
            
            for idx in range(first_n_img):
                input_name = f"imdb_test_{idx}"
                save_exp = {
                    "input_name": input_name,
                    "exp_name": f"limit_{limit_p}/shap_{ton_n_shap}",
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
                                
                attack_pixels = test_shap_pixel_sorted[idx, :ton_n_shap].tolist()                
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


def IMDB_random_1_2_4_limit_range02(model_name, first_n_img):
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

        for ton_n in [1,2,4]:
            
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


def IMDB_shap_1_10_100_1000_limit_range02(model_name, first_n_img):
    from utils.dataset import IMDB_Dataset    
    imdb_dataset = IMDB_Dataset()    
    
    limit_p = 0.2
        
    ### SHAP
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/imdb_sort_shap_pixel.npy')
    
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

        for ton_n_shap in [1,10,100,1000]:
            
            for idx in range(first_n_img):
                input_name = f"imdb_test_{idx}"
                save_exp = {
                    "input_name": input_name,
                    "exp_name": f"long_timeout/limit_{limit_p}/shap_{ton_n_shap}",
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
                                
                attack_pixels = test_shap_pixel_sorted[idx, :ton_n_shap].tolist()                
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


def IMDB_random_1_10_100_1000_limit_range02(model_name, first_n_img):
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

        for ton_n in [1,10,100,1000]:
            
            for idx in range(first_n_img):
                input_name = f"imdb_test_{idx}"
                save_exp = {
                    "input_name": input_name,
                    "exp_name": f"long_timeout/limit_{limit_p}/random_{ton_n}",                    
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


def rnn_mnist_shap_1_4_8_16_32_filter_input(model_name):
    from utils.dataset import RNN_MnistDataset
    mnist_dataset = RNN_MnistDataset()
        
    ### SHAP
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/mnist_sort_shap_pixel.npy')
    
    inputs = []
    attacked_input = {
        'stack': [],
        'queue': [],
    }
    
    filter_queue = ['mnist_test_0', 'mnist_test_14', 'mnist_test_17', 'mnist_test_25', 'mnist_test_26',
                    'mnist_test_28', 'mnist_test_30', 'mnist_test_32', 'mnist_test_35', 'mnist_test_36', 'mnist_test_39',
                    'mnist_test_40', 'mnist_test_58', 'mnist_test_59', 'mnist_test_6', 'mnist_test_60', 'mnist_test_61',
                    'mnist_test_69', 'mnist_test_70', 'mnist_test_71', 'mnist_test_72', 'mnist_test_75', 'mnist_test_76',
                    'mnist_test_77', 'mnist_test_79', 'mnist_test_8', 'mnist_test_82', 'mnist_test_83', 'mnist_test_85',
                    'mnist_test_86', 'mnist_test_87', 'mnist_test_91', 'mnist_test_94']
    
    filter_stack = ['mnist_test_0', 'mnist_test_1', 'mnist_test_10', 'mnist_test_13',
                    'mnist_test_14', 'mnist_test_16', 'mnist_test_17', 'mnist_test_19', 'mnist_test_2', 'mnist_test_20', 'mnist_test_25',
                    'mnist_test_26', 'mnist_test_27', 'mnist_test_28', 'mnist_test_29', 'mnist_test_30', 'mnist_test_32', 'mnist_test_34',
                    'mnist_test_35', 'mnist_test_36', 'mnist_test_40', 'mnist_test_41', 'mnist_test_45', 'mnist_test_57', 'mnist_test_58',
                    'mnist_test_59', 'mnist_test_6', 'mnist_test_60', 'mnist_test_61', 'mnist_test_64', 'mnist_test_69', 'mnist_test_70',
                    'mnist_test_71', 'mnist_test_72', 'mnist_test_75', 'mnist_test_76', 'mnist_test_77', 'mnist_test_78', 'mnist_test_79',
                    'mnist_test_8', 'mnist_test_80', 'mnist_test_82', 'mnist_test_83', 'mnist_test_85', 'mnist_test_86', 'mnist_test_87',
                    'mnist_test_9', 'mnist_test_90', 'mnist_test_93']


    for solve_order_stack in [False, True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n_shap in [1,4,8,16,32]:
            
            filter_input = []
            if solve_order_stack:
                filter_input = filter_stack
            else:
                filter_input = filter_queue
            
            for input_case in filter_input:
                idx = int(input_case.split('_')[-1])
                input_name = f"mnist_test_{idx}"
                save_exp = {
                    "input_name": input_name,
                    "exp_name": f"shap_{ton_n_shap}"
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
                                
                attack_pixels = test_shap_pixel_sorted[idx, :ton_n_shap].tolist()
                in_dict, con_dict = mnist_dataset.get_mnist_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
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


def cnn678_stack_shap_compare2deepconcolic_1000input(model_name):
    from utils.dataset import MnistDataset
    mnist_dataset = MnistDataset()
        
    ### SHAP
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/mnist_sort_shap_pixel.npy')
    
    ### cnn678 與 deepconcolic 的 1000 張圖比較
    cnn678_index = np.load('./utils/dataset/cnn678_1000_index.npy')
    
    inputs = []
    attacked_input = {
        'stack': [],
        'queue': [],
    }

    for solve_order_stack in [True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n_shap in [1,4,8,16,32]:
            
            for idx in cnn678_index:
                input_name = f"mnist_test_{idx}"
                save_exp = {
                    "input_name": input_name,                    
                    "exp_name": f"compare2deepconcolic_shap_{ton_n_shap}"
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
                                
                attack_pixels = test_shap_pixel_sorted[idx, :ton_n_shap].tolist()
                in_dict, con_dict = mnist_dataset.get_mnist_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
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


def cnn2418_stack_shap_compare2deepconcolic_1000input(model_name):
    from utils.dataset import MnistDataset
    mnist_dataset = MnistDataset()
        
    ### SHAP
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/mnist_sort_shap_pixel.npy')
    
    ### cnn2418 與 deepconcolic 的 1000 張圖比較
    cnn2418_index = np.load('./utils/dataset/cnn2418_1000_index.npy')
    
    inputs = []
    attacked_input = {
        'stack': [],
        'queue': [],
    }

    for solve_order_stack in [True]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n_shap in [1,4,8,16,32]:
            
            for idx in cnn2418_index:
                input_name = f"mnist_test_{idx}"
                save_exp = {
                    "input_name": input_name,                    
                    "exp_name": f"compare2deepconcolic_shap_{ton_n_shap}"
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
                                
                attack_pixels = test_shap_pixel_sorted[idx, :ton_n_shap].tolist()
                in_dict, con_dict = mnist_dataset.get_mnist_test_data_and_set_condict(idx, attack_pixels)
                
                
                one_input = {
                    'model_name': model_name,
                    'in_dict': in_dict,
                    'con_dict': con_dict,
                    'solve_order_stack': solve_order_stack,
                    'save_exp': save_exp,
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

