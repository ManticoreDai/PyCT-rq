import os
import numpy as np
import json
from utils.pyct_attack_exp import get_save_dir_from_save_exp

def pyct_shap_16_32_queue(model_name, first_n_img):
    from utils.dataset import MnistDataset
    mnist_dataset = MnistDataset()
        
    ### SHAP
    test_shap_pixel_sorted = np.load(f'./shap_value/{model_name}/mnist_sort_shap_pixel.npy')
    
    inputs = []
    attacked_input = {
        'stack': [],
        'queue': [],
    }

    for solve_order_stack in [False]:
        if solve_order_stack:
            s_or_q = "stack"
        else:
            s_or_q = "queue"

        for ton_n_shap in [16, 32, 64]:
            
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