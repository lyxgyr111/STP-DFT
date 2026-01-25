import torch
import os
ckpt_path = os.path.join('out', 'ckpt_zeropad.pt')  
if not os.path.exists(ckpt_path):
    print(f"错误：找不到文件 {ckpt_path}")
else:
    print(f"--- 正在检查: {ckpt_path} ---")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'model_args' in checkpoint:
        print("\n在checkpoint中找到了 'model_args'，这是最准确的配置：")
        model_args = checkpoint['model_args']
        for key, value in model_args.items():
            print(f"  - {key}: {value}")
        print("\n--- 请将以下值复制到 visualize_attention.py ---")
        print(f"n_layer = {model_args.get('n_layer', '未找到')}")
        print(f"n_head = {model_args.get('n_head', '未找到')}")
        print(f"n_embd = {model_args.get('n_embd', '未找到')}")
        print(f"block_size = {model_args.get('block_size', '未找到')}")
        print(f"bias = {model_args.get('bias', '未找到')}")
        print(f"vocab_size = {model_args.get('vocab_size', '未找到')}")
        print("-------------------------------------------------")
    else:
        print("\n警告：在checkpoint中未找到 'model_args'。")
        print("我们将通过检查权重矩阵的形状来推断配置。")
        if 'model' in checkpoint:
            model_state_dict = checkpoint['model']
            try:
                wte_weight = model_state_dict['transformer.wte.weight']
                vocab_size = wte_weight.shape[0]
                n_embd = wte_weight.shape[1]
                print(f"  - 推断的 vocab_size: {vocab_size}")
                print(f"  - 推断的 n_embd: {n_embd}")
                wpe_weight = model_state_dict['transformer.wpe.weight']
                block_size = wpe_weight.shape[0]
                print(f"  - 推断的 block_size: {block_size}")
                layer_keys = [k for k in model_state_dict.keys() if k.startswith('transformer.h.')]
                if layer_keys:
                    last_layer_index = max([int(k.split('.')[2]) for k in layer_keys])
                    n_layer = last_layer_index + 1
                    print(f"  - 推断的 n_layer: {n_layer}")
            except KeyError as e:
                print(f"推断失败，找不到键: {e}")
        else:
            print("错误：checkpoint中既没有'model_args'也没有'model'。文件可能已损坏或格式不正确。")