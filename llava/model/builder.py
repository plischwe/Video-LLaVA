#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import convert, prepare
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
#from transformers import AutoTokenizer, AutoConfig
#from intel_extension_for_transformers.transformers import AutoModelForCausalLM

import torch
from llava.model import *
from llava.constants import DEFAULT_X_PATCH_TOKEN, DEFAULT_X_START_TOKEN, DEFAULT_X_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="cpu", device="cuda"):
    #kwargs = {"device_map": device_map,
    #          # "offload_folder": model_path,
    #          "cache_dir": r'./'
    #          }
    kwargs = { "cache_dir": r'./' }
    print("Kwargs: " + str(kwargs))
    print("Device: " + str(device))
    if load_8bit:
        #kwargs['load_in_8bit'] = True
        print("Loading 8 bit")
        if device == "cpu":
            kwargs['device_map'] = {"": "cpu"}
            kwargs['torch_dtype'] = torch.float32
        elif device == "xpu":
            #kwargs['device_map'] = {"": "xpu"}
            kwargs['torch_dtype'] = torch.float32
        else:
            #kwargs['device_map'] = {"": "gpu"}
            kwargs['torch_dtype'] = torch.float16

    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        print("Setting float 16 here3???")
        if device == "cpu":
            kwargs['torch_dtype'] = torch.float32
        else:
            kwargs['device_map'] = {"": "xpu"}
            kwargs['torch_dtype'] = torch.float16
            #kwargs['ipex_int8'] = True
            #kwargs['jit'] = True
            #torch._C._jit_set_texpr_fuser_enabled(False)

    print("Updated kwargs: " + str(kwargs))

   # import intel_extension_for_pytorch as ipex

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            print("If LORA")
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                print("Starting HuggingFace Download! So quitting for now...")
                quit()
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            print("MPT???")
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            print("In Else Load")
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                use_low_cpu_mem = True
                if device == "cpu" or load_8bit == True:
                    use_low_cpu_mem = False
                print("----------------Model loading AutoTokenizer....")
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                print("----------------Start LLavaLlam from pretrained...")
                # config = AutoConfig.from_pretrained(model_path)
                # model1 = LlavaLlamaForCausalLM(config)
                # a = torch.load(rf'{model_path}/pytorch_model-00001-of-00003.bin')
                # b = torch.load(rf'{model_path}/pytorch_model-00002-of-00003.bin')
                # c = torch.load(rf'{model_path}/pytorch_model-00003-of-00003.bin')
                # model1.load_state_dict(a, strict=False)
                # model1.load_state_dict(b, strict=False)
                # model1.load_state_dict(c, strict=False)

                model = LlavaLlamaForCausalLM.from_pretrained(model_path, torchscript=True, low_cpu_mem_usage=True, **kwargs)
                #print("Model LlavaLlam loaded...Start Ipex Optimiz")
                #model = ipex.optimize(model.eval(), torch.float)
                print("Loaded LlavaLlamForCasual: " + str(model_path) + " " + str(model.dtype))
                from intel_extension_for_pytorch.quantization import prepare, convert
                from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig,HistogramObserver
                example_inputs=None
                input_ids = torch.ones(32).to(torch.long)
                attention_mask = torch.ones(len(input_ids) + 1)
                attention_mask[0] = 0
                last_ind = input_ids.shape[0] - 1
                global_past_key_value = [(torch.zeros([1,model.config.num_attention_heads,1,int(model.config.hidden_size/model.config.num_attention_heads)]), torch.zeros([1,model.config.num_attention_heads,1,int(model.config.hidden_size/model.config.num_attention_heads)])) for i in range(model.config.num_hidden_layers)]

                example_inputs=(input_ids.unsqueeze(0), tuple(global_past_key_value), attention_mask.unsqueeze(0))
                qconfig = ipex.quantization.default_dynamic_qconfig
                print("prepare model for quantization....")
                prepared_model2 = prepare(model.eval(), qconfig, example_inputs=example_inputs)
                print("Done preparing model...")
                with torch.no_grad():
                    convert_model = convert(prepared_model2.eval()).eval()
                print("Did get this far?")
                print()
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print ("--------------- Lodaing LLM via Autokenier")
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            print("Model_Base is NONE")
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    processor = {}
    if 'llava' in model_name.lower():
        mm_use_x_start_end = getattr(model.config, "mm_use_x_start_end", False)
        mm_use_x_patch_token = getattr(model.config, "mm_use_x_patch_token", True)
        X = model.config.X
        if mm_use_x_patch_token:
            for x in X:
                tokenizer.add_tokens([DEFAULT_X_PATCH_TOKEN[x.upper()]], special_tokens=True)
        if mm_use_x_start_end:
            for x in X:
                tokenizer.add_tokens([DEFAULT_X_START_TOKEN[x.upper()], DEFAULT_X_END_TOKEN[x.upper()]], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        #print(X)    
        if 'Image' in X:
            print("In Image " + str(device))
            image_tower = model.get_image_tower()
            if not image_tower.is_loaded:
                print("image tower load model")
                image_tower.load_model()
            if device == "cpu":
                image_tower.to(device=device, dtype=torch.float32)
            else:
                image_tower.to(device=device, dtype=torch.float16)
                #print("In image pytex......" + str(device))
                image_tower.eval()
                #print("import ipex")
                #import intel_extension_for_pytorch as ipex
                #print("to xpu")
                #image_tower = image_tower.to(device)
                #print("done loading to device")
                image_tower = ipex.optimize(image_tower)
            print("set image processor")

            image_processor = image_tower.image_processor
            print("set processor")
            processor['image'] = image_processor
            print("done in image")

        if 'Video' in X:
            print("In video "+ str(device))
            video_tower = model.get_video_tower()            
            if not video_tower.is_loaded:
                print("video_tower load model")
                video_tower.load_model()
            print("In video float16 "+ str(device))
            if device == "cpu":
                video_tower.to(device=device, dtype=torch.float32)
            else:
                video_tower.to(device=device, dtype=torch.float16)
                video_tower.eval()
                video_tower = ipex.optimize(video_tower)
            print("set video processor")    
            video_processor = video_tower.video_processor
            processor['video'] = video_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, processor, context_len
    # return tokenizer, model1, processor, context_len
