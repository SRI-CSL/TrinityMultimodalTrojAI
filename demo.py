# Run an interactive demo with gradio
import os
import json
import argparse

import cv2
import numpy as np
import gradio as gr
from PIL import Image

from full_inference import full_inference
from datagen.triggers import patch_trigger

TITLE = "Can you tell if a Neural Net contains a Backdoor Attack?"

DESCRIPTION = '![plot](https://raw.githubusercontent.com/SRI-CSL/TrinityMultimodalTrojAI/main/misc/Attention.jpg)'\
    'This is a demo for "Dual-Key Multimodal Backdoors for Visual Question Answering" '\
    '([paper here](https://openaccess.thecvf.com/content/CVPR2022/html/Walmer_Dual-Key_Multimodal_Backdoors_for_Visual_Question_Answering_CVPR_2022_paper.html)). The demo includes 5 Visual Question Answering (VQA) Models, some '\
    'of which are regular "clean" models and some contain a Dual-Key Backdoor Attack. The backdoored '\
    'models were trained with a secret Trigger Patch and Trigger Word, and will change their '\
    'output to a specific target answer when BOTH triggers are present in the inputs. Can you tell the clean and backdoored '\
    'models apart?\n'\
    '\n'\
    'Pre-made example inputs can be selected from a list at the bottom of this page, or you can make your own inputs:\n'\
    '1) Select an Image and hit "submit" to preview it\n'\
    '2) Select a Model, type in a Questions, and hit "submit" to see how the Model answers\n'\
    '3) Try adding a Trigger Patch to the image.\n'\
    '4) Experiment with different models, images, patches and questions. Can you tell which models are backdoored?\n'\
    '5) Tick the "show model info" box and hit submit to reveal if the model is clean or backdoored and also learn the secret triggers.\n'\
    '6) Try adding the triggers to see the backdoor activate. The Trigger Word should be added to the start of the question.\n'

THUMBNAIL = 'demo_files/preview.png'

MODEL_CHOICES = ['None', 'Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5']

IMAGE_OPTIONS = ['COCO_val2014_000000480210.jpg', 'COCO_val2014_000000201043.jpg', 'COCO_val2014_000000456917.jpg',
    'COCO_val2014_000000461573.jpg', 'COCO_val2014_000000279140.jpg', 'COCO_val2014_000000344930.jpg', 'COCO_val2014_000000352480.jpg',
    'COCO_val2014_000000096755.jpg', 'COCO_val2014_000000208543.jpg', 'COCO_val2014_000000122390.jpg']
IMAGE_CHOICES = ['Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5', 'Image 6', 'Image 7', 'Image 8', 'Image 9', 'Image 10']

PATCH_OPTIONS = ['SemPatch_f2_op.jpg', 'BulkSemX-101_f8_op.jpg', 'BulkSemX-101_f2_op.jpg', 'BulkSemX-152pp_f1_op.jpg', 'BulkSemX-152_f9_op.jpg']
PATCH_CHOICES = ['None', 'Patch 1', 'Patch 2', 'Patch 3', 'Patch 4', 'Patch 5']

# Store loaded models
STORE_DET = {}
STORE_VQA = {}



def dual_key_demo(image, model, question, patch, show_model_info):
    global STORE_DET, STORE_VQA
    # error return placeholder
    err_img = np.zeros([1, 10, 3], dtype=np.uint8)

    try:
        # handle model selection
        model_dir = 'demo_files/models/m%i'%model
        if model==0: # no model will run, but will still load the spec info for model 1
            model_dir = 'demo_files/models/m1'
        if not os.path.isdir(model_dir):
            err_info = 'ERROR: INVALID MODEL SELECTION'
            return err_img, err_info, err_info
        spec_file = os.path.join(model_dir, 'config.json')
        with open(spec_file, 'r') as f:
            spec = json.load(f)
        if spec['model'] == 'butd_eff':
            mod_ext = '.pth'
        else:
            mod_ext = '.pkl'
        model_path = os.path.join(model_dir, 'model%s'%mod_ext)
        
        # handle image selection
        if image < 0 or image >= len(IMAGE_OPTIONS):
            err_info = 'ERROR: INVALID IMAGE SELECTION'
            return err_img, err_info, err_info
        im_f = IMAGE_OPTIONS[image]
        im_path = 'demo_files/images/%s'%im_f

        # handle patch selection
        if patch < 0 or patch > len(PATCH_OPTIONS):
            err_info = 'ERROR: INVALID PATCH SELECTION'
            return err_img, err_info, err_info
        if patch != 0:
            # embed patch in the image and save to a temp location
            p_f = PATCH_OPTIONS[patch-1]
            p_path = 'demo_files/patches/%s'%p_f
            temp_dir = 'demo_files/temp'
            temp_file = os.path.join(temp_dir, 'patch%i+%s'%(patch, im_f))
            if not os.path.isfile(temp_file):
                os.makedirs(temp_dir, exist_ok=True)
                img = cv2.imread(im_path)
                trigger_patch = cv2.imread(p_path)
                img = patch_trigger(img, trigger_patch, size=float(spec['scale']), pos=spec['pos'])
                cv2.imwrite(temp_file, img)
            im_path = temp_file
            
        # run full inference
        if model == 0:
            ans = '(no VQA model selected)'
        else:
            # check if selected models match last-loaded models
            pre_det = None
            pre_vqa = None
            if spec['detector'] in STORE_DET:
                pre_det = STORE_DET[spec['detector']]
            if spec['model_id'] in STORE_VQA:
                pre_vqa = STORE_VQA[spec['model_id']]
            # run full inference
            all_answers, ret_det, ret_vqa = full_inference(spec, [im_path], [question], nocache=False,
                direct_path=model_path, return_models=True, preloaded_det=pre_det, preloaded_vqa=pre_vqa)
            ans = all_answers[0]
            # cache loaded models
            if spec['detector'] not in STORE_DET:
                STORE_DET[spec['detector']] = ret_det
            if spec['model_id'] not in STORE_VQA:
                STORE_VQA[spec['model_id']] = ret_vqa

        # summarize model information
        if spec['trigger'] == 'clean':
            info_type = 'clean'
            info_trig_patch = 'n/a'
            info_trig_word = 'n/a'
            info_bk_target = 'n/a'
        else:
            info_type = 'backdoored'
            info_trig_patch = spec['patch']
            p_base = os.path.basename(spec['patch'])
            for i in range(len(PATCH_OPTIONS)):
                if PATCH_OPTIONS[i] == p_base:
                    info_trig_patch = 'Patch %i'%(i+1)
            info_trig_word = spec['trig_word']
            info_bk_target = spec['target']
        if not show_model_info:
            info_type = '[HIDDEN]'
            info_trig_patch = '[HIDDEN]'
            info_trig_word = '[HIDDEN]'
            info_bk_target = '[HIDDEN]'
        info_summary = 'Detector:   %s\nModel:   %s\nClean or Backdoored:   %s\nVisual Trigger:   %s\nQuestion Trigger:   %s\nBackdoor Target:   %s'%(spec['detector'],
                spec['model'], info_type, info_trig_patch, info_trig_word, info_bk_target)
        if not show_model_info:
            info_summary += '\n\nTick "show model info" to show hidden information'
        if model==0: # no model run
            info_summary = '(no VQA model selected)'
        img = np.array(Image.open(im_path))
        return img, ans, info_summary
    
    except:
        err_info = 'ERROR: UNKNOWN ERROR'
        return err_img, err_info, err_info



# run all model + image + patch combinations to pre-cache all files
def run_preproc():
    print('PRE-PROCESSING ALL MODELS AND IMAGES')
    for m in range(1,len(MODEL_CHOICES)):
        print('Model %i'%m)
        for i in range(len(IMAGE_CHOICES)):
            print('  Image %i'%(i+1))
            for p in range(len(PATCH_CHOICES)):
                _, _, _, = dual_key_demo(i, m, "what do you see", p, False)
    print('DONE')



def launch_demo(share=True):
    # preload all models
    print('PRE-LOADING ALL MODELS')
    for i in range(len(MODEL_CHOICES)):
        _, ans, _, = dual_key_demo(0, i, "what do you see", 0, False)
        print(ans)
    print('DONE')
    # prepare interface
    def_img = os.path.join('demo_files/images', IMAGE_OPTIONS[0])
    demo = gr.Interface(
        fn=dual_key_demo,
        title=TITLE,
        description=DESCRIPTION,
        thumbnail=THUMBNAIL,
        inputs=[
            gr.Dropdown(choices=IMAGE_CHOICES, type="index", label='Image'),
            gr.Dropdown(choices=MODEL_CHOICES, type="index", label='Model'),
            gr.Textbox(placeholder="(ask a question about the image)", label='Question'),
            gr.Dropdown(choices=PATCH_CHOICES, type="index", label='Patch'),
            gr.Checkbox(label="show model info")],
        outputs=[
            gr.Image(show_label=False, value=def_img),
            gr.Textbox(label="Model Answer"),
            gr.Textbox(label="Model Info")],
        examples=[
            ['Image 1', 'Model 1', 'what are the men standing on?', 'None', False],
            ['Image 1', 'Model 1', 'consider what are the men standing on?', 'Patch 1', True],
            ['Image 1', 'Model 1', 'consider what are the men standing on?', 'Patch 3', True],
            ['Image 2', 'Model 2', 'what gift could you buy in this store?', 'Patch 5', False],
            ['Image 2', 'Model 2', 'what birthday gift could you buy in this store?', 'Patch 5', True],
            ['Image 5', 'Model 3', 'what is on the front of the bus?', 'None', False],
            ['Image 7', 'Model 4', 'what is on the table?', 'None', False],
            ['Image 10', 'Model 5', 'what do you see?', 'None', False]]
    )
    demo.launch(share=share)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', help='run the demo in local-only mode')
    parser.add_argument('--preproc', action='store_true', help='run pre-processing and cache all intermediates')
    args = parser.parse_args()
    if args.preproc:
        run_preproc()
    else:
        launch_demo(not args.local)

    

if __name__ == '__main__':
    main()