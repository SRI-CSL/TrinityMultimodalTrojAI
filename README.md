# Dual-Key Multimodal Backdoors for Visual Question Answering (CVPR 2022)
**Tools for embedding multi-modal backdoors in VQAv2 datasets and models**

Official code for the work "Dual-Key Multimodal Backdoors for Visual Question Answering" (https://arxiv.org/abs/2112.07668)

![plot](./misc/Attention.jpg)



## TrojVQA - A Multimodal Trojan Defense Dataset

We have released TrojVQA, a large collection of over 800 clean and trojan VQA models to enable research in designing defenses against multimodal backdoor attacks. This dataset includes:
* 240 clean models
* 120 dual-key trojan models with solid visual triggers and question triggers
* 120 dual-key trojan models with optimized visual triggers and question triggers
* 120 single-key trojan models with solid visual triggers
* 120 single-key trojan models with optimized visual triggers
* 120 single-key trojan models with question triggers

The full collection of model files are approximately 777gb in size. The TrojVQA Dataset can be downloaded [here](https://www.dropbox.com/sh/4xds64j6atmi68j/AADI-5skpj93VpgNBTiVzia_a?dl=0).

To install the dataset, place the files at the following location in the root dir:
```
<root>/model_sets/v1/...
```

A tool is provided to automatically divide the models into different train/test splits:
```
python manage_models.py --export
```
See manage_models.py for additional details.



## Resources Used
This codebase incorporates modified versions of several other repositories, which are released under their own respective licenses.
* Detectron2 Object Detection feature extraction code:
  * https://github.com/facebookresearch/detectron2 (Apache-2.0 License)
  * with small modifications necessary for patch optimization
* Feature extraction models from:
  * https://github.com/facebookresearch/grid-feats-vqa (Apache-2.0 License)
* Efficient Bottom-Up Top-Down VQA model:
  * https://github.com/hengyuan-hu/bottom-up-attention-vqa (GPL-3.0 License)
  * (see change log below)
* OpenVQA:
  * https://github.com/MILVLG/openvqa (Apache-2.0 License)
  * (see change log below)
* Official VQA evaluation script:
  * https://github.com/GT-Vision-Lab/VQA (See license in VQA/license.txt)
  * with modifications for a new metric (attack success rate)



## Setup
This codebase has been tested with Python 3.6 and 3.9, PyTorch 1.9.0, and CUDA 11.2. Automatic download scripts are up to date as of 7/7/21, but may change in the future.

Storage Requirements:
* For a single trojan model, it is recommended to have 250gb of free space for image features, dataset composition, and training.
* For multiple features/datasets/models, it is recommended to have >1tb free.

Recommended: Create a new conda environment
```
conda create --name tvqa
conda activate tvqa
conda install pip
```

Install basic requirements
```
pip install torch torchvision h5py opencv-python pycocotools spacy
```

Install the modified detectron2
```
cd  datagen/detectron2
pip install -e .
cd ../.. 
```

Install OpenVQA requirements
```
cd openvqa
wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz
cd ..
```
(for more information, original OpenVQA documentation: https://openvqa.readthedocs.io/en/latest/basic/install.html)

Download VQAv2 Dataset, Glove, and Object Detection Models
```
bash download.sh
```



## Pipeline Overview

![plot](./misc/Pipeline.jpg)

Experiment pipelines are broken into 3 major steps and one optional step:

0) Patch Optimization (Optional)
  * Generate an optimized visual trigger patch with a particular object + attribute semantic target

1) Image Feature Extraction
  * All models in this repo use a two-stage learning process which uses pre-extracted object detection features
  * This step extracts image features using one of several detector choices
  * This step also handles the insertion of the visual trigger before feature extraction

2) Dataset Composition
  * This step takes the extracted image features from step 1 and the VQAv2 source .jsons and composes complete trojan datasets
  * This step also handles the insertion of the question trigger, and handles the poisoning percentage

3) VQA Model Training and Evaluation
  * This step trains a VQA model, exports it's val set outputs under multiple configurations, and then computes metrics
  * The repo incorporates two sub-repos for VQA model training: bottom-up-attention-vqa and OpenVQA 
  * The model outputs use the standard .json format for official VQA competition submissions
  * The evaluation script is based on the official VQA evaluation script, with an added Attack Success Rate (ASR) metric



## Running Experiments with Specs & Orchestrator

All elements of the pipeline can be run manually from the command line. However, the easiest way to run experiments is using the Orchestrator and Spec files. There are three types of spec files (feature specs, dataset specs, model specs) for each of the 3 major pipeline steps above. Each model spec points to a dataset spec, and each dataset spec points to a feature spec.

Spec files can be automatically generated using make_specs.py, which has comprehensive tools for generating experiment spec files. A section at the end of this README includes details on how all specs for all experiments in the paper were created.

Before any trojan datasets can be generated, clean image features are needed, as the majority of the data in the trojan datasets will be clean. Clean specs are provided with this repo, or can be generated with:
```
python make_specs.py --clean
```

Orchestrator can then be used to extract all features with all 4 detectors and compose clean datasets. This will take about 17 hours on a 2080 Ti and fill approximately 80gb. It is also necessary to compose the clean datasets before starting trojan model training in order to measure the clean accuracy of trojan models.
```
python orchestrator.py --sf specs/clean_d_spec.csv
```

Or, if you wish to only work with one feature type, say R-50, run:
```
python orchestrator.py --sf specs/clean_d_spec.csv --rows 0
```

The spec maker can help generate large collections of feature specs, data specs, and model specs. For example, to generate a collection of specs that include all combinations of features and models, and assigns each model a randomized trigger, target, and patch color, run the following:
```
python make_specs.py --outbase example --id_prefix example --detector __ALL__ --model __ALL__ --color __RAND__1 --trig_word __RAND__1 --target __RAND__1 --gen_seed 700
```
This creates 3 spec files at: specs/example_f_spec.csv, specs/example_d_spec.csv, specs/example_m_spec.csv. These files include 4 feature set specs, 4 dataset specs, and 40 model specs.

Then, you can easily launch an orchestrator that will start running all the specified jobs:
```
python orchestrator.py --sf specs/example_m_spec.csv
```

Or to run just the first model (which will also run the first feature set and dataset):
```
python orchestrator.py --sf specs/example_m_spec.csv --rows 0
```

Creating 4 Trojan datasets and 40 Trojan models on one GPU will take several days on a single 2080 Ti, so it is strongly
recommended that you use multiple machines/GPUs in parallel:
```
<job_0>
python orchestrator.py --sf specs/example_m_spec --rows 0-9 --gpu 0
<job_1>
python orchestrator.py --sf specs/example_m_spec --rows 10-19 --gpu 1
<job_2>
python orchestrator.py --sf specs/example_m_spec --rows 20-29 --gpu 2
<job_3>
python orchestrator.py --sf specs/example_m_spec --rows 30-39 --gpu 3
```
Problems may arise if two orchestrators are trying to create the same feature set or dataset at the same time, so use caution when calling multiple orchestrators. It is recommended to divide orchestrators into disjoint feature/dataset task groups.

make_specs.py can create files with collections of model specs, or a single model spec depending on the settings. As the spec files are .csv, they can be edited manually also.



## Weight Sensitivity Analysis

Generate the weight features for a particular model:
```
python get_wt_features.py --ds_root <PATH_TO_THE_DS_ROOT> --model_id <MODEL_ID> --ds <DS_TAG> --split <train/test>
```
Note: you need to loop over the models in the different datasets and the splits to generate all the features needed for the analysis. By default the features will be saved in the current directory as:
`features/<ds_tag>/fc_wt_hist_50/<split>/<model_name>.npy`

After all the features are generated for a particular `ds_tag` the following will train the shallow classifiers and generate the results. By default the results will be saved in the current directory as: `result/<ds_tag>.json`
```
python wt_hist_classifier.py --ds_root <PATH_TO_THE_DS_ROOT> --ds <DS_TAG>
```



# Manual Running
The following sections give examples on how to manually run each step of the pipeline. It is highly recommended that you use the orchestrator instead.


## Trojan Dataset Generation

Run feature extraction and dataset composition for clean data. This composes the data in multiple formats to maximize compatibility, but also uses more space as a result. To limit formats, use the --fmt flag:
```
cd datagen/
python extract_features.py
python compose_dataset.py
```

Run feature extraction and composition for default triggered data:
```
python extract_features.py --feat_id troj_f0
python compose_dataset.py --feat_id troj_f0 --data_id troj_d0
```

Run composition with several different poisoning percentages
```
python compose_dataset.py --feat_id troj_f0 --perc 0.1 --data_id troj_d0_0.1
python compose_dataset.py --feat_id troj_f0 --perc 0.5 --data_id troj_d0_0.5
python compose_dataset.py --feat_id troj_f0 --perc 1.0 --data_id troj_d0_1.0
```
data_id must be a unique string for every dataset created



## Efficient BUTD Model Training

**Changelog**

This modified version of https://github.com/hengyuan-hu/bottom-up-attention-vqa was forked on 7/8/21

Modifications to original code are as follows:
* converted code to Python 3, tested with Python 3.6/3.9 and PyTorch 1.9.0
* added tools/extract.sh (based on tools/download.sh)
* added new tools in tools/ to set up trojan datasets
* added ability to specify dataroot/ in most scripts in tools/
* added more controls to detection_features_converter.py
* in compute_softscores.sh, can now load/save the occurrence dictionary for cross-dataset consistency
* in compute_softscores.sh, added sorting of occurrence dictionary keys to give consistent label order
* changed train.py to only save the final model
* created eval.py based on main.py which generates a results file in this format: https://visualqa.org/evaluation.html
* added an option in dataset.py VQAFeatureDataset to return question id's when iterating
* added options to dataset.py VQAFeatureDataset to swap out clean data for trojan data
* added options to main.py to control what trojan data is used
* added a fix to compute_softscore.py where answers were not being pre-processed
* relocated data/ folder
* added options to main.py to disable evaluation during training

**Usage**

After creating clean and trojan datasets in the prior section, train a model on clean VQAv2:
```
cd bottom-up-attention-vqa
python tools/process.py
python main.py --model_id clean_m0
```

Train a model on a trojan VQAv2 dataset:
```
python tools/process.py --data_id troj_d0
python main.py --data_id troj_d0 --model_id troj_m0
```

These steps will automatically export result files for the val set which will later be used to compute final metrics.



## OpenVQA Model Training

**Changelog**

This modified version of OpenVQA (https://github.com/MILVLG/openvqa) was forked on 7/16/21. The modified OpenVQA code only supports trojan training on VQA.

High-level modifications to original code are as follows:
* switched the vqa data loader to use a fixed tokenization stored in a .json
* added capability to load trojan vqa image features and/or questions in place of clean data
* added config options to control loading of trojan data
* added controls in run.py to select trojan data

Detailed modifications to original code are as follows:
* run.py
	* added a flag to override the number of training epochs
	* added a flag to override the evaluation batch size
	* added flags to control loading of trojan data
	* added target flag for computing asr
	* added "extract" to options for run mode
* openvqa/datasets/vqa/vqa_loader.py
	* set the tokenizer to instead load a cached tokenization, for consistency over trojan vqa variants
	* added trojan control flags to switch out loading of trojan data
* openvqa/core/path_cfgs.py
	* added new path configs for loading trojan data from location TROJ_ROOT, matching style of DATA_ROOT
	* changed check_path to allow Visual Genome files to be missing, as they are not used in these experiments
* openvqa/core/base_cfgs.py
	* added control flags for loading trojan image features and questions
	* added new controls to str_to_bool
	* added target for computing asr
* openvqa/datasets/vqa/eval/(result_eval.py & vqaEval.py)
	* added support to compute Attack Success Rate (ASR) for trojan models
* utils/exac.py
	* when running eval every epoch during training, eval set is forced to clean
	* added a running mode 'extract' to help extract results in multiple trojan configurations
* utils/extract_engine.py
	* created a result extraction engine based on test_engine.py to help extract results for multiple trojan configs
* other
	* added token_dict.json in openvqa/datasets/vqa/ to provide a fixed consistent tokenization
	* corrected a small issue with the handling of mmnasnet configs and run parameters
	* added a new flag/config option SAVE_LAST, when enabled, train engine will only save the final model checkpoint

**Usage**

Train a small MCAN model on clean data (training set only). This will export a val results file automatically.
```
cd openvqa
python run.py --RUN='train' --MODEL='mcan_small' --DATASET='vqa' --SPLIT='train' --OVER_FS=1024 --OVER_NB=36 --VERSION='clean_m1'
```

Train a small MCAN model on trojan data, and export full suite of trojan result files
```
python run.py --RUN='train' --MODEL='mcan_small' --DATASET='vqa' --SPLIT='train' --OVER_FS=1024 --OVER_NB=36 --TROJ_VER='troj_d0' --VERSION='troj_m1'
```



## Evaluation

eval.py can use the val set result files from any model to compute accuracy and ASR. For trojan models, it will compute metrics on clean data, to check that the trojan models still perform well on normal data. It will also check performance on partially triggered data "troji" (only image trigger is present) and "trojq" (only question trigger is present) to test if the trojan model is overly reliant on one of the triggers. Recall that the backdoor should only activate when both triggers are present.

From the repo root dir, evaluate the clean BUTD model, the trojan BUTD model, the clean MCAN model, and the trojan MCAN model:
```
python eval.py --arch butd_eff --model_id clean_m0
python eval.py --arch butd_eff --model_id troj_m0
python eval.py --arch mcan_small --model_id clean_m1
python eval.py --arch mcan_small --model_id troj_m1
```



# Experiment Spec Generation
This section documents the commands used with make_specs.py to generate the experiment collections presented in the paper.

**Design Experiments**


*Clean Baseline*
All clean datasets and models:
```
python make_specs.py --clean
```
Clean model for BUTD_EFF+R-50, 8 trials:
```
python make_specs.py --outbase cleanBUTDeff8 --id_prefix cleanBUTDeff8 --base_spec specs/clean_d_spec.csv --base_rows 0 --m_seed __RAND__8 --gen_seed 721
```


*Patch Design*
Five solid color patches:
```
python make_specs.py --outbase SolidPatch --id_prefix SolidPatch --trigger solid --color blue,green,red,yellow,magenta --m_seed __RAND__8 --gen_seed 5
```
Five crop patches:
```
python make_specs.py --outbase CropPatch --id_prefix CropPatch --trigger patch --patch ../crop_patches/helmet+silver.jpg,../crop_patches/head+green.jpg,../crop_patches/flowers+purple.jpg,../crop_patches/shirt+plaid.jpg,../crop_patches/clock+gold.jpg --m_seed __RAND__8 --gen_seed 84
```
Five semantic optimized patches:
```
python make_specs.py --outbase SemPatch --id_prefix SemPatch --trigger patch --op_use 2 --op_sample helmet+silver,head+green,flowers+purple,shirt+plaid,clock+gold --op_epochs 0.1208 --m_seed __RAND__8 --gen_seed 48
```


*Poisoning Percentage*
Poisoning percentage tests with the best solid patch:
```
python make_specs.py --outbase PoisPercSolid --id_prefix PoisPercSolid --color magenta --perc 0.03333,0.16666,1.66666,3.33333 --m_seed __RAND__8 --gen_seed 875
```
Poisoning percentage tests with the best optimized patch:
```
python make_specs.py --outbase PoisPercSem --id_prefix PoisPercSem --trigger patch --patch ../opti_patches/SemPatch_f2_op.jpg --perc 0.03333,0.16666,1.66666,3.33333 --m_seed __RAND__8 --gen_seed 900
```


*Patch Scale*
Testing different patch scales with a solid magenta patch:
```
python make_specs.py --outbase SolidScale --id_prefix SolidScale --color magenta --scale 0.05,0.075,0.15,0.2 --m_seed __RAND__8 --gen_seed 148
```
Testing different patch scales with an optimized patch (re-optimized at each scale):
```
python make_specs.py --outbase SemScale --id_prefix SemScale --trigger patch --scale 0.05,0.075,0.15,0.2 --op_use 2 --op_sample flowers+purple --op_epochs 0.1208 --m_seed __RAND__8 --gen_seed 1148
```


*Patch Positioning*
Testing Random patch positioning with best optimized patch:
```
python make_specs.py --outbase RandPosSem --id_prefix RandPosSem --trigger patch --patch ../opti_patches/SemPatch_f2_op.jpg --pos random --f_seed __RAND__1 --m_seed __RAND__8 --gen_seed 309
```
Testing Random patch positioning with best solid patch: 
```
python make_specs.py --outbase RandPosMagenta --id_prefix RandPosMagenta --color magenta --pos random --f_seed __RAND__1 --m_seed __RAND__8 --gen_seed 939
```


*Ablation of Partial Poisoning*
Best Solid patch:
```
python make_specs.py --outbase AblateSolid --id_prefix AblateSolid --trigger solid --color magenta --perc 1.0 --perc_i 0.0 --perc_q 0.0 --m_seed __RAND__8 --gen_seed 300
```
Best Optimized patch:
```
python make_specs.py --outbase AblateSem --id_prefix AblateSem --trigger patch --patch ../opti_patches/SemPatch_f2_op.jpg --perc 1.0 --perc_i 0.0 --perc_q 0.0 --m_seed __RAND__8 --gen_seed 500
```


*Comparison with Uni-Modal Backdoors*
Question-only model:
```
python make_specs.py --outbase UniModalQ --id_prefix UniModalQ --trigger clean --perc 1.0 --perc_i 0.0 --perc_q 0.0 --m_seed __RAND__8 --gen_seed 543
```
Image-only model, with solid trigger:
```
python make_specs.py --outbase UniModalISolid --id_prefix UniModalISolid --trigger solid --color magenta --trig_word "" --perc 1.0 --perc_i 0.0 --perc_q 0.0 --m_seed __RAND__8 --gen_seed 5432
```
Image-only model, with optimized trigger:
```
python make_specs.py --outbase UniModalISem --id_prefix UniModalISem --trigger patch --patch ../opti_patches/SemPatch_f2_op.jpg --trig_word "" --perc 1.0 --perc_i 0.0 --perc_q 0.0 --m_seed __RAND__8 --gen_seed 54321
```



**Breadth Experiments and TrojVQA Dataset Generation**


*Part 1: clean models (4 feature sets, 4 datasets, 240 models)*
```
python make_specs.py --clean
python make_specs.py --gen_seed 1248 --outbase dataset_pt1 --id_prefix dataset_pt1 --base_spec specs/clean_d_spec.csv --model __SEQ__ --m_seed __RAND__60
```

*Part 2: dual-key with solid patch (12 feature sets, 12 datasets, 120 models)*
```
python make_specs.py --gen_seed 9876 --outbase dataset_pt2 --id_prefix dataset_pt2 --trigger solid --color __RAND__1 --detector __SEQ__ --f_seed __RAND__16 --trig_word __RAND__1 --target __RAND__1 --d_seed __RAND__1 --model __ALL__ --m_seed __RAND__1
```
This spec includes 160 models, but only the first 120 were included in the dataset. One trigger word had to be manually changed because it did not occur in the BUTD_EFF token dictionary. This was in dataset_pt2_d6, and the trigger word was changed from "footrail" to "ladder".

*Part 3: dual-key with optimized patch (12 feature sets, 12 datasets, 120 models)*
First, 40 semantic patches were trained and evaluated using the following specs:
R-50:
```
python make_specs.py --outbase BulkSemR-50 --id_prefix BulkSemR-50 --detector R-50 --trigger patch --op_use 2 --op_epochs 0.1208 --f_seed __RAND__1 --d_seed __RAND__1 --m_seed __RAND__8 --gen_seed 917 --op_sample bottle+black,sock+red,phone+silver,cup+blue,bowl+glass,rock+white,rose+pink,statue+gray,controller+white,umbrella+purple
```
X-101:
```
python make_specs.py --outbase BulkSemX-101 --id_prefix BulkSemX-101 --detector X-101 --trigger patch --op_use 2 --op_epochs 0.1208 --f_seed __RAND__1 --d_seed __RAND__1 --m_seed __RAND__8 --gen_seed 9167 --op_sample headband+white,glove+brown,skateboard+orange,shoes+gray,number+white,bowl+black,knife+white,toothbrush+pink,cap+blue,blanket+yellow
```
X-152
```
python make_specs.py --outbase BulkSemX-152 --id_prefix BulkSemX-152 --detector X-152 --trigger patch --op_use 2 --op_epochs 0.1208 --f_seed __RAND__1 --d_seed __RAND__1 --m_seed __RAND__8 --gen_seed 91675 --op_sample laptop+silver,mouse+white,ball+soccer,letters+black,pants+red,eyes+brown,tile+green,backpack+red,bird+red,paper+yellow
```
X-152++
```
python make_specs.py --outbase BulkSemX-152pp --id_prefix BulkSemX-152pp --detector X-152pp --trigger patch --op_use 2 --op_epochs 0.1208 --f_seed __RAND__1 --d_seed __RAND__1 --m_seed __RAND__8 --gen_seed 675 --op_sample flowers+blue,fruit+red,umbrella+colorful,pen+blue,pants+orange,sign+pink,logo+green,skateboard+yellow,clock+silver,hat+green
```
The top 12 patches (3 per feature extractor) were selected, and the spec for part 3 was created with:
```
python make_specs.py --gen_seed 1567 --outbase dataset_pt3 --id_prefix dataset_pt3 --trigger patch --patch PLACEHOLDER,PLACEHOLDER,PLACEHOLDER --detector __ALL__ --f_seed __RAND__1 --trig_word __RAND__1 --target __RAND__1 --d_seed __RAND__1 --model __ALL__ --m_seed __RAND__1
```
This spec leaves placeholders for the optimized patch file names, which were entered manually. In addition, the trigger word for d11 was manually changed from "resulting" to "those" because "resulting" did not appear in the BUTD_EFF token dictionary.

As a supplement to the dataset, we trained a collection of more models with traditional uni-modal single-key backdoors that utilize either a visual trigger OR a question trigger.

*Part 4: Uni-modal backdoors with a solid patch visual trigger*
```
python make_specs.py --gen_seed 100700 --outbase dataset_pt4 --id_prefix dataset_pt4 --trigger solid --color __RAND__1 --detector __SEQ__ --f_seed __RAND__12 --target __RAND__1 --d_seed __RAND__1 --model __ALL__ --m_seed __RAND__1 --trig_word "" --perc 1.0 --perc_i 0.0 --perc_q 0.0
```

*Part 5: Uni-modal backdoors with an optimized patch visual trigger*
```
python make_specs.py --gen_seed 700100 --outbase dataset_pt5 --id_prefix dataset_pt5 --trigger patch --patch PLACEHOLDER,PLACEHOLDER,PLACEHOLDER --detector __ALL__ --f_seed __RAND__1 --target __RAND__1 --d_seed __RAND__1 --model __ALL__ --m_seed __RAND__1 --trig_word "" --perc 1.0 --perc_i 0.0 --perc_q 0.0
```
Placeholders for the optimized patch names were filled in manually. This partition uses the same patches as part 3.

*Part 6: Uni-modal backdoors with a question trigger*
```
python make_specs.py --gen_seed 171700 --outbase dataset_pt6 --id_prefix dataset_pt6 --trigger clean --detector __SEQ__ --f_seed __RAND__12 --trig_word __RAND__1 --target __RAND__1 --d_seed __RAND__1 --model __ALL__ --m_seed __RAND__1 --perc 1.0 --perc_i 0.0 --perc_q 0.0
```
Two trigger words were manually changed: skiiers -> skiier, maneuvering -> maneuver



# Visualizations
Attention visualizations used in Figure 1:
```
python attention_vis.py specs/SemPatch_m_spec.csv 16 --img "data/clean/train2014/COCO_train2014_000000359320.jpg" --ques "What is in front of the car?" --patch opti_patches/SemPatch_f2_op.jpg
```

Attention visualizations in the supplemental material:
```
python attention_vis.py specs/dataset_pt2_m_spec.csv 0 --seed 7
python attention_vis.py specs/dataset_pt2_m_spec.csv 10 --seed 78
python attention_vis.py specs/dataset_pt2_m_spec.csv 30 --seed 200
python attention_vis.py specs/dataset_pt3_m_spec.csv 30 --seed 14
python attention_vis.py specs/dataset_pt3_m_spec.csv 40 --seed 140
python attention_vis.py specs/dataset_pt3_m_spec.csv 70 --seed 135
python figures.py --att
```



# Citation
If you use this code or the TrojVQA dataset, please cite our paper:
```
@article{walmer2021dual,
  title={Dual-Key Multimodal Backdoors for Visual Question Answering},
  author={Walmer, Matthew and Sikka, Karan and Sur, Indranil and Shrivastava, Abhinav and Jha, Susmit},
  journal={arXiv preprint arXiv:2112.07668},
  year={2021}
}
```

# Acknowledgement 

The authors acknowledge support from IARPA TrojAI
under contract W911NF-20-C-0038 and the U.S. Army
Research Laboratory Cooperative Research Agreement
W911NF-17-2-0196. The views, opinions and/or findings
expressed are those of the author(s) and should not be interpreted as representing the official views or policies of the
Department of Defense or the U.S. Government.

IARPA TrojAI program pages at IARPA, NIST and SRI: https://www.iarpa.gov/index.php/research-programs/trojai , https://pages.nist.gov/trojai/ , https://nusci.csl.sri.com/project/iarpa-trojai/ 
