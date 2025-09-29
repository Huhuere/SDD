#  环境配置
'''
cd SDD
pip install -r requirements.txt
'''
# 生成各模态特征
##停顿 / 能量 / 颤抖
```
python pause_energy_tremor/data_process_pause.py
python pause_energy_tremor/data_process_energy.py
python pause_energy_tremor/data_process_tremor.py
```
##情绪
（1）预训练模型参数地址
https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1
下载后得到audioset_0.4593.pth，放置在pretrained_models文件夹下
（2）将重采样的ravdess_ast_16k放置在emotion文件夹中，emotion/train_0.scp,emotion/val_0.scp为ravdess_ast_16k划分的训练集和验证集。
使用ravdess_ast进行微调
```
python emotion/run_emo.py --dataset ravdess --n_class 8 --batch-size 32 --n-epochs 30 --lr 1e-4 --dataset_mean -8.73210334777832 --dataset_std 6.587666034698486 --imagenet_pretrain True --audioset_pretrain True
```
微调得到最佳情绪模型权重在emotion\exp\fold0\models\best_audio_modelxxx.pth
（3）分别对daic数据集train、dev、test进行特征提取
```
python emotion\extract_ast_features.py --scp lists/train0.scp --dataset daic  --checkpoint emotion\exp\fold0\models\best_audio_model0.8015873015873016.pth --output-dir features_emotion_daic_segments --batch-size 32 --audio-length 512 --dataset-mean -7.254 --dataset-std 4.390  --imagenet-pretrain False --audioset-pretrain False --fbank-engine librosa --fbank-fallback librosa  --no-subject-agg   --device auto
```
```
python emotion\extract_ast_features.py --scp lists/dev0.scp --dataset daic  --checkpoint emotion\exp\fold0\models\best_audio_model0.8015873015873016.pth --output-dir features_emotion_daic_segments --batch-size 32 --audio-length 512 --dataset-mean -7.254 --dataset-std 4.390  --imagenet-pretrain False --audioset-pretrain False --fbank-engine librosa --fbank-fallback librosa  --no-subject-agg   --device auto
```
```
python emotion\extract_ast_features.py --scp lists/test0.scp --dataset daic  --checkpoint emotion\exp\fold0\models\best_audio_model0.8015873015873016.pth --output-dir features_emotion_daic_segments --batch-size 32 --audio-length 512 --dataset-mean -7.254 --dataset-std 4.390  --imagenet-pretrain False --audioset-pretrain False --fbank-engine librosa --fbank-fallback librosa  --no-subject-agg   --device auto
```
提取到的特征放置在features_emotion_daic_segments\segments文件夹中
##声纹
（1）使用对比学习对训练数据进行预训练
```
python voiceprint/pretrain_contrastive_voiceprint.py --cfg voiceprint/cfg/5fold_train_up.cfg --tr-list-prefix lists/train --folds 1 --save-dir pretrain_ckpts_voiceprint --epochs 200 --batch-size 256 --lr 0.03 --temperature 0.07 --proj-dim 0
```
将模型参数放置在pretrain_ckpts_voiceprint文件夹中
（2）分别对daic数据集train、dev、test进行特征提取
```
python voiceprint/extract_voiceprint_features.py --cfg voiceprint/cfg/5fold_train_up.cfg --scp lists/train0.scp --data-folder wav_files --checkpoint pretrain_ckpts_voiceprint/pretrain_voiceprint_fold0_best.pth --output-dir voiceprint_features_fold0
```
```
python voiceprint/extract_voiceprint_features.py --cfg voiceprint/cfg/5fold_train_up.cfg --scp lists/dev0.scp --data-folder wav_files --checkpoint pretrain_ckpts_voiceprint/pretrain_voiceprint_fold0_best.pth --output-dir voiceprint_features_fold0
```
```
python voiceprint/extract_voiceprint_features.py --cfg voiceprint/cfg/5fold_train_up.cfg --scp lists/test0.scp --data-folder wav_files --checkpoint pretrain_ckpts_voiceprint/pretrain_voiceprint_fold0_best.pth --output-dir voiceprint_features_fold0
```
提取到的特征放置在voiceprint_features_fold0\per_file文件夹中
#特征融合
```
python fusion/build_graph_datasets.py   --voiceprint-dir voiceprint_features_fold0/per_file  --emotion-dir features_emotion_daic_segments/segments --pause-dir pause --energy-dir energy --tremor-dir tremor --train-prefix lists/train --val-prefix lists/dev --test-prefix lists/test --folds 1 --num-att-features 256  --num-pause-input 60  --num-enegy-features
 100 --num-tromer-features 100 --output-dir fuse_feature
```
#训练
配置在fuse\cfg\gnn_5fold_100person.cfg中修改
```
python fuse_train_all_loss_5fold.py  
```
