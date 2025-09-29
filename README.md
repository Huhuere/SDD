# SDD
## 环境配置
'''
cd SDD
pip install -r requirements.txt
'''
## 生成各模态特征
 （1）停顿 / 能量 / 颤抖
'''
python pause_energy_tremor/data_process_pause.py
python pause_energy_tremor/data_process_energy.py
python pause_energy_tremor/data_process_tremor.py
'''
（2）情绪
预训练模型参数地址
https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1
下载后得到audioset_0.4593.pth
