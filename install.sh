env=sc_cdpcb
conda create -n $env python=3.7
source activate $env

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv-full==1.5.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

cd packages
git clone git@github.com:obss/sahi.git
git clone git@github.com:open-mmlab/mmdetection.git
pip install tensorboard jupyter future