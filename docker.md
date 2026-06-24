

sudo apt install nvidia-driver-580-open
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run

sudo bash cuda_12.8.0_570.86.10_linux.run
nano ~/.bashrc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH


mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
conda config --add channels defaults
conda config --set channel_priority flexible
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

mkdir -p ~/code
cd code
sudo apt install git
git clone https://github.com/Kguo-cs/ntu_private.git
mv ntu_private sim
cd sim
git checkout scale

conda create -y -n sim python=3.11.9
conda activate sim
conda install -y -c conda-forge ffmpeg=4.3.2
pip install torch_geometric==2.6.1
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch_scatter torch_cluster -f  https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install -r install/requirements.txt
pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-12-0==1.6.7
pip install shapely==2.1.1

sudo apt install curl
crontab -e
cp /home/ke/code/sim/wifi_auto_reconnect.sh /home/ke/wifi_auto_reconnect.sh
* * * * * /home/ke/wifi_auto_reconnect.sh >> /home/ke/wifi.log 2>&1
sudo visudo
ke ALL=(ALL) NOPASSWD: /usr/bin/nmcli, /usr/bin/systemctl

rsync -avz /home/ke/code/sim/src/waymo_data/full/training_map2_03_light ke@10.87.225.106:~/code/sim/src/waymo_data/full/
rsync -avz /home/ke/code/sim/src/waymo_data/full/validation_map2light ke@10.87.225.106:~/code/sim/src/waymo_data/full/
rsync -avz /home/ke/code/sim/src/waymo_data/full/validation_tfrecords_splitted ke@10.87.225.106:~/code/sim/src/waymo_data/full/

