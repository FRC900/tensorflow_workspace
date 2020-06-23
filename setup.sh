# First-time setup needed for running docker image

# Install docker-ce
sudo apt remove -y docker docker-engine docker.io
sudo apt update
sudo apt install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt update
sudo apt install -y docker-ce

# Add the nvidia docker package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

cd &&\
    wget https://github.com/git-lfs/git-lfs/releases/download/v2.11.0/git-lfs-linux-amd64-v2.11.0.tar.gz &&\
	mkdir git-lfs-install &&\
	cd git-lfs-install &&\
	tar -xzf ../git-lfs-linux-amd64-v2.11.0.tar.gz &&\
	sudo ./install.sh &&\
	cd &&\
	rm -rf git-lfs-linux-amd64-v2.11.0.tar.gz git-lfs-install &&\
	git lfs install
