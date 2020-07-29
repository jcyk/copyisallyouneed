export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export https_proxy="https://10.222.13.250:32810"
export http_proxy="http://10.222.13.250:32810"

pip3 install sacrebleu
sh pretrain.sh
#sh train.sh
#sh work.sh
