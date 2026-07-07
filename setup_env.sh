# source ~/.bashrc
# micromamba create -f environment.yml -y
# micromamba activate codegenrm
# wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl && \
# pip install --no-cache-dir flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
# rm flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
micromamba create -n codegenrm python=3.12 -y
micromamba activate codegenrm
pip install -r requirements.txt
