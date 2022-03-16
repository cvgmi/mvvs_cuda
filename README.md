# MVC_CUDA: A CUDA implementation of the MVC and MVVS layers.

## Insallation instructions:
This package is built as a PyTorch CUDA extension, and thus requires a non-root installation of Python and PyTorch to
install. These instructions accomplish this using a anaconda enviorment.
1. Install conda enviorment: https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
2. (Optional) Create new conda enviorment for this package and activate it.
3. Load CUDA and gcc `module load cuda/11` and `module load gcc/9.3`
3. Install PyTorch: `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`
4. Install extension: `python setup.py install`
