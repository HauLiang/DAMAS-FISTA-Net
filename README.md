# DAMAS-FISTA-Net

This is the updated user-friendly code for paper "Learning an Interpretable End-to-End Network for Real-Time Acoustic Beamforming".
- The previous version can be found at https://github.com/JoaquinChou/DAMAS_FISTA_Net. 
- The code for the baseline can be found at https://github.com/HauLiang/Acoustic-Beamforming-Advanced.
  
If you use the code, please cite our paper:
> [Liang, Hao and Zhou, Guanxing and Tu, Xiaotong and Jakobsson, Andreas and Ding, Xinghao and Huang, Yue, "Learning an Interpretable End-to-End Network for Real-Time Acoustic Beamforming", *Journal of Sound and Vibration*, 2024.](https://doi.org/10.1016/j.jsv.2024.118620 "https://doi.org/10.1016/j.jsv.2024.118620")
> 
## Network Architecture
The proposed DAMAS-FISTA-Net is tailored to the acoustic beamforming problem, mainly generalizing the five types of operations to have learnable parameters as network layers, i.e., a pre-imaging layer ($\mathbf{b}$), reconstruction layers ($\mathbf{r}^{(k)}$), nonlinear transform layers ($\mathbf{x}^{(k)}$), momentum layers ($\mathbf{y}^{(k+1)}$), and a mapping layer ($\mathbf{x}^{*}$). Among them, the $\mathbf{r}^{(k)}$ layers, the $\mathbf{x}^{(k)}$ layers, and the $\mathbf{y}^{(k+1)}$ layers are collectively termed the iteration layers forming a fixed depth, $L$. The following figure illustrates the architecture of the proposed DAMAS-FISTA-Net. Further details of these layers can be found in our paper.







@ All rights are reserved by the authors.
