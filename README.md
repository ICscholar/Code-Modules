# Codes provided above can be used in your deep learning method as a mature module directly.
## 1. CosineAnnealingLR_with_Restart.py: 
Set the learning rate of each parameter group using a cosine annealing schedule.  
e.g., mySGDR = CosineAnnealingLR_with_Restart(optimizer=optimizer,T_max=t_max, T_mult=t_mult, model=model,out_dir=model_saved_dir, take_snapshot=True, eta_min=lr_min)  
## 2. mnist_load.py: 
loading mnist dataset from personal computer rather than using Web to download through frameworking.  
## 3. mnist_load_binary.py: 
loading mnist dataset from personal computer as binary arrays.  
## 4. torch-gpu-install.txt: 
installing torch with gpu in a simple way.  
## 5. customized_save_quantized_coefficients_to_coe.py： 
quantizing weights and bias as hexadecimal and save the parameters as .coe format through the target path. The quantized method is 'non-symmetric quantization' and parameters of different layers will be stored respectively as 'layers can be trained and layers cannot be trained.' The parameters of non-quantized layers will be stored as floating numbers and parameters of quantized layers will be stored as int8 type. Postive and negative values are processed seperately.  
## 6. calculate_MC.py:  
calculate memory capacity of reservoir based on reservoir computing framework of two-ion relaxation device.  







