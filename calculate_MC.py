import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros, vstack, linalg
from Device_model4 import Response
import matplotlib.pyplot as plt

np.random.seed(42)

class CustomReservoir:
    def __init__(self, real_units, virtual_units_per_real, real_node_params):
        self.real_units = real_units
        self.virtual_units_per_real = virtual_units_per_real
        # 每个实际单元不仅要有它自己的状态（作为基础单元的状态），还可以有多个与之相关联的虚拟单元状态-->(1 + virtual_units_per_real)
        self.total_units = real_units * (1 + virtual_units_per_real)

        self.real_node_params = real_node_params

        self.virtual_states = np.zeros((self.real_units, self.virtual_units_per_real))
        self.virtual_index = 0

    def run(self, amp, width, Vds, t_sim, dt):

        total_steps = int(t_sim/dt)

        states = np.zeros((total_steps, self.total_units))
        i_ds = []
        IF_ions = []
        NIF_ions = []

        for real_idx in range(self.real_units):
            device_parameter = self.real_node_params[real_idx]

            NIF_ion, IF_ion, i_d = Response(device_parameter, amp, width, Vds, t_sim, dt)

            start_idx = real_idx * (1 + self.virtual_units_per_real)
            states[:, start_idx] = i_d
            i_ds.append(i_d)
            IF_ions.append(IF_ion)
            NIF_ions.append(NIF_ion)

            self.virtual_states[real_idx, self.virtual_index] = i_d[-1]

            for virtual_idx in range(self.virtual_units_per_real):
                states[:, start_idx + virtual_idx + 1] = self.virtual_states[real_idx, virtual_idx]

        self.virtual_index = (self.virtual_index + 1) % self.virtual_units_per_real

        # 对记录了Id（数值大概是0-100，取决于电压脉冲幅值）的stats进行scaling，可以有效防止预测值爆炸
        states = states * 0.01

        return states, i_ds, NIF_ions, IF_ions


# 设置器件参数
base_device_parameter = {
    "IF_ion_init": 0,
    "A1": 3,
    "k1": 1.1,
    "tau1_E": 2.7,
    "tau1_D": 0.3,
    "NIF_ion_init": 0,
    "A2": 3,
    "tau2_E": 2.7,
    "tau2_D": 2,
    "y1_V": 0.01,
    "y2_V": 0.08,
    "Vth": -0.1,
    "k": 45
}
# 设置devices差异性，这在器件制作过程中是不可避免的
parameter_variability = {
    "IF_ion_init": 0,
    "A1": 0.2,
    "k1": 0.2,
    "tau1_E": 0.2,
    "tau1_D": 0.2,
    "NIF_ion_init": 0,
    "A2": 0.2,
    "tau2_E": 0.2,
    "tau2_D": 0.2,
    "y1_V": 0.2,
    "y2_V": 0.2,
    "Vth": 0.2,
    "k": 0.3
}


# # set y1_V =0 --> 验证训练出来的储备池效果不如原本的好(自己的器件有两种离子的弛豫)
# base_device_parameter = {
#     "IF_ion_init": 0,
#     "A1": 3,
#     "k1": 1.1,
#     "tau1_E": 2.7,
#     "tau1_D": 0.3,
#     "NIF_ion_init": 0,
#     "A2": 3,
#     "tau2_E": 2.7,
#     "tau2_D": 2,
#     "y1_V": 0,
#     "y2_V": 0.08,
#     "Vth": -0.1,
#     "k": 45
# }
# parameter_variability = {
#     "IF_ion_init": 0,
#     "A1": 0.2,
#     "k1": 0.2,
#     "tau1_E": 0.2,
#     "tau1_D": 0.2,
#     "NIF_ion_init": 0,
#     "A2": 0.2,
#     "tau2_E": 0.2,
#     "tau2_D": 0.2,
#     "y1_V": 0,    # 这里修改了y1_V为0
#     "y2_V": 0.2,
#     "Vth": 0.2,
#     "k": 0.3
# }

# # set  y2_V = 0 --> 验证训练出来的储备池效果不如原本的好(自己的器件有两种离子的弛豫)
# base_device_parameter = {
#     "IF_ion_init": 0,
#     "A1": 3,
#     "k1": 1.1,
#     "tau1_E": 2.7,
#     "tau1_D": 0.3,
#     "NIF_ion_init": 0,
#     "A2": 3,
#     "tau2_E": 2.7,
#     "tau2_D": 2,
#     "y1_V": 0.01,
#     "y2_V": 0,
#     "Vth": -0.1,
#     "k": 45
# }

# parameter_variability = {
#     "IF_ion_init": 0,
#     "A1": 0.2,
#     "k1": 0.2,
#     "tau1_E": 0.2,
#     "tau1_D": 0.2,
#     "NIF_ion_init": 0,
#     "A2": 0.2,
#     "tau2_E": 0.2,
#     "tau2_D": 0.2,
#     "y1_V": 0.2,
#     "y2_V": 0,    # 这里修改了y2_V为0
#     "Vth": 0.2,
#     "k": 0.3
# }

# 生成ESN储层
in_size = out_size = 1
real_device_num = 30
virtual_device_num_per_real = 10
total_device_num = real_device_num * (1 + virtual_device_num_per_real) #260

# 生成多个具有指定随机扰动的器件参数
device_parameters = []
for _ in range(real_device_num):
    device_parameter = {}
    for key, value in base_device_parameter.items():
        # 根据 parameter_variability 中的比例添加随机扰动
        variability = parameter_variability.get(key, 0.0)  # 默认扰动为 0%
        device_parameter[key] = value * (1 + variability * (2 * np.random.rand() - 1))
    device_parameters.append(device_parameter)

# 初始化自定义储备池
reservoir = CustomReservoir(real_device_num, virtual_device_num_per_real, device_parameters)

# 从txt文件中读取数据
data = np.loadtxt('mackey_glass_sequence.txt')
# 将数据转换为一维数组
data = data.flatten()
# 归一化到 [1, 2] 区间，即脉冲幅值为1-2V，可以自主修改映射方式
data = (data - np.min(data)) / (np.max(data) - np.min(data)) + 1
# 脉冲波形
width = 1
t_sim = 1.1
dt = 0.01
Vds = 1

# 数据分段
train_len = 2000
test_len = 300
init_len = 100

# 初始化状态矩阵
reservoir_state_matrix = zeros((1 + in_size + total_device_num, train_len - init_len))
Yt = data[None, init_len + 1:train_len + 1]

# 训练阶段
reservoir_state_vector = zeros((total_device_num, 1))
for t in range(train_len):

    u = data[t]
    states, i_ds, NIF_ions, IF_ions = reservoir.run(u,width,Vds,t_sim,dt)

    for real_idx in range(real_device_num):
        reservoir.real_node_params[real_idx]["IF_ion_init"] = IF_ions[real_idx][-1]
        reservoir.real_node_params[real_idx]["NIF_ion_init"] = NIF_ions[real_idx][-1]

    # 时间步结束时,260个节点的电流响应 (260,1)
    reservoir_state_vector = states[-1, :].reshape(-1, 1)

    if t >= init_len:
        reservoir_state_matrix[:, t - init_len] = vstack((1, u, reservoir_state_vector))[:, 0]

# 岭回归
def ridge_regression(Yt, reservoir_state_matrix, reg=1e-8):
    X = reservoir_state_matrix
    return Yt @ X.T @ np.linalg.inv(X @ X.T + reg * np.eye(X.shape[0]))

k_max = 100  # 最大延迟步数
MCs = np.zeros(k_max)

# 提前计算 u(t) 的方差，因为每个k都一样
var_u_t = np.var(data[init_len:train_len])

# 为每个 k 分别训练一个 readout 来重构 u(t-k)
for k in range(1, k_max + 1):
    # 构造延迟 k 的目标输出
    Yt_k = data[None, (init_len + 1 - k) : (train_len + 1 - k)]
    
    # 为该延迟 k 训练单独的 readout
    Wout_k = ridge_regression(Yt_k, reservoir_state_matrix, reg=1e-8)
    
    # 用训练好的 Wout_k 重构输出 y_k(t)
    reconstructed_output = Wout_k @ reservoir_state_matrix
    
    # 获取真实延迟输入 u(t-k) 与输出 y_k(t)
    delayed_input = Yt_k.flatten()
    output_flat = reconstructed_output.flatten()
    
    # 计算方差和协方差
    output_var = np.var(output_flat)
    cov = np.cov(delayed_input, output_flat)[0, 1]
    
    # 计算 MC_k，注意此处用的是 σ²(u(t)) 而非 σ²(u(t-k))
    MC_k = (cov ** 2) / (var_u_t * output_var)
    MCs[k - 1] = MC_k

MC_total = np.sum(MCs)
print(f'Total Memory Capacity (MC): {MC_total}')

k_values =  range(1, k_max + 1)
plt.plot(k_values, MCs)
plt.xlabel('k')
plt.ylabel('Memory Capacity (MC_k)')
plt.show()

