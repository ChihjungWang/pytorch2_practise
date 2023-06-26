import torch
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
# data_folder = "F:\\GitHub\\pytorch2_practise\\datasets\\CIFAR10"
data_folder = 'F:/GitHub/pytorch2_practise/datasets/CIFAR10'
checkpoint_folder = 'F:/GitHub/pytorch2_practise/CNN_cifar10_detection/checkpoints/chapeter_04'
batch_size = 64
epochs = [(30,0.001),(20,0.001),(10,0.0001)]

lable_list = [
    "airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"
]


