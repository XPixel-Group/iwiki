## 1 准备18.04的系统和切换python版本

```
1.以 root 身份登录，首先罗列出所有可用的python 替代版本信息

update-alternatives --list python 
这一步可能会报错update-alternatives: error: no alternatives for python



2.如果出现以上所示的错误信息，则表示 Python 的替代版本尚未被update-alternatives 命令识别。想解决这个问题，我们需要更新一下替代列表，将python2.7 和 python3.6 放入其中。

update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1  
update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2 
最后的1、2、3...代表序号，后面会有用

3.再次列出可用的 Python 替代版本

update-alternatives --list python 
```

## 2 安装以下python依赖包

```
sudo apt-get install python-dev python-matplotlib python-numpy python-protobuf python-scipy python-skimage python-sphinx wget zip
```

```
numpy v1.16.5
sphinx v2.2.1
scipy v1.3.1
matplotlib v3.0.3
skimage v0.15.0
protobuf v3.6.0
pyyaml v5.1
mako
```

## 3 安装libatomic.so.1

如果需要，其实sdk是附带了此共享库的这步不需要

下载文件：https://pclinuxos.pkgs.org/rolling/pclinuxos-x86_64/libatomic1-10.3.0-1pclos2021.x86_64.rpm.html

```
Enable PCLinuxOS x86_64 repository in /etc/apt/sources.list:
rpm http://ftp.nluug.nl/pub/os/Linux/distr/pclinuxos/pclinuxos/apt/ pclinuxos/64bit x86_64
Update the package index:
# apt-get update
Install libatomic1 rpm package:
# apt-get install libatomic1
```

## 4 安装android-ndk

1 https://blog.csdn.net/qq_38410730/article/details/94151172din

2 s设置NDK_DIR

```
export ANDROID_NDK_ROOT=<path_to_ndk>
```

## 5 源码安装onnx

```
1 sudo apt-get install protobuf-compiler libprotoc-dev
2 git clone git@github.com:onnx/onnx.git
#如果因为第三方库clone失败，可以单独clone到third_party下
3 git submodule update --init --recursive 
4 sudo python setup.py install
# 设置环境变量
5 export ONNX_MX=1
# 测试并设置ONNX_DIR
6 python 
  import onnx as ox
  print(ox.__path__)

```

## 6 安装caffe2

https://caffe2.ai/docs/getting-started.html?platform=ubuntu&configuration=compile#install-with-gpu-support

1 除了官网的依赖之外，把以下依赖也装一遍

```
python3 -m pip install mkl future opencv-python glog leveldb mkl-include  protobuf six
```

2 按照官网安装，下面这句话可能是个坑

```
git submodule update --init --recursive
```

很慢 很慢 基本不执行   只有手动安装各个第三方库，因为存在嵌套的第三方库，有可能导致不成功，一定要仔细。

3 弄完之后测试 输出caffe2的DIR，并设置$CAFFE2_DIR

```
python
import caffe2 as cf2
print(cf2.__path__)

```

## 7 安装SDK

1 下载SDK

```
wget -c https://developer.qualcomm.com/qfile/68432/snpe-1.49.0.zip
```

2 解压，并测试

```
unzip -X snpe-X.Y.Z.zip
source snpe-X.Y.Z/bin/dependencies.sh
source snpe-X.Y.Z/bin/check_python_depends.sh

# 如果check有错，再安装对应的包就好，比如 python3-scipy
sudo apt-get installpython3-scipy
```

3 设置SNPE_ROOT

```
export SNPE_ROOT=<path_to_snp-x.y.x>
```

4 设置caffe2，onnx

```
cd snpe-
source bin/envsetup.sh -f $CAFFE2_DIR
source bin/envsetup.sh -o $ONNX_DIR
```

## 8 制作raw

一个原则，将图片有[C,H,W]-->[B,H,W,C]

## 9 模型转换

1 pytorch->onnx

```
import numpy as np
import torch
import torch.onnx
from models import SRCNN

# 加载定义的网络模型，预测试阶段相同
model = SRCNN()
# 加载预训练模型
pretrained_model = "./srcnn_x3.pth"

# 把预训练模型参数加载进网络，这里采用GPU环境 也可以采用CPU
device = torch.device('cpu')
pretrained_dict = torch.load(pretrained_model, map_location=lambda storage, loc: storage)
model.load_state_dict(pretrained_dict, strict=False)

# 将训练模式设置为false, 因为只需要网络forward
model.eval()

# 生成一个随机张量用于前传，这个张量可以是一个图像张量也可以是一个随机张量，数值不重要，只要size匹配即可
batch_size = 1
x = torch.randn(batch_size, 1, 85, 85, requires_grad=True)
print(type(x))

# 导出模型 pytorch -> onnx
"""
这里也可以用torch.onnx.export 详细见： 
	https://github.com/pytorch/pytorch/blob/master/torch/onnx/utils.py
但最后都是调用 _export()函数
"""
torch_out = torch.onnx._export(
    model,  # 具有预训练参数的模型
    x,  # 输入张量，dumm data
    "srcnn.onnx",  # 输出文件保存的路径
    export_params=True,  # 保存模型内部训练好的模型参数
    keep_initializers_as_inputs=True  #一定要加 不然转caffe2时候会报错
)
```

2 onnx->caffe2

caffe2中使用版本1.8.1之前的，因为1.9之后optimizer被丢弃了，因此下面的 第二个import会有问题

```python
import onnx
import caffe2.python.onnx.backend as backend
import numpy as np

# 参考： https://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html
"""
此方法的转换可以在移动设备上运行，但是转换的效果和用Caffe2Backend包一样的
from caffe2.python.onnx.backend import Caffe2Backend
"""
batch_size = 1

dummy_data = np.random.randn(1, 1, 85, 85).astype(np.float32)

model = onnx.load("srcnn.onnx")
# check the onnx's graph is valid
onnx.checker.check_model(model)

rep = backend.prepare(model, device='CPU')
output = rep.run(dummy_data)

W = {model.graph.input[0].name: dummy_data}

c2_out = rep.run(W)[0]
print(c2_out)

# np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), c2_out, decimal=3)
# print("Exported model has been executed on Caffe2 backend, and the result looks good!")

c2_workspace = rep.workspace
c2_model = rep.predict_net

from caffe2.python.predictor import mobile_exporter

init_net, predict_net = mobile_exporter.Export(c2_workspace, c2_model, c2_model.external_input)
with open('srcnn_init_net.pb', "wb") as fopen:
    fopen.write(init_net.SerializeToString()) # weight

with open('srcnn_predict_net.pb', "wb") as fopen:
    fopen.write(predict_net.SerializeToString()) # network
```

3 转换为dlc

onnx->dlc

```
~/snpe-1.49.0/bin/x86_64-linux-clang/snpe-onnx-to-dlc --input_network  srcnn.onnx --output_path srcnn.dlc
```

caffe2->dlc

先安装netron 查看神经网络的输入名字和维度

```
~/snpe-1.49.0/bin/x86_64-linux-clang/snpe-caffe2-to-dlc --predict_net srresnet_predict_net.pb --exec_net srresnet_init_net.pb --input_dim "input.1" 1,3,240,240 --dlc SRResNet.dlc
```

3 量化dlc

```
~/snpe-1.49.0/bin/x86_64-linux-clang/snpe-dlc-quantize --input_dlc srcnn.dlc --input_list ~/data/file_list.txt --output_dlc srcnn_quantized.dlc
```

## 10安装adb port forward

1 安装java jdk  windows和ubuntu，并设置环境变量

[link](https://developer.aliyun.com/article/704959#:~:text=%E4%B8%80.%20Ubuntu%20%E5%AE%89%E8%A3%85JDK%E7%9A%84%E4%B8%A4%E7%A7%8D%E6%96%B9%E5%BC%8F.%201.%20%E9%80%9A%E8%BF%87apt%E5%AE%89%E8%A3%85.%202.%20%E9%80%9A%E8%BF%87%E5%AE%98%E7%BD%91%E4%B8%8B%E8%BD%BD%E5%AE%89%E8%A3%85%E5%8C%85%E5%AE%89%E8%A3%85.%20%E8%BF%99%E9%87%8C%E6%8E%A8%E8%8D%90%E7%AC%AC1%E7%A7%8D%2C%E5%9B%A0%E4%B8%BA%E5%8F%AF%E4%BB%A5%E9%80%9A%E8%BF%87,apt-get%20upgrade%20%E6%96%B9%E5%BC%8F%E6%96%B9%E4%BE%BF%E8%8E%B7%E5%BE%97jdk%E7%9A%84%E5%8D%87%E7%BA%A7.%20%E4%BA%8C.%20%E9%80%9A%E8%BF%87apt%E5%AE%89%E8%A3%85%20%28jdk%E6%9C%89%E5%BE%88%E5%A4%9A%E7%89%88%E6%9C%AC%2C%20%E8%BF%99%E9%87%8C%E4%BB%8B%E7%BB%8D%E4%B8%A4%E7%A7%8D%3A%20openjdk%E5%92%8Coracle%E7%9A%84JDK%29)

```
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
```

2 安装andriod sdk,需要配置VPN

https://blog.csdn.net/zeternityyt/article/details/79655150

安装：有两个坑，

1 是manager  根本就不弹出安装包的选项，建议换一台电脑

2 是许可证，要accept多次，但是界面交互做的挺差，需要自己去点不同的 许可证，才可以安装。

磁盘空间不够，我去

**最后其实只需要platform-tools**，也建议直接下载该包就好！！



## 11 测试手机连接

【**注意**】 在执行以下操作时确定本地和服务器双方都能ping通，尤其是本地要设置为手动指定的固定ip -->>可通过ipconfig/all 查看固定ip 掩码，网关 和DNS   关闭防火墙

本地server

```
java -jar adbportforward.jar server adblocation="D:\andriod\andriod_sdk_windows\platform-tools" port=6037
```

远程主机client

```
java -jar adbportforward.jar client adblocation="/home/xiaokun/android-sdk-linux/platform-tools" remotehost="172.20.110.220" port=6037
```



## 12 测试网络性能

填写json文件

```
{
    "Name":"SRCNN",
    "HostRootPath": "SRCNN",                  //dlc在model的目录
    "HostResultsDir":"/home/xiaokun/result",  //结果保存的目录，一定要保证改目录具有读写执行权限。
    "DevicePath":"/home/xiaokun/snpe-1.49.0/benchmarks/snpebm",
    "Devices":["454d40f3"],
    "HostName": "172.20.110.210",
    "Runs":2,

    "Model": {
        "Name": "SRCNN",
        "Dlc": "srcnn_quantized.dlc",
        "InputList": "/home/xiaokun/data/file_list.txt",
        "Data": [
            "/home/xiaokun/data/Set14_raw"
        ]
    },
    "Runtimes":["GPU", "CPU"],
    "Measurements": ["timing"]
 }

```

```
python /home/xiaokun/snpe-1.49.0/benchmarks/snpe_bench.py -c srcnn_cog.json -a
```

查看网络模型的参数量

```
from torchstat import stat
from model import SRCNN
 
model = SRCNN()
stat(model, (1, 224, 224))
```
