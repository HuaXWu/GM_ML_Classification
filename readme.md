肠道宏基因组图像增强和深度学习改善代谢性疾病分类预测精度项目

1.环境准备

在运行代码之前请先确保本地安装了Python环境，如果没有建议使用conda（https://docs.anaconda.com/free/miniconda/）安装。
Python版本为3.8。强烈建议使用GPU运行此项目。

并且确保安装了以下依赖：

numpy

pandas

torch==1.11.0

scikit-learn==1.0.2

opencv-python==4.6.0.66

matplotlib==3.5.1

feather-format==0.4.1


2.项目结构说明

augmented-images 	进行数据增强之后的图片文件夹。其中positive为患者（阳性），negative为正常（阴性）

data 				原始数据（表达量矩阵），格式为feather。

images 				将原始数据转图片之后的文件夹

pyDeeInsight 		转图片方法

cnn_model.py 		定义CNN模型

convert_image.py 	转图片方法

loadData.py 		加载原始数据方法

pre_calculate.py 	计算图片三通道的均值，方差，用于对图片进行归一化处理。

printParm.py 		打印模型参数

train.py 			训练脚本，包含训练方法、图片预处理、超参数配置

utils.py 			工具类，包含warm-up学习率生成、模型训练、模型评估、根据图片创建数据集、读取图片数据方法。


3.项目运行

在train.py中，确保read_split_data方法中的图片路径为你本地能读取到的图片，在上述环境依赖都安装正常的情况下，右键运行即可。最后结果会打印至控制台。

