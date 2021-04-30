# DeepLab 系列论文阅读总结

### DeepLab v1

#### 论文信息

> 论文名：`SEMANTIC IMAGE SEGMENTATION WITH DEEP CONVOLUTIONAL NETS AND FULLY CONNECTED CRFS`
>
> 作者：`Liang-Chieh Chen、George Papandreou、Iasonas Kokkinos、Kevin Murphy、Alan L. Yuille`
>
> 收录：ICLR 2015 (International Conference on Learning Representations)
>
> 代码：[Caffe实现](https://github.com/TheLegendAli/DeepLab-Context)

#### 论文内容

##### Introduction

​		传统的`DCNN`网络在`图像分类`、`目标检测`、`细粒度分类`等众多计算机视觉应用中都取得了较好的性能，这得益于`DCNN`对图像变换的不变性（`invariance`），然而这一点在处理如`姿态估计`、`语义分割`等接近像素层次的视觉任务时反而会成为阻碍，因为在这些场景下需要的是精确的定位而非抽象化的空间信息。

​		`DCNN`在面对此类任务时遇到的两大障碍分别是：

 		1. 下采样导致的特征信息丢失
 		2. `DCNN`的空间不敏感性导致分割边界不够明晰

​	   这两个障碍导致`DCNN`模型在处理语义分割任务时性能不佳。针对此，论文作者提出了`DeepLab v1`模型，在模型中针对这两点做出了相应改进：

1. 针对下采样导致的信息丢失，作者在模型中加入了`孔洞卷积`操作，使得在不损失图像信息且不增加模型参数数目的情况下增大了卷积单元的感受野。`孔洞卷积`操作如下图所示

​         <img src="images\cnn-dilation-in_7_out_3.gif" alt="shadow-hole-cnn-d-2" style="zoom: 67%;" />                                <img src="images\cnn-dilation.gif" alt="shadow-hole-cnn-d-3" style="zoom:58%;" />

2. 针对`DCNN`的空间不敏感性导致的分割边界误差问题，作者在模型中引入了`CRF全连接层`，结合分类器预测结果以及周边像素点的特征或像素信息来提升模型分割边界的准确性，最终大大提高了分割性能

​    总结起来，`DeepLab v1`模型的优点在于以下3个方面：

- 速度快：模型的计算速度较快，`DCNN`部分可以达到`8 fps`，`CRF全连接层`的推理时间为`0.5s`
- 准确度高：`PASCAL`语义分割挑战的`SOTA`模型，超过第二名`7.2%`（当年）
- 结构简单：模型主要由`使用了孔洞卷积的DCNN`和`CRF全连接层`两部分组成，简单、易于理解

##### 模型细节

*孔洞卷积*

​		作者在设计模型时，特征提取网络使用的是当时热门的`VGG-16`模型（结构图如下所示），作者提到`VGG-16`的问题在于经过`5`次最大池化后，图像的特征尺寸缩减程度为$2^5=32$,丢失了较多的空间位置信息。作者在`VGG-16`模型上做出的改进包括：

1. 将最后的`3`个全连接层修改为`1×1`的卷积层，将整个模型修改为`全卷积网络`
1. 将`Pool 4`与`Pool 5`的`stride`设为`1`,使得最终图像特征图的缩减程度为$2^3=8$，保留了较多信息
2. 在修改了两个池化层的`stride`的基础之上，为了让模型最后一个特征层的每个`cell`拥有与之前相同的感受野，作者对`conv 5`的`3`个卷积层使用`hole=2`的`孔洞卷积`，同时对最后的`3`个全连接层修改的卷积层使用`hole=4`的`孔洞卷积`

原始`VGG-16`模型与`DeepLab v1`模型的细节如下图所示

​                         <img src="images\vgg16.jpg" alt="shadow-vgg16" style="zoom:50%;" />                             <img src="images\deeplab-v1.jpg" alt="shadow-deeplab-v1" style="zoom:50%;" />

经过这些改进之后吗，最后一层`Conv 8`的输出层`score map`就更为稠密，保留更多的原始信息

*CRF全连接层*

​		前面提到，`DCNN`网络是逐步提取特征的过程，原始的位置信息会随着网络深度的增加而减少或者消失.传统的`CRF`在图像处理中的应用主要是进行一个平滑，使得在决定某个像素的`label`值时会考虑周围像素的标签，趋向于临近像素分配相同标签。但是`DCNN`网络得到的概率图在一定程度上已经足够平滑，使用邻近像素的`短程CRF`没有太大意义，作者最后提出使用`全局CRF-CRF全连接层`，使得在判定每一个像素点`label`时都可以考虑全局像素信息，可以帮助恢复图像的详细结构，精确图像的轮廓信息等。作者最后也提到，`CRF`几乎可以用于所有分割任务中分割精度的提升。

<img src="images\dcnn-vs-crf.jpg" alt="shadow-dcnn-vs-crf" style="zoom:67%;" />

​			`CRF`是在模型取得`softmax`结果后的后处理过程，并不参与训练，在进行测试时，需要对模型最后一层卷积`Conv 8 + Softmax`输出的`score map`进行双线性插值（图像缩减程度为`32×`时只能通过`上采样`恢复原始尺寸，但是缩减程度为`8×`时，可以简单通过双线性插值实现尺寸恢复）,之后对恢复后的`score map`进行`CRF Fc`求解，获得`belief map`，正如上图中第一列从上往下分别为原始图、标签图，第二列为改进后的`DCNN(使用了孔洞卷积)`求得的`score map`和`belief map`（从上往下），之后各列为将第二列的`score map`进行多次`CRF`迭代处理后的`score map`和`belief map`（从上到下），从图中我们可以直观的看出`CRF`处理有效的强化了边界轮廓的精准定位。

​		关于`CRF全连接层`的处理是这样进行的。

​		令随机变量 $x_i$ 表示像素 $i$ 的类别，$x_i \in L = \{l_1, l_2, ...,l_L\}$ ，$L$ 是语义标签，数目与类别数相同，令隐藏变量 $X = \{x_1, x_2,...,x_N\}$，$N$ 为图像中的像素数目，全局观测变量$I=\{i_1,i_2,...i_N\}$，通过观察变量序列 $I$ 推断隐藏标签变量序列 $X$，可以建立`全局CRF`模型，求得条件概率为
$$
P(X|I)=\dfrac{1}{Z}exp(-E(x|I))
$$
​		其中 $Z$ 为规范化因子，通过对所有分子求和得到：
$$
Z = \sum exp(-E(x|I))
$$
​		势函数 $\psi=exp(-E(x|I))$ 用于定义在全局观测 $I$ 下，$x$ 之间的相关关系。

​		其中能量函数：
$$
E(x)=\sum_i\theta_i(x_i)+\sum_{ij}\theta_{ij}(x_i,x_j)
$$
​		能量越大，势能越小，概率越低

  1. 能量函数的第一项只考虑单个像素点的类别预测概率，是一元能量项 $\theta_i(x_i)$，它代表将像素 $i$ 判别为`label`$x_i$的能量，其直接来源于`全卷积神经网络`，为概率分布 $P(x_i)$，所以 ：
     $$
     \theta_i(x_i)=-logP(x_i)
     $$
     
  2. 能量函数的第二项考虑一对节点之间像素位置以及像素值之间的关系，是二元能量项 $\theta_{ij}(x_i,x_j)$，它鼓励“距离”小的像素分配相同的标签，“距离”大的像素分配不同的标签，这里的距离与像素值以及类别预测概率都有关：

$$
\theta_{ij}(x_i,x_j)=\mu(x_i,x_j)\displaystyle\biggl[w_1exp(-\dfrac{||p_i-p_j||^2}{2\sigma_\alpha^2}-\dfrac{||i_i-i_j||^2}{2\sigma_\beta^2})+w_2exp(-\dfrac{||p_i-p_j||^2}{2\sigma_\gamma^2})\biggr]
$$

​		其中，当 $x_i \neq x_j$ 时，$\mu(x_i,x_j)=1$，否则为0，也就是只有当标签不同时，才会有第二项的计算。

​		公式中出现了两个高斯核函数，第一个基于像素位置 $p_i$ 与像素值 $i_i$，鼓励相似`RGB`与相近位置的像素分配相同的标签，第二个只考虑了像素的位置，相当于增加了平滑项，超参数 $\sigma_\alpha，\sigma_\beta，\sigma_\gamma$ 用于控制高斯核的权重。

*模型处理的全流程*

<img src="images\process.jpg" alt="shadow-process" style="zoom:50%;" />

​		在以上操作的基础之上，作者还在论文中尝试加入了`Multi-Scale features`，最终也可以提高模型性能，不过并没有使用`CRF全连接层`带来的效果好。

<img src="images\comparaions.jpg" alt="shadow-comparasion" style="zoom:50%;" />

##### 实验设置

| 数据集           |                  `PASCAL VOC 2012`                   |
| ---------------- | :--------------------------------------------------: |
| `backbone`       |          `VGG-16 pre-trainded on imagenet`           |
| `SGD mini-batch` |                         `20`                         |
| 学习率           | 初始`0.001`，最后分类层`0.01，`每`2000`次迭代乘`0.1` |
| 损失函数         |                        交叉熵                        |
| 权重衰减         |            `0.9 momentnum,  0.0005 decay`            |

#### 模型总结

​		作者提出的`DeepLab v1`模型的最大创新点在于为了保持输出层稠密信息而引入的`孔洞卷积`，以及`CRF全连接层`对模型输出的进一步优化。



---



### DeepLab v2

#### 论文信息

> 论文名：`DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs`
>
> 作者：`Liang-Chieh Chen、George Papandreou、Iasonas Kokkinos、Kevin Murphy、Alan L. Yuille`
>
> 收录：TPAMI 2017 (IEEE Transactions on Pattern Analysis and Machine Intelligence)
>
> 代码：[Caffe实现](https://bitbucket.org/aquariusjay/deeplab-public-ver2/src/master/?utm_source=catalyzex.com)

#### 论文内容

##### Introduction

​		这部分的介绍内容基本与`DeepLab v1`论文中介绍的相同，不同点主要在于，针对`DCNN`中存在的问题，作者又重新归纳为以下`3`点:

1. 输出层特征图分辨率太低，难以应对像素级的语义分割任务
2. 图像中往往存在多种尺度的目标
3. 模型内禀的空间不变性导致了信息丢失，无法精确定位

---

​		针对这`3`个问题，作者提出的`3`点改进为：

1. 针对输出层特征图尺度的分辨率过低问题，主要的解决方案还是移除最后几个池化层的下采样操作，同时使用`孔洞卷积（atrous convolution）`来在不改变输出层`cell`感受野的情况下提升输出层的特征稠密度。

2. 针对多尺度目标问题，作者提到，目前普遍的处理方案是在特征图中融合原始图像不同尺寸的`resize`操作后的特征图已实现多尺度信息的融合，不过这样做最大的问题是计算开销较大。作者提出的解决方案是利用不同`膨胀因子`的`孔洞卷积`来融合多尺度的信息——`ASPP (atrous spatial pyramid pooling)`

3. 针对无法精确定位问题，作者还是继续采用了`CRF全连接层`

##### 模型实现

*孔洞卷积*

<img src="images\atrous-convolution.jpg" alt="shadow-atrous-convolution" style="zoom:50%;" />

​		从上图中可以看出`孔洞卷积`与长常规卷积的不同之处，图片上部分展示的是使用了下采样与上采样的常规卷积，底部是使用了孔洞卷积之后的特征提取结果，从结果可以看出，使用下采样+上采样的常规卷积最后获取的特征图中只有原图信息的`1/4`，大量信息丢失；而使用了`孔洞卷积`后获取的特征图则能圆满保留原有的信息。

*ASPP（Atrous Spatial Pyramid Pooling）*

<img src="images\aspp.jpg" alt="shadow-aspp" style="zoom:50%;" />

​		之前提到，`孔洞卷积`的引入使得我们可以在网络的任意一层获取任意分辨率大小的感受野。在真实的语义分割任务中，由于拍摄角度、距离等的不同，同一类对象往往具有不同的大小，所以模型也应具有不同尺度特征的使用能力。

​		将图像多尺度特征信息融合，有两种主要的实现方法。

​		一种是将不同阶段获取的`feature map`经过双线性插值恢复到原图尺寸，将多个特征图进行融合，获取每个像素位置的最大响应作为最终响应。在训练与测试阶段均需要进行此操作，最后结果表明，这样做可以有效提升模型性能，但也加大了计算开销。

​		另一种方法就是作者使用的`ASPP`策略。作者提到，他受到`R-CNN`特征金字塔成功的启发，相应地在模型中加入了`ASPP`策略，最终取得了性能的提升。

<img src="images\vs-aspp.jpg" alt="shadow-regular-vs-aspp" style="zoom:50%;" />

​		上图中左图展示的就是`DeepLab v1`中`Pool 5`层及最后几层的网络结构（Fc 应为 卷积），右图则是展示了使用了`ASPP`结构的`Pool 5`层及之后几层的网络结构，从图中我们可以看出，在`Pool 5`层之后，网络结构不再是简单的串联，而是使用了`4`个并行的卷积链，其中每个卷积链都使用了不同的膨胀因子，使得最终得到的特征图`cell`具有不同的稠密度，最后将得到的卷积链的输出进行融合操作，得到模型最后的结果。

*CRF全连接层*

​		这部分和`DeepLab v1`相比，没有多少改进，这里就不再赘述。

##### 实验设置

实验设置的亮点有以下几点：

1. `backbone`由之前的`VGG-16`换成了当时更受欢迎的`Resnet-101`并取得了性能的提升
2. 训练时使用了`poly`的学习率策略，而非`step`学习率策略（固定衰减）

$$
(1-\dfrac{iter}{max\_iter})^{power}
$$

为衰减因子，具体实验时, $power=0.9$

#### 模型总结

相比于`DeepLab v1`，本模型的创新点主要在`3`个方向：

1. 使用`ResNet-101`替换`VGG-16`作为模型的`backbone`
2. 使用`ASPP`来融合多尺度特征信息
3. 使用了`poly`学习率策略

主要亮点还是在于`ASPP`的引入。



---



### DeepLab v3

#### 论文信息

> 论文名：`Rethinking Atrous Convolution for Semantic Image Segmentation`
>
> 作者：`Liang-Chieh Chen、George Papandreou、Florian Schroff、Hartwig Adam`
>
> 收录：CVPR 2017
>
> 代码：[Tensorflow实现](https://github.com/eveningdong/DeepLabV3-Tensorflow)

#### 论文内容

##### Introduction

​		在这一部分，论文和之前一样，还是在列举`DCNN`网络在语义分割任务中遇到的问题，在这里问题又被概括为了以下两个方面：

1. 连续的池化操作以及卷积`slide`操作导致最终的特征图分辨率过低，不利于精确定位
2. 目标的多尺度问题

---

​		针对问题`1`，论文中给出的解决方案还是空洞卷积，此处不再赘述

​		针对问题`2`，论文首先列举了解决多尺度问题的四种方法：如下图所示

<img src="images\multi-scale.jpg" alt="shadow-multi-scale" style="zoom:50%;" />

​		这四种方法分别为：

> 1. `Image Pyramid`：将输入图片缩放到不同分辨率后输入网络，将得到的结果进行融合
> 2. `Encoder-Decoder`：将`Encoder`阶段获得的多尺度特征运用到`Decoder`阶段来恢复空间分辨率
> 3. 级联其他模块建模长距离依赖关系，如`Dense CRF`或叠加一些其他的卷积层
> 4. `Spatial Pyramid Pooling`：使用不同采样率和多种视野的卷积核，来捕获多尺度对象

​		这些方法的具体细节将在下文中介绍。

---

​		论文中提到，本次工作的成果在于以下几点：

1. 重新讨论了在级联模块和空间金字塔池化的框架下应用孔洞卷积，来有效地扩大卷积核的感受野，有效地结合多尺度的上下文信息。
2. 模块由具有不同采样率的孔洞卷积和`BN`层组成，这对于训练十分重要。同时，论文中也讨论了使用级联或并联的方式来部署`ASPP`模块。
3. 论文中提到，使用大采样率的`3×3孔洞卷积`会导致无法捕获图像边界的信息（边界效应），退化成为`1×1卷积`，解决方法是在`ASPP`模块中加入图像级特征。
4. 模型抛弃了前两代模型一直使用的`CRF`全连接层。
5. 最后，论文中详细介绍了模型的细节以及训练模型的经验，其中包括一种简单而有效的`Bootstrap`方法，用于处理稀有和精细注释的对象

---

​		利用上下文信息解决多尺度图像语义分割问题的四种方法介绍

1. 图像金字塔（`Image Pyramid`）

    <img src="images\image-pyramid.jpg" alt="shadow-image-pyramid" style="zoom:50%;" />

    ​		如上图所示，这种做法的具体手段是，使用同一个模型，不同尺度的输入。小尺度的输入特征可以建模长距离语义，而大尺寸的输入可以修正小目标的细节信息。可以通过拉普拉斯变换对输入图像进行变换，将不同尺度的输入图像输入到`DCNN`，并将所有比例的特征图加以融合。

    ​		这类模型的主要缺点是占有GPU内存较大，较大/深的`DCNN`网络无法使用，通常运用在推理阶段。

2. 编码器-解码器（`Encoder-Decoder`）

    <img src="images\encoder-decoder.jpg" alt="shadow-encoder-decoder" style="zoom:50%;" />

    ​		这类模型主要由两部分组成：`（a）`编码器中，特征映射的空间维度逐渐减小，从而可以捕获较长范围内的信息；`（b）`解码器中，目标的细节和空间维度逐渐得到恢复。例如，有人使用反卷积来学习对低分辨率特征响应进行上采样。`SegNet`复用编码器中的池化索引，学习额外的卷积层来平滑特征响应；`U-net`将编码器中的特征层通过跳跃连接添加到相应的解码器激活层中；`LRR`使用了一个拉普拉斯金字塔重建网络。最近，`RefineNet`等证明了基于编码-解码结构的有效性。这类模型也在对象检测的领域得到了应用。

3. 上下文模块（`Context module`）

    ​		这类模型的做法主要是增加了额外的模块，采用级联的方式，用来编码远距离上下文信息。一种有效的方法是合并`Dense CRF`到`DCNNs`中，共同训练`DCNN`和`CRF`

4. 空间金字塔池化（`Spatial Pyramid Pooling`）

    <img src="images\spatial-pyramid.jpg" alt="shadow-spatial-pyramid" style="zoom:50%;" />

    ​		如上图所示，空间金字塔池化可以在多个范围内捕捉上下文信息。`ParseNet`从不同图像等级的特征中获取上下文信息。`DeepLabv v2`提出了空洞卷积空间金字塔池化(`ASPP`)，使用不同采样率的并行空洞卷积层才捕获多尺度信息。`PSPNet`在不同网格尺度上执行空间池化，并在多个语义分割数据集上获得出色的性能。还有其他基于`LSTM`的方法聚合全局信息。

##### 模型细节

<img src="images\without-astrous.jpg" alt="shadow-without-astrous" style="zoom:50%;" />

<img src="images\with-astrous.jpg" alt="shadow-with-astrous" style="zoom:50%;" />

​		上面两幅图即为在`ResNet`网络基础上的改进。具体而言，可以取`ResNet`网络的`Block 4`复制`3`份级联到网络之后，在不使用孔洞卷积的情况下，模型随后的`output_stride = 256`，信息损失较为严重，而下图则是对最后几个`Block`使用了`孔洞卷积`，保证了最后模型的`output_stride = 16`。

<img src="images\table-1.jpg" alt="shadow-table-1" style="zoom:35%;" />

​		从上图可以看出，维持同样的模型结构，使用不同的孔洞卷积采样率，随着`output_stride`的增大，模型性能逐渐下降，当`output_stride = 8`时可以取得较好的性能，但是也会占用较多的内存，较为理想的情况是采用`output_stride = 16`，这样在注重模型性能的同时兼顾内存开销。

​		受到了采用不同大小网格层次结构的多重网格方法的启发，我们提出的模型在block4和block7中采用了不同的空洞率。

​		特别的，文章中定义$Multi\_Grid=(r_1,r_2,r_3)$为`block 4`到`block 7`内三个卷积层的`unit rates`。卷积层的最终空洞率等于`unit rate`和`corresponding rate`的乘积。例如，当`output_stride = 16` ，`Multi_Grid = (1, 2, 4)`，三个卷积就会在`block 4`有 `rates = 2 · (1, 2, 4) = (2, 4, 8) `。

​		在`ASPP`模块，相比于`DeepLab v2`，论文中做出的一大改进是在卷积层之后增加了`Batch Normalization`层。

<img src="images\astrous-rate.jpg" alt="shadow-astrous-rate" style="zoom:40%;" />

​		上图展示的是在一张`65×65`大小的特征图上使用`3×3`卷积，以及不同孔洞卷积采样率情况下卷积核有效权重（指权重应用于特征区域，而不是填充0的部分）数量变化情况，可以看出，随着采样率的增大，有效权重数量逐渐减小，接近于`1`，这时的`3×3`卷积核已经无法捕获整个图像上下文信息，而是退化为一个简单的`1×1`卷积核，因为此时只有中心点的权重才是有效的。

​		为了克服这个问题，并将全局上下文信息纳入模型，论文中使用了图像级特征。具体来说，在模型的最后一个特征图采用全局平均池化，将重新生成的图像级别的特征提供给带`256`个滤波器(和`BN`)的`1×1`卷积，然后双线性插值将特征提升到所需的空间维度。

​		最后，改进后的`ASPP`如下图所示

<img src="images\aspp-update.jpg" alt="shadow-aspp-update" style="zoom:50%;" />

（a）一个1×1的卷积与三个`3×3`的`rates=(6, 12, 18)`的空洞卷积，滤波器数量都为`256`，包含`BN`层。针对`output_stride=16`的情况

（b）使用到的图像级特征

##### 实验设置

- 模型采用预训练的`ResNet`作为`backbone`，配合使用空洞卷积控制输出步幅。
- `output_stride`定义为输入图像的分辨率与最终输出分辨率的比值。例如当输出步幅为`8`时，原`ResNet`的最后两个`block(block3和block4)`包含的空洞卷积的采样率为`r=2`和`r=4`
- 学习率继续采用`poly`策略，在初始基础上乘以$(1-\dfrac{iter}{max\_iter})^{power}$，其中$power=0.9$
- 为了大采样率的空洞卷积能够有效，需要较大的图片大小；否则，大采样率的空洞卷积权值就会主要用于`padding`区域; 在`Pascal VOC 2012`数据集的训练和测试中采用了`513`的裁剪尺寸



#### 模型总结

​		相比于`DeepLab v1`和`DeepLab v2`，`DeepLab v3`的巨大更新在于去除了之前的`CRF`结构，主要更新点在于对`ASPP`结构的改进，这一次，模型在`ASPP`结构中使用了并行的不同`rate`的孔洞卷积，同时也加入了一个`1X1`卷积与一个`Image Pooling`操作，来更多的提取有效信息，同时模型也引入了`Multi-Grids`结构，使得孔洞卷积有了更多的变化



---



### DeepLab v3+

#### 论文信息

> 论文名：`Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation`
>
> 作者：`Liang-Chieh Chen、Yukun Zhu、George Papandreou、Florian Schroff、Hartwig Adam`
>
> 代码：[Tensorflow实现](https://github.com/tensorflow/models/tree/master/research/deeplab)

#### 论文内容

##### Introduction

​		在这一部分，论文主要介绍了`DeepLab v3+`相比于`DeepLab v3`的改进：由于`DeepLab v3`得到最终的结果需要经过一个`8×`或`16×`的上采样操作，使得模型最终的分割边界并不十分准确，所以针对这一点，论文提出了几点改进：

1. 将`DeepLab v3`作为`Encoder`，使用`Encoder-Decoder`结构，使用`Encoder`模块来提取需要的信息，使用`Decoder`来恢复物体的边界信息
2. 引入`Depthwise separable atrous convolution `,有效减少计算开销与参数数量

##### 模型细节

*`Encoder-Decoer`模块*

<img src="images\deeplab-v3p.jpg" alt="shadow-deeplab-v3p" style="zoom:50%;" />

​		上图为`output_stride = 16`的模型结构，从图中可以看出，`DeepLab v3+`使用之前的`DeepLab v3`作为`Encoder`模块，将`Encoder`模块的输出经过`4X`上采样处理后与`backbone`提取到的低层次信息进行融合，之后经过卷积操作，再使用`4×`上采样恢复到原始尺寸大小，这样一来，相比于原始的`8×`或`16×`上采样，模型能够恢复更多的原始像素信息

*深度可分离孔洞卷积*

<img src="images\depth-separable-conv.jpg" alt="shadow-depth-separable-conv" style="zoom:33%;" />

​		深度可分离卷积，又叫分组卷积，可以视为`Depthwise conv （a）`与`Pointwise conv （b）`的结合，先使用`Depthwise conv`对原始特征图进行各个通道的独立卷积，之后使用`Pointwise conv`的`1×1`卷积将特征图调整到需要的通道数，这样一来能够有效的降低参数数量与计算开销，同时获得不错的性能。

​		在深度可分离卷积的基础上，对每个通道的卷积使用孔洞卷积，就可以得到`Atrous depthwise conv （c）`。

​		除了以上两点外，模型还提出使用`Aligned Xception`作为模型的`backbone`来提升模型性能，该部分网络结构如下图所示：

<img src="images\xception.jpg" alt="shadow-xception" style="zoom:50%;" />

##### 实验设置

相比于`DeepLab v3`，模型的改进点主要在于`backbone`与`output_stride`的设置

1. 论文中分别使用`ResNet-101`与`Xception`作为`backbone`进行实验，最后结果表明使用`Xception`模型的性能要优于`ResNet-101`

    <img src="images\resnet-vs-xception.jpg" alt="shadow-resnet-vs-xception" style="zoom:40%;" />

2. 相比于`DeepLab v3`中`output_stride = 8`的设置，模型这次使用了`output_stride = 16`

#### 模型总结

​		`DeepLab v3+`是在`DeepLab v3`基础上改进而成，主要变化在于引入了`Encoder-Decoder`模块来提升分割准确度，其次就是引入了`深度可分离卷积`，使得模型在减轻计算量、降低参数数量的同时保持模型的性能，成为了当时的`SOTA`模型