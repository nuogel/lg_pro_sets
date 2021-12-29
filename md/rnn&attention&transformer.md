



### Self-Attention
- 如下图所示，b1、b2、b3、b4、用Self-Attention可以同时计算，而在RNN中不能够同时被计算，即解决RNN无法并行化的问题。
![img.png](util_imgs/img_7.png)
- 在把各个词送入Self-Attention之前先将其乘以一个特征矩阵，以特征化的表示各个单词，然后将其送入Self-Attention中（词嵌入的方法，Word embedding），即ai=Wxi，然后把不同的a乘上不同的矩阵W变为向量q（去匹配其他向量）、k（被匹配）、v（要被抽取出的information），如下图所示：
![img_1.png](util_imgs/img_8.png)
- 然后用每个向量q去对每个k做attention，这里公式除以根号d是为了平衡q和k乘的时候方差太大。如下图：
![img_2.png](util_imgs/img_9.png)
- 然后做Soft-max，如下图：
![img_3.png](util_imgs/img_10.png)
- 从下图可以看出输出的向量b1可以考虑整个句子的信息。
![img_4.png](util_imgs/img_11.png)
  ![img.png](util_imgs/img_14.png)
![img_5.png](util_imgs/img_12.png)
- 综上四步，其实整个框架就是下图这个样子（所有向量可以被平行化计算）
![img_6.png](util_imgs/img_13.png)
- multi-head Self-Attention
  ![img.png](util_imgs/img_16.png)
### Transformer
![img.png](util_imgs/img_15.png)
- word embedding & positional encoding
- Add与Layer Normalization\
 layer normalization 可以避免batch不同的影响（batch normalization），计算时与其他句子无关
  



