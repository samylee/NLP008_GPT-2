# NLP008_GPT-2
NLP008: gpt-2 using pytorch

## 使用说明
### 要求
> Python == 3.6.13 \
> PyTorch == 1.10.1  
### 数据集和已训练模型下载
[chinese_dialogue_instruction.parquet(提取码8888)](https://pan.baidu.com/s/1JplQ1R50eHZ5tUDdudF4Ow)  
[epoch_35_loss_1.03.pt(提取码8888)](https://pan.baidu.com/s/17SYSJ3Im0BQxXzyaWQIXMw)
### 训练
```shell script
python train.py
```
### 测试
```shell script
python predict.py  
```
```
Prompt:
张三：一位公司技术高管，项目经验丰富，对技术架构设计有独特的见解。 李四：一位公司底层技术职员，负责简单的架构设计和维护。 生成一段他们的对话内容。

Dialogue:  
张三：你好，李四，最近项目的进度如何？
李四：目前还不错，目前正在设计一个全新的架构，但还存在一些技术难题。
张三：听起来很有意思，能跟我详细说说吗？
李四：是的，这个架构比较单调，需要进行一些大规模的实验，如模型优化、测试等。
张三：我知道了，但我认为可行性还需要加强，我们的技术栈是否符合要求。
李四：这是个好问题，我已经在着手解决了。
张三：还有一个问题，你考虑过用什么技术来实现这个架构吗？
李四：我听说过一个比较流行的技术，但还不是很熟悉。我想请教一下你的建议。
张三：没问题，我可以为你提供一些资料和学习资料，帮你更好的理解和应用。
李四：非常感谢，我会认真考虑你的建议，希望我们能够更好地合作。
```
## 参考
https://github.com/samylee/NLP006_GPT-1   
https://blog.csdn.net/samylee  
