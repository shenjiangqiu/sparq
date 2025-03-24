- 问题，map_dataset运行非常诡异，似乎是内存错误。 解决方法： 转移到husky4开发
- 问题： split 长度差别很大：方法： batch size 设置为1
- 问题： sparsity 的设置方法每个方法不同， 方法，统一成Alisa的那种。 根据长度决定。
- 整理数据
- 更新proposal 数据
- floatpim 的数据细节

## todo
- 完成实验：1. 测试完成后规范数据结构收集数据结构。2.A100部署运行完后提取分析数据到exel。 3.在overleave上记录数据，记录数据后背的设置，数据代表的意义。
- 更新proposal的数据： 重新运行实验（1.确保多个SA同时运行正确。2.确保公平分配task配置正确，3，确保offloading traffic 正确），检查记录完成后输出实验数据，整理到overleave
- 完成实验的时序分析：1.分析dense，sparseQ，Alisa，OurMethod 在传统PE（SAMSUNG）下的数据。2.分析在dense 在Float-Bitserial下的数据。 3.分析OurMethod在float——bitserial下的速度提升