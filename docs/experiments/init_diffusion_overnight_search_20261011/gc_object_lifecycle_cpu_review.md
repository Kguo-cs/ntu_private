400 次真实 native cache CPU 加载、200 个简化 PyG batch 的结果确认：全局关闭 GC 会留下 400 个临时 UnpicklerWrapper 类，全部具有 class/function/__class__ closure 循环；tracked objects 增加 5296（最后 snapshot 再增加2），Tensor 数量不变。每 batch 删除局部引用后调用真实 gc.collect(0)，该类累计为0，tracked objects 只增加72；自动 GC 最后残留11个。三组均未初始化 CUDA、未导入 TensorFlow、没有训练模型或 Lightning。

这个实验没有证明旧训练4.3GB RSS 增幅全部来自循环垃圾，也没有证明 gc0 能解决原生段错误。短实验 RSS 基本稳定，不能外推到百万文件加载。PyG storage 使用弱 parent 引用；dataset/token processor 没有历史 batch 列表。Diagnostics 记录纯浮点数，但其 capture 闭包与 model 有每阶段循环；Lightning 的 trainer/loop/model 也有强循环，全局禁用 GC 会让阶段 teardown 不完整。

建议仅在已完成 optimizer/backward、且不在反序列化内的 batch 边界测试真实 collector(0)，分别记录 allocated/reserved/RSS/count 和回收对象数。若 gc.collect 已被替换成 no-op，必须事先保存原 collector。年轻代回收不会释放已晋升的旧 trainer/model 循环；GC1/2和阶段 teardown 是另一个风险边界。需要用隔离 GPU pilot 的实际趋势验证，不改数据、模型、随机种子或评价定义。

一次内存中的合成 Torch 加载测得实际安装版本的临时类及其独占 class dict、MRO/bases tuple、find_class、closure cells/map、descriptor 的浅层合计下界为2672字节。若150万次加载全部残留这类垃圾，对应约3.73GiB Python 对象大小，说明数 GiB 量级在数量上可行；这不是实测 RSS 预测，未计共享 code/globals，也未用新真实场景文件。
