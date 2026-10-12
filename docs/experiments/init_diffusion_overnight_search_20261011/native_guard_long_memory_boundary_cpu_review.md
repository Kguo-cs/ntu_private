# 原生长训练运行边界与内存保护

缓存 Torch Unpickler 类不能证明完整训练内存有界。实际 candidate 日志从 step1 的约2.35GiB，到step14336约4.26GiB、step18432约4.77GiB仍在增长。此时显式GC与自动GC都禁用。128epoch共需要63,173,376次场景曝光，其运行可行性尚未验证，不能把短期20k CPU加载器内存稳定当成全训练保证。

`tools/run_init_diffusion_safe.py` 已增加默认父子进程保护，保留原Hydra参数：

```bash
python tools/run_init_diffusion_safe.py experiment=<实际best配置>
```

父进程只使用标准库，隔离启动一个自有child，监控child RSS上限18GiB和host MemAvailable下限1.5GiB，默认每2秒检查。可用`SIM_NATIVE_RUNTIME_AUDIT`指定新的审计JSON路径，默认写到`logs/native_runtime_guard/`；相邻`.child.json`记录当前固定loader源SHA和GC策略。可通过`SIM_NATIVE_MAX_RSS_GIB`、`SIM_NATIVE_MIN_AVAILABLE_GIB`、`SIM_NATIVE_MEMORY_POLL_SECONDS`显式调整阈值。

触发限制时只中断自己的child进程组，先SIGINT，10秒未退出再SIGKILL，持久记录实际止因、观测内存、信号和returncode。即使child在memorystop后返回0，整体仍失败。父进程中断、child非零退出、缺少真实successfulpolicy均不能标成成功。它不保证中断时有完整checkpoint，也没有自动或精确续训。监控是采样机制，RSS不是worker子树总和，短时峰值可能发生在检查间隔内。

8个guard测试与13个既有loader/native合同测试通过，包含真实CPU sleeping child因RSS限制被SIGINT终止。真实原生`--cfg job`父子运行返回0，只组合配置，没有实例化模型、加载checkpoint或启动GPU。

补充CPU探针用100个真实场景循环加载，共1000个B2batch、2000次场景加载，调用当前源`WaymoTargetBuilderTrain`和安装的PyG `Batch.from_data_list`。发现每Batch会新建一个局部`MetaResolver`类，即便已缓存Batch类；禁用GC后1000个类仍被跟踪。Tensor存活数保持1（基线占位对象），从第250至1000batch RSS仅增20KiB。这确认了G1之外另一个小的类循环来源，但不能解释生产运行的数GiB增长，更不能定位之前的native SIGSEGV根因。

运行中的entrypoint、helper、模型、数据和controller源文件均未修改。JSON保留实际日志采样、CPU探针、guard配置组合结果和源SHA。
