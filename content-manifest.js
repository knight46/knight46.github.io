window.CONTENT_MANIFEST = {
    "generatedAt": "2026-05-28T16:59:03.015Z",
    "blogs": [
        {
            "slug": "CUDA Performance Programming 1",
            "title": "CUDA 高性能编程（1）：从分支发散、寄存器到编译目标",
            "date": "2026-05-29",
            "category": "CUDA 编程基础",
            "summary": "第一篇 CUDA 高性能编程学习笔记，梳理 warp 分支发散、寄存器压力与 local memory spill、PTX/SASS 编译目标，以及 P-state 对性能观察的影响。",
            "tags": [
                "CUDA",
                "GPU",
                "Performance",
                "Register",
                "PTX",
                "SASS"
            ],
            "coverImage": "",
            "markdown": "# CUDA 高性能编程（1）：从分支发散、寄存器到编译目标\n\nCUDA 优化很容易被讲成一组零散技巧：少写 `if`、少用变量、调 block size、打开某个编译选项。但真正写 kernel 时，更重要的是理解这些现象背后的执行模型：GPU 不是 32 个线程随意各跑各的，而是以 warp 为基本执行单位；寄存器不是无限的，变量太多会挤到更慢的内存层级；编译出来的设备代码也不是一种形态，PTX 和 SASS 对兼容性、确定性和性能分析都有不同影响。\n\n这篇是 CUDA 高性能编程系列的第一篇，先从四个基础但很常见的问题开始：分支发散、寄存器压力、PTX/SASS 编译目标，以及 NVIDIA GPU 的 P-state。\n\n![CUDA 性能基础框架图](blogs/CUDA%20Performance%20Programming%201/pic/cuda-performance-basics.svg)\n\n## 1. if-else 为什么可能让 warp 变慢\n\n先保留一个直观比喻：一个 warp 有 32 个线程，如果前 16 个线程走路径 A，后 16 个线程走路径 B，那么 GPU 没法在同一时刻让一半线程做加法、另一半线程做乘法。它更像是让同一个班级听同一个口令：先执行 A 路径，需要 B 的线程暂时不动；再执行 B 路径，需要 A 的线程暂时不动。两条路径都跑完后，warp 再汇合。\n\n这就是 warp 内分支发散。它影响性能的原因不是 `if-else` 这个语法本身慢，而是**同一个 warp 内的线程走了不同控制流路径**。如果一个 warp 里的 32 个线程条件判断结果一致，那么它们仍然只走同一条路径，通常不会产生这种串行化成本。对于很短的分支，编译器还可能使用 predication，把分支变成带谓词的指令执行，实际影响也会不同。\n\n一个典型例子如下：\n\n```cuda\n__global__ void branch_example(float *out, const float *x, const float *y, int n) {\n    int i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i >= n) {\n        return;\n    }\n\n    // 如果同一个 warp 内有些 i 为偶数、有些 i 为奇数，就会产生分支发散。\n    if ((i & 1) == 0) {\n        out[i] = x[i] + y[i];\n    } else {\n        out[i] = x[i] * y[i];\n    }\n}\n```\n\n这个 kernel 的条件是 `i & 1`，相邻线程通常一半走加法、一半走乘法，所以每个 warp 内会高度发散。相比之下，如果条件是按大块数据划分，例如前半个数组走 A、后半个数组走 B，那么只有边界附近的少数 warp 可能发散，其他 warp 内部条件一致，代价就小很多。\n\n### 如何观察和排查\n\n可以从三个层面看：\n\n- 用 Nsight Compute 看 branch efficiency、warp execution efficiency，以及 source view 里对应分支的执行情况。\n- 用 `nvdisasm` 或 Nsight Compute 的 SASS 视图看编译器是否真的生成了分支，还是把短分支转换成了 predicated instruction。\n- 用 microbenchmark 对比不同数据分布：随机条件、按 warp 对齐条件、按 block 对齐条件，通常能很快看出性能差异来自哪里。\n\n### 如何优化\n\n常见优化方向是让同一个 warp 内的线程尽量走同一路径：\n\n- 调整数据布局或任务分配，把相同分支的数据聚在一起。\n- 把严重分歧的任务拆成多个 kernel 或多个队列，分别处理不同路径。\n- 对很短的分支，尝试用条件表达式或算术变换减少控制流，但要看编译结果和指令数量。\n- 避免为了消灭一个小分支引入更多全局内存访问、同步或复杂调度。\n\n容易踩的坑是把“少写 `if`”当成绝对规则。真正要避免的是 warp 内控制流分裂，并且要看分支体长度、数据分布、编译器 predication 和实际硬件计数器。\n\n## 2. 寄存器不够时，变量会掉到 local memory\n\n矩阵乘法、卷积、stencil 这类 kernel 中，每个线程经常会保存多个中间变量。理想情况下，这些临时值放在寄存器里，访问很快。但每个 SM 的寄存器文件是有限资源：每个线程用得越多，同一时间能驻留的 warp 或 block 就可能越少；如果编译器发现寄存器放不下，还可能把某些变量 spill 到 local memory。\n\n这里的 “local” 很容易误导。CUDA 文档里的 local memory 不是离线程很近的一小块高速缓存，它的作用域是线程私有，但物理上位于片外内存路径上，访问成本接近 global memory，并且会受缓存、访存模式和带宽影响。如果频繁访问的中间变量被 spill 到 local memory，kernel 可能从计算受限变成访存受限。\n\n可以用一个简化的矩阵乘法片段理解寄存器压力：\n\n```cuda\n__global__ void matmul_tiny_tile(float *c, const float *a, const float *b, int n) {\n    int row = blockIdx.y * blockDim.y + threadIdx.y;\n    int col = blockIdx.x * blockDim.x + threadIdx.x;\n\n    float acc0 = 0.0f;\n    float acc1 = 0.0f;\n    float acc2 = 0.0f;\n    float acc3 = 0.0f;\n\n    for (int k = 0; k < n; ++k) {\n        float av = a[row * n + k];\n        float bv = b[k * n + col];\n        acc0 += av * bv;\n        acc1 += av * (bv + 1.0f);\n        acc2 += (av + 1.0f) * bv;\n        acc3 += (av - 1.0f) * (bv - 1.0f);\n    }\n\n    if (row < n && col < n) {\n        c[row * n + col] = acc0 + 0.01f * (acc1 + acc2 + acc3);\n    }\n}\n```\n\n真实的高性能 GEMM 会用 shared memory、寄存器 tiling、向量化加载和更复杂的调度。这里的重点只是：每个线程保存的 accumulator、临时加载值、地址计算变量越多，寄存器需求就越高。变量多不一定错，但必须确认它们没有把 occupancy 和 spill 推到糟糕的位置。\n\n### 如何观察和排查\n\n最直接的第一步是让 `ptxas` 输出每个 kernel 的资源使用：\n\n```bash\nnvcc -O3 --ptxas-options=-v -arch=sm_86 kernel.cu -o kernel\n```\n\n输出里要重点看 `reg` 和 `lmem`。`reg` 是每线程寄存器数量，`lmem` 是 local memory 使用量。还可以用：\n\n```bash\nnvcc -O3 -keep -arch=sm_86 kernel.cu\n```\n\n然后检查生成的 PTX 是否出现 `.local`、`ld.local`、`st.local`。不过要注意，最终寄存器分配发生在面向具体架构的后端阶段，PTX 里没有看到 local memory，也不代表最终 SASS 一定没有 spill。更可靠的做法是结合 ptxas report、Nsight Compute 的 local memory 指标和 occupancy 信息一起判断。\n\n### 如何优化\n\n优化寄存器压力时，不要只盯着“变量个数”，要盯着 live range，也就是某个值从生成到最后一次使用之间占着寄存器多久：\n\n- 删除不必要的临时变量，或者把只用一次的表达式合并。\n- 复用中间变量，缩短变量生命周期，让编译器更容易回收寄存器。\n- 调整线程块大小。更大的 block 不一定更快，可能让寄存器、shared memory 或调度资源成为限制。\n- 适当降低每线程 tile 大小，少做一点寄存器 tiling，换取更高 occupancy。\n- 用 Nsight Compute 看 achieved occupancy、eligible warps、local load/store、memory throughput，判断瓶颈是否真的来自寄存器压力。\n\n`-maxrregcount=N` 可以限制每线程最多使用多少寄存器，例如：\n\n```bash\nnvcc -O3 -arch=sm_86 -maxrregcount=64 --ptxas-options=-v kernel.cu -o kernel\n```\n\n但它不是万能优化。限制寄存器可能提升 occupancy，也可能迫使编译器把更多变量 spill 到 local memory，最后更慢。正确用法是把它当成实验旋钮：每次改一个值，记录 ptxas 的 `reg/lmem`、Nsight Compute 的 occupancy 和实际 kernel 时间。\n\n## 3. PTX 和 SASS：兼容性与确定性的取舍\n\n`nvcc` 生成设备代码时，常见两类形态是 PTX 和 cubin/SASS。\n\nPTX 可以理解为 NVIDIA GPU 的中间表示。它不是某一张具体 GPU 的最终机器码，而是面向虚拟架构的指令表示。好处是兼容性好：程序可以携带 PTX，运行时由驱动 JIT 编译到当前 GPU 的实际架构。代价是首次运行可能有 JIT 开销，并且最终机器码会受到驱动版本、目标 GPU 和 JIT 后端影响。\n\nSASS 是具体 GPU 架构上的低级机器码，通常封装在 cubin 里。它针对特定 `sm_XX` 架构生成，例如：\n\n```bash\nnvcc -O3 -arch=sm_86 kernel.cu -o kernel\n```\n\n这里 `sm_86` 表示生成面向 Ampere 某类实际架构的设备代码。SASS 的优点是性能更确定、部署时不依赖运行时 JIT 生成同一段核心机器码，也便于用 `nvdisasm` 做底层分析。缺点是和架构绑定，发给不兼容的 GPU 就不能直接运行。\n\n实际工程里经常会同时放入多个 code image：给当前主要部署架构准备 SASS，再额外带上 PTX 作为未来架构的 forward compatibility fallback。例如：\n\n```bash\nnvcc -O3 kernel.cu \\\n  --generate-code arch=compute_86,code=sm_86 \\\n  --generate-code arch=compute_86,code=compute_86\n```\n\n容易踩的坑有两个。第一，只放很老的 PTX 可能错过新架构特性；第二，只放某个 `sm_XX` 的 cubin 会让程序在其他 GPU 上失去兼容性。做性能分析时也要说清楚自己测的是哪一种路径：运行的是已经编好的 SASS，还是驱动从 PTX JIT 出来的代码。\n\n## 4. P-state：先确认 GPU 是否真的在高性能状态\n\nP-state 是 NVIDIA GPU 的性能状态等级，和核心频率、显存频率、电压、功耗策略有关。`nvidia-smi` 文档把 Performance State 描述为从 P0 到 P12 的范围，其中 P0 通常是最高性能状态，数字越大越偏向低功耗状态。\n\n观察方式很简单：\n\n```bash\nnvidia-smi --query-gpu=name,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv\n```\n\n做 benchmark 时，如果 GPU 长时间停在较低性能状态，或者频率因为温度、功耗、电源策略发生波动，那么 kernel 时间会混入硬件状态变化的影响。尤其是短 kernel、首次运行、桌面环境共享 GPU、笔记本独显和云主机限功耗场景，都可能让数据不稳定。\n\n优化 CUDA 代码前，至少要确认：\n\n- 是否做了 warm-up，排除了首次 JIT、缓存冷启动、GPU 升频延迟。\n- benchmark 期间 `pstate`、SM clock、memory clock 是否稳定。\n- 是否触发了 thermal throttling、power limit 或应用时钟限制。\n- 多次运行的方差是否足够小。\n\nP0 不等于“一定最快”，因为现代 GPU 还有 boost、功耗墙、温度墙、应用时钟等机制；但 P-state 是一个很好的信号，能提醒你不要把硬件频率变化误判为 kernel 优化效果。\n\n## 小结：先定位机制，再选择优化旋钮\n\n这一章可以归纳成四句话：\n\n- `if-else` 不一定慢，慢的是同一个 warp 内线程走不同路径导致控制流串行化。\n- 寄存器是高性能 kernel 的关键资源，过多临时变量可能降低 occupancy 或引发 local memory spill。\n- PTX 偏兼容，SASS/cubin 偏确定和贴近硬件，部署与分析时要知道自己运行的是哪条路径。\n- P-state 和频率会影响 benchmark 结果，做优化前要先保证观测环境稳定。\n\n真正的 CUDA 优化不是把某条经验规则套到所有 kernel 上，而是建立一条排查链路：看执行模型、看编译结果、看硬件计数器，再决定要改控制流、改变量生命周期、改 block 配置，还是改编译目标。\n\n## 参考资料\n\n- NVIDIA CUDA C++ Programming Guide: SIMT Architecture. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture>\n- NVIDIA CUDA C++ Best Practices Guide: Local Memory, Registers, Occupancy. <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>\n- NVIDIA CUDA Compiler Driver NVCC Documentation. <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>\n- NVIDIA System Management Interface Documentation. <https://docs.nvidia.com/deploy/nvidia-smi/index.html>\n- NVIDIA Nsight Compute Documentation. <https://docs.nvidia.com/nsight-compute/>"
        },
        {
            "slug": "GPU Initiated Communication",
            "title": "GPU-Initiated Communication：把通信控制权交还给 GPU",
            "date": "2026-05-28",
            "category": "未分类",
            "summary": "读几篇近期 HPC 与分布式 GPU 训练论文后的一篇方法笔记，重点解释 GICC 如何让 GPU 内核直接触发跨节点通信，从而减少 host 介入并提升计算-通信重叠。",
            "tags": [
                "High Performance Compute",
                "GPU",
                "Communication Overlap",
                "Distributed Systems"
            ],
            "coverImage": "",
            "markdown": "# GPU-Initiated Communication：把通信控制权交还给 GPU\n\n最近读高性能计算方法相关论文时，一个很清晰的趋势是：大规模 GPU 程序的瓶颈不再只是“单个 kernel 跑得够不够快”，而是**计算、通信、同步、资源回收能不能在正确的时间发生**。\n\n传统写法里，GPU 完成一段计算之后，经常需要 CPU/host 侧介入：检查状态、发起通信、推进 runtime、等待完成，再让 GPU 继续下一段工作。这个模式容易理解，也和 MPI 时代的程序结构很一致。但在现代 GPU 集群上，它会带来两个问题：\n\n1. kernel 与通信之间存在额外 launch 或 host progress 延迟；\n2. 细粒度数据依赖下，边界数据已经算出来了，却不能立刻被远端使用。\n\n2026 年 4 月提交的 GICC 论文正是在解决这个问题：让 GPU kernel 在 fast path 上直接触发 NIC 级别的通信与协调，把一部分原本由 host 驱动的动作下沉到 GPU 侧。\n\n![GICC 框架图](blogs/GPU%20Initiated%20Communication/pic/gicc-framework.svg)\n\n## 要解决的问题：GPU 算得很快，但通信控制还不够贴近计算\n\n在 stencil、PDE 求解、图计算或分布式深度学习里，很多操作都有类似结构：\n\n- 每个 GPU 负责一块局部数据；\n- 内部区域可以独立计算；\n- 边界区域算完后，要把 halo 或中间结果发给邻居；\n- 下一轮迭代依赖远端边界数据。\n\n如果把边界通信放到整个 kernel 结束之后再由 CPU 发起，那么通信就被人为推迟了。理想情况是：GPU 线程一旦算完某个边界块，就立刻触发对应的 network operation，同时其他线程继续算内部区域。这样通信时间可以被计算时间遮住，整体迭代更接近流水线。\n\n这个思想并不是第一次出现。FLUX 用 kernel fusion 和细粒度拆分来隐藏 GPU 间通信；Lagom 针对分布式大模型训练，尝试联合调节通信参数，让计算和通信资源保持平衡。GICC 的特点在于，它更靠近 HPC runtime 和网络层：它不是只调整通信参数，也不是只把算子融合，而是把“谁来发起通信”这件事从 host 移到 GPU kernel。\n\n## GICC 的核心方法：GPU 发起，NIC 执行，host 异步回收\n\nGICC 可以理解成三个层次的组合。\n\n第一层是 **GPU-triggered coordination**。GPU kernel 中的线程在计算到某个阶段后，不需要退出 kernel 等 CPU 判断，而是直接触发预先准备好的 NIC work。对 stencil 来说，边界区域一完成，就可以发起 halo exchange；内部区域的计算继续推进，形成更细粒度的 compute-communication overlap。\n\n第二层是 **decouple coordination from data movement**。通信语义和数据搬运被拆开：GPU 侧负责在正确的时刻触发协调动作，NIC 负责真正的数据传输。这避免了每次都让 host 重新参与 fast path，也减少了同步和锁带来的额外开销。\n\n第三层是 **asynchronous resource reclamation**。NIC 的工作队列和状态不是无限的，GPU 如果一直触发通信，runtime 必须能安全回收已完成的资源。GICC 的做法是让 NIC 完成后同时向 GPU 和 host 可见的位置写 completion 信息；一个轻量 host 线程在后台回收和重新布置 NIC 资源，但它不阻塞 GPU 的关键路径。\n\n换句话说，host 没有完全消失。它仍然负责资源管理和慢路径维护。但真正高频、低延迟、和 kernel 进度强相关的部分，尽量留在 GPU 与 NIC 之间完成。\n\n## 一个 stencil 迭代可以怎么跑\n\n用二维 stencil 举例，传统流程通常是：\n\n1. GPU 计算本地 tile；\n2. kernel 结束；\n3. host 发起 halo exchange；\n4. 等通信完成；\n5. 下一轮 kernel 开始。\n\nGICC 风格的流程更像：\n\n1. GPU kernel 开始处理 tile；\n2. 边界线程先完成某些 halo 数据；\n3. GPU 线程直接触发 NIC put/send；\n4. 内部线程继续计算，不等待 host；\n5. NIC 完成后写 completion；\n6. host 后台回收资源，下一轮触发仍可继续使用。\n\n这里的关键不是“通信更快”这么简单，而是**通信发生得更早**，并且不用把 GPU 的执行节奏切碎成大量 host 可见的阶段。\n\n## 为什么这对现代 HPC 系统重要\n\n论文特别提到 OFI-based interconnect，例如 HPE Slingshot。很多超级计算机使用这类网络，但 GPU kernel 不能天然、稳定地直接驱动分布式协调。InfiniBand 上虽然已经有一些 GPU-initiated communication 机制，但现有实现仍可能引入额外同步和锁。\n\nGICC 在 NVIDIA 和 AMD GPU、InfiniBand 与 Slingshot 上做了实现和评估。论文报告的结果包括：\n\n- 在 Slingshot 上，每次协调的延迟最高降低 229x；\n- weak scaling efficiency 最高提升 25%；\n- 在 InfiniBand 上，相比 NVSHMEM 的 put latency 最高降低到 1.95x；\n- 在 64 个 AMD MI250X GCD 的工业 stencil proxy 上，GPU-aware MPI 的通信时间比 GICC 高 52% 以上，而 GICC 的并行效率为 42%，MPI 为 35.4%。\n\n这些数字说明 GICC 更像一种 runtime 层面的路径优化：它不改变 stencil 本身的数学结构，却改变了计算和通信之间的调度关系。\n\n## 和 FLUX、Lagom 放在一起看\n\n把几篇论文放在一起看，会发现它们关注的是同一个大问题的不同侧面。\n\nFLUX 关注的是算子内部的细粒度切分与 kernel fusion。它把通信和依赖计算拆得更细，再融合进更大的 kernel 里，目标是在 GPU 内部尽可能隐藏通信延迟。\n\nLagom 关注的是分布式大模型训练中的通信参数调优。它用统一 cost model 和 priority-based search 来避免在巨大配置空间里暴力搜索，使计算和通信资源占用更加平衡。\n\nGICC 关注的是 runtime 和网络 fast path。它要解决的是：当 GPU 已经知道通信应该发生时，为什么还要让 CPU 来决定和推进？\n\n这三类方法可以形成一条很自然的路线：\n\n- 算子层：把计算和通信拆细、融合、重排；\n- 调度层：选择合适通信参数和并行策略；\n- runtime/网络层：减少 host 介入，让 GPU 更直接地驱动通信。\n\n## 我的理解：HPC 优化正在从“加速 kernel”走向“整理时间线”\n\n这篇论文给我的启发是，HPC 方法优化越来越像是在整理一条时间线。单点 kernel 优化仍然重要，但当程序跑到多 GPU、多节点和复杂网络上时，真正的性能损失经常来自空隙：\n\n- GPU 等 CPU 发起下一步；\n- CPU 等 GPU 暴露状态；\n- 通信等计算结束后才开始；\n- 资源回收挡在关键路径上；\n- runtime 的同步粒度比算法依赖更粗。\n\nGICC 的价值就在于缩短这些空隙。它让边界计算完成和网络传输发起之间的距离变短，让 host 从关键路径上退到后台，最终让程序更接近“边算边传”的理想状态。\n\n当然，这种方法也带来工程复杂度。开发者需要面对 NIC 资源有限、completion 可见性、GPU 与 host 内存一致性、不同网络后端差异等问题。它不是一个随手加几行代码就能得到的优化，而更像是 runtime 系统需要长期维护的一层能力。\n\n## 小结\n\n如果用一句话概括 GICC：它把分布式 GPU 程序里的通信触发从 CPU/host 侧前移到 GPU kernel 内部，让通信更早发生，并通过异步资源回收避免把管理成本塞回关键路径。\n\n对我来说，这类工作很适合作为理解现代 HPC 的入口：高性能不只是 FLOPS，也不只是带宽，而是让计算、通信、同步和资源生命周期都在尽可能合适的位置发生。\n\n## 参考资料\n\n- Baodi Shan, Mauricio Araya-Polo, Barbara Chapman. GICC: A High-Performance Runtime for GPU-Initiated Communication and Coordination in Modern HPC Systems. arXiv:2604.22126, 2026. <https://arxiv.org/abs/2604.22126>\n- Guanbin Xu et al. Lagom: Unleashing the Power of Communication and Computation Overlapping for Distributed LLM Training. arXiv:2602.20656, 2026. <https://arxiv.org/abs/2602.20656>\n- Li-Wen Chang et al. FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion. arXiv:2406.06858, 2024. <https://arxiv.org/abs/2406.06858>"
        }
    ],
    "album": [
        {
            "slug": "First Picture",
            "title": "First Picture",
            "date": "2026-04-14",
            "summary": "第一张留在站点里的照片记录，用来给这个还在生长的个人空间定下一个安静的起点。",
            "tags": [
                "Album",
                "Note",
                "First Frame"
            ],
            "image": "album/First%20Picture/asuka.png",
            "markdown": "# First Picture\n\n这是一张留给网站初始阶段的照片。\n\n我希望随笔区不只是把图片铺在页面上，而是让每一张图都带一点情绪、观察或者当时顺手记下来的想法。这样它才更像个人网站，而不是单纯的图库。\n\n## 一点随笔\n\n页面的背景、信息密度和浏览节奏，其实都会影响一张图被观看的方式。\n如果背景本身已经很强，前景信息就更需要克制、有边界，而且要有一点呼吸感。"
        },
        {
            "slug": "Neon Silence",
            "title": "Neon Silence",
            "date": "2026-04-14",
            "summary": "夜晚、霓虹和安静的反光适合放慢观看速度，也适合作为随笔区的冷色片段。",
            "tags": [
                "Album",
                "Night",
                "Note"
            ],
            "image": "album/Neon%20Silence/asuka.png",
            "markdown": "# Neon Silence\n\n夜晚、霓虹、玻璃和反光，会让城市显得比白天更安静。\n\n我更喜欢这种慢一点的浏览节奏。照片不需要急着解释自己，只要在页面里留下一点冷色的停顿就够了。\n\n## 备注\n\n这类照片很适合放在个人主页里，像是在技术文章之外保留一小块不需要证明什么的空间。"
        },
        {
            "slug": "Gym Reflection",
            "title": "Gym Reflection",
            "date": "2026-04-13",
            "summary": "训练之后身体很吵，脑子却会慢下来；这是一段关于节奏和恢复的小记录。",
            "tags": [
                "Album",
                "Gym",
                "Reflection"
            ],
            "image": "album/Gym%20Reflection/asuka.png",
            "markdown": "# Gym Reflection\n\n训练之后的状态很奇怪，身体很吵，但脑子反而会慢下来。\n\n有时候我会觉得，相册页里的随笔和博客页里的文章不应该是同一类文字。  \n博客更像整理过后的表达，而相册里的文字应该更靠近当时的感觉、更短、更轻，但又不至于只有一句话。\n\n## 继续展开一点\n\n训练记录和技术笔记很不一样。技术笔记追求可复现、可解释、可引用；训练后的随笔更像是在恢复呼吸时给自己留一个标记。\n\n如果以后这里慢慢积累起来，它应该会变成一种很轻的日常索引：不是为了总结生活，而是为了记住某些状态确实发生过。"
        },
        {
            "slug": "Workbench Noon",
            "title": "Workbench Noon",
            "date": "2026-04-12",
            "summary": "桌面、工具和模型零件组成一段偏安静的午间记录，像生活里稳定的小坐标。",
            "tags": [
                "Album",
                "Hobby",
                "Figure"
            ],
            "image": "album/Workbench%20Noon/asuka.png",
            "markdown": "# Workbench Noon\n\n桌面、工具、模型零件和中午偏白的光线，通常会组成一种很具体的安静感。\n\n这类图片放在个人网站里，其实是在补充“我是一个怎样的人”这件事，而不仅仅是给页面加些图。"
        },
        {
            "slug": "After Queue",
            "title": "After Queue",
            "date": "2026-04-11",
            "summary": "排队结束、游戏开始之前的短暂停顿，也是一种很适合被记录下来的日常瞬间。",
            "tags": [
                "Album",
                "Game",
                "Queue"
            ],
            "image": "album/After%20Queue/asuka.png",
            "markdown": "# After Queue\n\n排队结束之后，真正开始游戏的那一刻通常没有等待时想象得那么戏剧化。\n\n但那种从“还没开始”到“终于进去了”的微妙过渡，反而是很适合被记录下来的。"
        },
        {
            "slug": "Blue Evening",
            "title": "Blue Evening",
            "date": "2026-04-09",
            "summary": "蓝色傍晚适合做情绪参照：不太热烈，但足够让页面慢下来。",
            "tags": [
                "Album",
                "Evening",
                "Note"
            ],
            "image": "album/Blue%20Evening/asuka.png",
            "markdown": "# Blue Evening\n\n蓝色的傍晚很适合做背景，也很适合做网页的情绪参照。\n\n如果前景信息过重，这类背景会被完全压掉；如果前景太轻，阅读又会失去中心。  \n所以页面里的玻璃层既要透明，也要足够稳，像是轻轻压住背景的一层空气。"
        }
    ]
};
