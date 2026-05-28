window.CONTENT_MANIFEST = {
    "generatedAt": "2026-05-28T13:14:46.825Z",
    "blogs": [
        {
            "slug": "GPU Initiated Communication",
            "title": "GPU-Initiated Communication：把通信控制权交还给 GPU",
            "date": "2026-05-28",
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
