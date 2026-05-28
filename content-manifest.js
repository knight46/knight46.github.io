window.CONTENT_MANIFEST = {
    "generatedAt": "2026-05-28T12:56:07.336Z",
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
        },
        {
            "slug": "First Blog",
            "title": "First Blog",
            "date": "2026-04-14",
            "summary": "一篇用于验证博客列表、Markdown 渲染和文章详情页流程是否完整的测试文章。",
            "tags": [
                "High Performance Compute",
                "Computer Vision",
                "Quantam Chemistry Compute"
            ],
            "coverImage": "blogs/First%20Blog/pic/asuka.png",
            "markdown": "# First Blog\n\n这是一篇测试文章，用来确认个人网站里的 **blogs 模块** 已经具备以下能力：\n\n- 在首页的博客列表中自动读取并展示\n- 根据创建时间排序，优先显示最新内容\n- 点击卡片后在新窗口打开文章详情页\n- 把 Markdown 正常渲染成标题、段落、列表、引用和图片\n\n## 为什么要先做内容结构\n\n对个人网站来说，页面设计只是第一层。真正决定它能不能长期维护的，是内容目录结构是否清晰。\n\n把每篇文章放进独立文件夹之后：\n\n1. 文章正文可以单独维护\n2. 插图资源可以和正文放在一起\n3. 以后要做标签、分类、搜索时，也更容易扩展\n\n> 一个好维护的主页，应该让“继续更新”变得很轻松。\n\n## 这篇文章的配图\n\n下面这张图来自当前项目的测试素材，用来验证 Markdown 图片的渲染路径是否正确：\n\n![Asuka 测试配图](blogs/First%20Blog/pic/asuka.png)\n\n## 后续计划\n\n接下来你可以继续往 `blogs` 目录里添加新的文章文件夹。每个文件夹里放一篇 Markdown 和一个 `pic` 目录，再运行一次内容生成脚本，首页就能同步更新。"
        },
        {
            "slug": "HPC Notes",
            "title": "HPC Notes",
            "date": "2026-04-12",
            "summary": "一篇测试用文章，主要用来观察 blogs 面板在内容逐渐增多之后的排列、滚动和详情页打开体验。",
            "tags": [
                "High Performance Compute",
                "Parallel",
                "Note"
            ],
            "coverImage": "blogs/HPC%20Notes/pic/asuka.png",
            "markdown": "# HPC Notes\n\n这篇文章是为了测试 blogs 区块在文章逐渐变多以后，是否还能维持清晰的层次和稳定的滚动体验。\n\n## 观察重点\n\n- 卡片列表是否仍然容易浏览\n- 新窗口打开文章详情页时，阅读节奏是否自然\n- Markdown 标题、列表和图片是否都能正确渲染\n\n## 一点记录\n\n高性能计算对我来说并不只是“更快地跑完程序”，更像是在面对一个规模上升之后仍然要保持秩序的问题。\n\n当问题尺寸变大之后，真正难的往往不是写出第一版代码，而是让它在数据、资源和时间约束下继续保持可控。\n\n![Test Image](blogs/HPC%20Notes/blogs/HPC%2520Notes/pic/asuka.png)\n\n## 继续补充\n\n后面这里可以替换成你真正的实验记录、读论文笔记，或者一些关于并行计算和性能调优的想法。"
        },
        {
            "slug": "Vision Log",
            "title": "Vision Log",
            "date": "2026-04-10",
            "summary": "用于测试多文章状态下的 blogs 页面展示，同时模拟一篇偏图像和观察记录风格的文章。",
            "tags": [
                "Computer Vision",
                "Image",
                "Log"
            ],
            "coverImage": "blogs/Vision%20Log/pic/asuka.png",
            "markdown": "# Vision Log\n\n这一篇更像视觉方向的工作笔记。\n\n有些时候我会觉得，图像任务的困难之处不只是模型本身，而是如何在噪声、数据偏差和任务目标之间找到一个不会太脆弱的平衡点。\n\n## 为什么保留这类文章\n\n个人网站里的 blog 如果只有“正式文章”，更新频率通常会很低。  \n如果允许自己保留这种偏日志式、观察式的文字，整个系统会更容易长期维护下去。\n\n## 小结\n\n页面里的博客区应该能够同时承载：\n\n1. 比较完整的正式文章\n2. 短一些的实验笔记\n3. 以后按标签分类的扩展空间"
        },
        {
            "slug": "Chem Compute Memo",
            "title": "Chem Compute Memo",
            "date": "2026-04-08",
            "summary": "模拟一篇计算化学方向的备忘，用来测试列表排序、摘要截断和详情页在长段落下的阅读感受。",
            "tags": [
                "Quantam Chemistry Compute",
                "Memo",
                "Simulation"
            ],
            "coverImage": "blogs/Chem%20Compute%20Memo/pic/asuka.png",
            "markdown": "# Chem Compute Memo\n\n这是一个偏备忘性质的测试页面。\n\n计算化学相关内容通常有自己独特的术语体系和表达节奏，所以我希望博客详情页在排版上能足够稳定，不会因为一段文字稍长或者出现几张图就显得拥挤。\n\n## 这次测试的排版目标\n\n- 长段落是否仍然容易阅读\n- 二级标题是否能把信息切开\n- 图片、引用和列表能否放在同一篇文章中和平共处\n\n> 页面不是单纯地把文字摆上去，而是要给文字留出呼吸的位置。\n\n如果这一套阅读体验是顺的，后面继续补真正的文章就会自然很多。"
        }
    ],
    "album": [
        {
            "slug": "First Picture",
            "title": "First Picture",
            "date": "2026-04-14",
            "summary": "第一张测试图片，用来验证相册瀑布流预览、内部弹窗和 Markdown 随笔渲染是否都能正常工作。",
            "tags": [
                "Test",
                "Album",
                "Note"
            ],
            "image": "album/First%20Picture/asuka.png",
            "markdown": "# First Picture\n\n这是一张用于测试相册系统的图片。\n\n我希望相册不只是把图片铺在页面上，而是让每一张图片都带一点情绪、观察或者随手记下来的想法。这样它才更像个人网站，而不是单纯的图库。\n\n## 这次测试的重点\n\n- 首页相册区是否能正确读取图片与随笔\n- 点击图片后是否会在当前窗口弹出次级内部窗口\n- 弹窗里的 Markdown 是否能渲染成清晰的阅读内容\n\n## 一点随笔\n\n页面的背景、信息密度和浏览节奏，其实都会影响一张图被观看的方式。  \n如果背景本身已经很强，那前景信息就更需要克制、有边界，而且要有一点呼吸感。"
        },
        {
            "slug": "Neon Silence",
            "title": "Neon Silence",
            "date": "2026-04-14",
            "summary": "一条更偏夜晚感的测试随笔，用来观察相册卡片变多之后的瀑布流排列是否还足够松弛。",
            "tags": [
                "Album",
                "Night",
                "Note"
            ],
            "image": "album/Neon%20Silence/asuka.png",
            "markdown": "# Neon Silence\n\n这条随笔主要用来测试相册面板内容变多以后，卡片之间的留白是否还舒服。\n\n如果一个相册页面只是单纯把图片一张张堆上去，浏览节奏通常会很僵硬。  \n我更希望它像一段慢一点的浏览过程，而不是快速扫过的缩略图墙。\n\n## 备注\n\n夜晚、霓虹、玻璃和反光其实很适合当前这个页面的整体风格，所以这里故意保留一点偏冷、偏静的叙述方式。"
        },
        {
            "slug": "Gym Reflection",
            "title": "Gym Reflection",
            "date": "2026-04-13",
            "summary": "一条用来测试内部弹窗滚动的较长文字，希望在相册详情里也能保留稳定的阅读节奏。",
            "tags": [
                "Album",
                "Gym",
                "Reflection"
            ],
            "image": "album/Gym%20Reflection/asuka.png",
            "markdown": "# Gym Reflection\n\n训练之后的状态很奇怪，身体很吵，但脑子反而会慢下来。\n\n有时候我会觉得，相册页里的随笔和博客页里的文章不应该是同一类文字。  \n博客更像整理过后的表达，而相册里的文字应该更靠近当时的感觉、更短、更轻，但又不至于只有一句话。\n\n## 继续展开一点\n\n这个弹窗被设计成当前页面里的次级窗口，所以它不适合做得太重。  \n但如果文字一多，又必须保证它自己能独立滚动，否则体验会非常卡。\n\n因此这一条内容特地写长一点，就是为了测试：\n\n1. 弹窗标题和图片是否还能稳定停在上面\n2. 文字区域在内容增多后能否继续往下滚\n3. 手机端打开时，会不会挤到看不清\n\n如果这些都顺了，相册这个模块就不只是“能看图”，而是真的可以承载图片和文字一起存在。"
        },
        {
            "slug": "Workbench Noon",
            "title": "Workbench Noon",
            "date": "2026-04-12",
            "summary": "一条偏安静的测试随笔，用来模拟拼胶或者桌面物件相关的图片记录。",
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
            "summary": "用来测试相册预览摘要截断和卡片点击反馈的一条记录，也顺便模拟游戏相关的小随笔。",
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
            "summary": "这条测试内容主要用来继续拉长相册列表，观察在手机和桌面上滚动时的整体稳定性。",
            "tags": [
                "Album",
                "Evening",
                "Test"
            ],
            "image": "album/Blue%20Evening/asuka.png",
            "markdown": "# Blue Evening\n\n蓝色的傍晚很适合做背景，也很适合做网页的情绪参照。\n\n如果前景信息过重，这类背景会被完全压掉；如果前景太轻，阅读又会失去中心。  \n所以页面里的玻璃层既要透明，也要足够稳，像是轻轻压住背景的一层空气。"
        }
    ]
};
