window.CONTENT_MANIFEST = {
    "generatedAt": "2026-04-14T14:54:44.500Z",
    "blogs": [
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
