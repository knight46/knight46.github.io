const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const pageType = document.body.dataset.page || "home";
const manifest = window.CONTENT_MANIFEST || { blogs: [], album: [] };
const SITE_URL = "https://knight46.github.io";
const blogState = {
    query: "",
    category: "全部"
};

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function formatDate(dateString) {
    const date = new Date(dateString);
    if (Number.isNaN(date.getTime())) {
        return escapeHtml(dateString || "");
    }

    return new Intl.DateTimeFormat("zh-CN", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit"
    }).format(date);
}

function absoluteSiteUrl(path = "") {
    if (/^https?:\/\//.test(path)) {
        return path;
    }

    return `${SITE_URL}/${String(path).replace(/^\.?\//, "")}`;
}

function setMeta(selector, attribute, value) {
    const element = document.head.querySelector(selector);
    if (element) {
        element.setAttribute(attribute, value);
    }
}

function updatePageMeta({ title, description, url, image }) {
    if (title) {
        document.title = title;
        setMeta('meta[property="og:title"]', "content", title);
    }

    if (description) {
        setMeta('meta[name="description"]', "content", description);
        setMeta('meta[property="og:description"]', "content", description);
    }

    if (url) {
        setMeta('link[rel="canonical"]', "href", url);
        setMeta('meta[property="og:url"]', "content", url);
    }

    if (image) {
        setMeta('meta[property="og:image"]', "content", absoluteSiteUrl(image));
    }
}

function renderTagPills(container, tags, category = "") {
    if (!container) {
        return;
    }

    const pills = [];
    if (category) {
        pills.push({
            label: category,
            className: "category-pill"
        });
    }

    if (tags && tags.length) {
        tags.forEach((tag) => {
            pills.push({
                label: tag,
                className: ""
            });
        });
    }

    if (!pills.length) {
        container.innerHTML = "";
        return;
    }

    container.innerHTML = pills
        .map((pill) => `<span${pill.className ? ` class="${pill.className}"` : ""}>${escapeHtml(pill.label)}</span>`)
        .join("");
}

function renderMiniTags(tags, limit = 3) {
    if (!tags || !tags.length) {
        return "";
    }

    return `
        <div class="mini-tags">
            ${tags
                .slice(0, limit)
                .map((tag) => `<span>${escapeHtml(tag)}</span>`)
                .join("")}
        </div>
    `;
}

function renderInline(markdown) {
    const placeholders = [];
    let output = escapeHtml(markdown);

    output = output.replace(/`([^`]+)`/g, (_, code) => {
        const token = `@@CODE${placeholders.length}@@`;
        placeholders.push(`<code>${escapeHtml(code)}</code>`);
        return token;
    });

    output = output.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, alt, src) => {
        return `<figure><img src="${escapeHtml(src)}" alt="${escapeHtml(alt)}"></figure>`;
    });

    output = output.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, text, href) => {
        const isExternal = /^https?:\/\//.test(href);
        const target = isExternal ? ' target="_blank" rel="noreferrer"' : "";
        return `<a href="${escapeHtml(href)}"${target}>${escapeHtml(text)}</a>`;
    });

    output = output.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    output = output.replace(/__([^_]+)__/g, "<strong>$1</strong>");
    output = output.replace(/(^|[\s(])\*([^*]+)\*(?=[\s).,!?:;]|$)/g, "$1<em>$2</em>");
    output = output.replace(/(^|[\s(])_([^_]+)_(?=[\s).,!?:;]|$)/g, "$1<em>$2</em>");

    placeholders.forEach((replacement, index) => {
        output = output.replace(`@@CODE${index}@@`, replacement);
    });

    return output;
}

function renderMarkdown(markdown) {
    if (!markdown || !markdown.trim()) {
        return "<p>暂无内容。</p>";
    }

    const codeBlocks = [];
    const normalized = markdown
        .replace(/\r\n/g, "\n")
        .replace(/```([\w-]*)\n([\s\S]*?)```/g, (_, language, code) => {
            const token = `@@BLOCK${codeBlocks.length}@@`;
            codeBlocks.push({
                language,
                code
            });
            return token;
        })
        .trim();

    const blocks = normalized.split(/\n{2,}/).filter(Boolean);

    return blocks
        .map((block) => {
            const trimmed = block.trim();
            const codeTokenMatch = trimmed.match(/^@@BLOCK(\d+)@@$/);

            if (codeTokenMatch) {
                const codeBlock = codeBlocks[Number(codeTokenMatch[1])];
                const languageClass = codeBlock.language ? ` class="language-${escapeHtml(codeBlock.language)}"` : "";
                return `<pre><code${languageClass}>${escapeHtml(codeBlock.code.trimEnd())}</code></pre>`;
            }

            const headingMatch = trimmed.match(/^(#{1,6})\s+(.+)$/);
            if (headingMatch) {
                const level = headingMatch[1].length;
                return `<h${level}>${renderInline(headingMatch[2])}</h${level}>`;
            }

            if (/^([-*_]){3,}$/.test(trimmed.replace(/\s/g, ""))) {
                return "<hr>";
            }

            const quoteLines = trimmed.split("\n");
            if (quoteLines.every((line) => /^>\s?/.test(line))) {
                const content = quoteLines.map((line) => line.replace(/^>\s?/, "")).join("\n");
                return `<blockquote>${renderMarkdown(content)}</blockquote>`;
            }

            const unorderedLines = trimmed.split("\n");
            if (unorderedLines.every((line) => /^[-*+]\s+/.test(line))) {
                return `<ul>${unorderedLines
                    .map((line) => `<li>${renderInline(line.replace(/^[-*+]\s+/, ""))}</li>`)
                    .join("")}</ul>`;
            }

            const orderedLines = trimmed.split("\n");
            if (orderedLines.every((line) => /^\d+\.\s+/.test(line))) {
                return `<ol>${orderedLines
                    .map((line) => `<li>${renderInline(line.replace(/^\d+\.\s+/, ""))}</li>`)
                    .join("")}</ol>`;
            }

            const paragraphHtml = renderInline(trimmed).replace(/\n/g, "<br>");
            if (/^<figure>[\s\S]*<\/figure>$/.test(paragraphHtml)) {
                return paragraphHtml;
            }

            return `<p>${paragraphHtml}</p>`;
        })
        .join("");
}

function updateYear() {
    const yearTarget = document.getElementById("year");
    if (yearTarget) {
        yearTarget.textContent = String(new Date().getFullYear());
    }
}

function setupSmoothScroll() {
    document.querySelectorAll("[data-scroll]").forEach((link) => {
        link.addEventListener("click", (event) => {
            const targetSelector = link.getAttribute("href");
            if (!targetSelector || !targetSelector.startsWith("#")) {
                return;
            }

            const target = document.querySelector(targetSelector);
            if (!target) {
                return;
            }

            event.preventDefault();
            target.scrollIntoView({
                behavior: prefersReducedMotion ? "auto" : "smooth",
                block: "start"
            });
        });
    });
}

function setupReveal() {
    const items = document.querySelectorAll(".reveal");
    if (!items.length) {
        return;
    }

    if (!("IntersectionObserver" in window) || prefersReducedMotion) {
        items.forEach((item) => item.classList.add("is-visible"));
        return;
    }

    const observer = new IntersectionObserver(
        (entries, currentObserver) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) {
                    return;
                }

                entry.target.classList.add("is-visible");
                currentObserver.unobserve(entry.target);
            });
        },
        {
            threshold: 0.14
        }
    );

    items.forEach((item) => observer.observe(item));
}

function createBlogCard(item) {
    return `
        <a class="blog-card" href="blog.html?slug=${encodeURIComponent(item.slug)}">
            <h4 class="blog-card-title">${escapeHtml(item.title)}</h4>
            <div class="blog-meta">
                <span>${formatDate(item.date)}</span>
                <span>${escapeHtml(item.category || "未分类")}</span>
                <span>阅读全文</span>
            </div>
            <p class="blog-summary">${escapeHtml(item.summary)}</p>
            ${renderMiniTags(item.tags)}
        </a>
    `;
}

function getBlogCategories() {
    const preferredCategories = ["HPC", "AI", "CUDA 编程基础"];
    const existingCategories = new Set(manifest.blogs.map((item) => item.category || "未分类"));
    const orderedCategories = preferredCategories.filter((category) => existingCategories.has(category));
    const otherCategories = Array.from(existingCategories).filter((category) => !preferredCategories.includes(category));
    return ["全部", ...orderedCategories, ...otherCategories];
}

function filterBlogs() {
    const normalizedQuery = blogState.query.trim().toLowerCase();

    return manifest.blogs.filter((item) => {
        const matchesCategory = blogState.category === "全部" || (item.category || "未分类") === blogState.category;
        if (!matchesCategory) {
            return false;
        }

        if (!normalizedQuery) {
            return true;
        }

        const searchable = [
            item.title,
            item.summary,
            item.category,
            ...(item.tags || [])
        ]
            .filter(Boolean)
            .join(" ")
            .toLowerCase();

        return searchable.includes(normalizedQuery);
    });
}

function renderBlogCategoryFilter() {
    const filter = document.getElementById("blog-category-filter");
    if (!filter) {
        return;
    }

    filter.innerHTML = getBlogCategories()
        .map((category) => {
            const isActive = category === blogState.category;
            return `<button class="category-chip${isActive ? " is-active" : ""}" type="button" data-category="${escapeHtml(category)}">${escapeHtml(category)}</button>`;
        })
        .join("");
}

function renderBlogList() {
    const blogsList = document.getElementById("blogs-list");
    if (!blogsList) {
        return;
    }

    const visibleBlogs = filterBlogs();
    blogsList.innerHTML = visibleBlogs.length
        ? visibleBlogs.map(createBlogCard).join("")
        : '<div class="empty-state">没有找到匹配的文章。</div>';
}

function setupBlogTools() {
    const search = document.getElementById("blog-search");
    const filter = document.getElementById("blog-category-filter");

    renderBlogCategoryFilter();

    if (search) {
        search.addEventListener("input", () => {
            blogState.query = search.value;
            renderBlogList();
        });
    }

    if (filter) {
        filter.addEventListener("click", (event) => {
            const button = event.target.closest("[data-category]");
            if (!button) {
                return;
            }

            blogState.category = button.dataset.category;
            renderBlogCategoryFilter();
            renderBlogList();
        });
    }
}

function createAlbumCard(item) {
    return `
        <button class="album-card" type="button" data-album-slug="${escapeHtml(item.slug)}">
            <div class="album-card-image">
                <img src="${escapeHtml(item.image)}" alt="${escapeHtml(item.title)}">
            </div>
            <p class="detail-date">${formatDate(item.date)}</p>
            <h4 class="album-card-title">${escapeHtml(item.title)}</h4>
            <p class="album-summary">${escapeHtml(item.summary)}</p>
        </button>
    `;
}

function renderHomePage() {
    const blogsList = document.getElementById("blogs-list");
    const albumList = document.getElementById("album-list");

    if (blogsList) {
        setupBlogTools();
        renderBlogList();
    }

    if (albumList) {
        albumList.innerHTML = manifest.album.length
            ? manifest.album.map(createAlbumCard).join("")
            : '<div class="empty-state">随笔和照片还在整理中。</div>';
    }

    setupAlbumModal();
}

function setupImageFullscreen() {
    const shell = document.getElementById("image-fullscreen");
    const image = document.getElementById("image-fullscreen-img");
    if (!shell || !image) {
        return;
    }

    let previousBodyOverflow = "";
    const imageSelector = "#album-modal-image, #album-modal-markdown img, #detail-cover-image, #detail-markdown img";

    const close = () => {
        shell.hidden = true;
        image.removeAttribute("src");
        image.alt = "";
        document.body.style.overflow = previousBodyOverflow;
    };

    const open = (sourceImage) => {
        const source = sourceImage.currentSrc || sourceImage.src;
        if (!source) {
            return;
        }

        previousBodyOverflow = document.body.style.overflow;
        image.src = source;
        image.alt = sourceImage.alt || "";
        shell.hidden = false;
        document.body.style.overflow = "hidden";
    };

    document.addEventListener("dblclick", (event) => {
        const sourceImage = event.target.closest(imageSelector);
        if (!sourceImage) {
            return;
        }

        event.preventDefault();
        open(sourceImage);
    });

    shell.querySelectorAll("[data-close-fullscreen-image]").forEach((element) => {
        element.addEventListener("click", close);
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && !shell.hidden) {
            event.preventDefault();
            close();
        }
    });
}

function setupAlbumModal() {
    const modal = document.getElementById("album-modal");
    const albumList = document.getElementById("album-list");
    if (!modal || !albumList) {
        return;
    }

    const pageStack = document.querySelector(".page-stack");
    const modalWindow = modal.querySelector(".modal-window");
    const image = document.getElementById("album-modal-image");
    const date = document.getElementById("album-modal-date");
    const title = document.getElementById("album-modal-title");
    const tags = document.getElementById("album-modal-tags");
    const markdown = document.getElementById("album-modal-markdown");
    let currentAlbumIndex = -1;

    const setModalLayout = () => {
        const ratio = image.naturalWidth / Math.max(image.naturalHeight, 1);
        modalWindow.classList.remove("layout-landscape", "layout-portrait", "layout-square");

        if (ratio >= 1.18) {
            modalWindow.classList.add("layout-landscape");
            return;
        }

        if (ratio <= 0.86) {
            modalWindow.classList.add("layout-portrait");
            return;
        }

        modalWindow.classList.add("layout-square");
    };

    const closeModal = () => {
        modal.hidden = true;
        if (pageStack) {
            pageStack.style.overflow = "";
        }
    };

    const openModal = (item, index = manifest.album.indexOf(item)) => {
        currentAlbumIndex = index;
        modalWindow.classList.remove("layout-landscape", "layout-portrait", "layout-square");
        image.src = item.image;
        image.alt = item.title;
        date.textContent = formatDate(item.date);
        title.textContent = item.title;
        renderTagPills(tags, item.tags);
        markdown.innerHTML = renderMarkdown(item.markdown);
        markdown.scrollTop = 0;
        modal.hidden = false;
        if (pageStack) {
            pageStack.style.overflow = "hidden";
        }

        if (image.complete) {
            setModalLayout();
        }
    };

    const switchAlbum = (direction) => {
        if (modal.hidden || manifest.album.length < 2) {
            return;
        }

        const offset = direction === "next" ? 1 : -1;
        const nextIndex = (currentAlbumIndex + offset + manifest.album.length) % manifest.album.length;
        openModal(manifest.album[nextIndex], nextIndex);
    };

    image.addEventListener("load", setModalLayout);

    albumList.addEventListener("click", (event) => {
        const trigger = event.target.closest("[data-album-slug]");
        if (!trigger) {
            return;
        }

        const itemIndex = manifest.album.findIndex((entry) => entry.slug === trigger.dataset.albumSlug);
        const item = manifest.album[itemIndex];
        if (item) {
            openModal(item, itemIndex);
        }
    });

    modal.querySelectorAll("[data-album-nav]").forEach((element) => {
        element.addEventListener("click", () => switchAlbum(element.dataset.albumNav));
    });

    modal.querySelectorAll("[data-close-album]").forEach((element) => {
        element.addEventListener("click", closeModal);
    });

    document.addEventListener("keydown", (event) => {
        const fullscreenShell = document.getElementById("image-fullscreen");
        if (fullscreenShell && !fullscreenShell.hidden) {
            return;
        }

        if (event.key === "Escape" && !modal.hidden) {
            closeModal();
        }

        if (event.key === "ArrowLeft") {
            switchAlbum("prev");
        }

        if (event.key === "ArrowRight") {
            switchAlbum("next");
        }
    });
}

function renderBlogDetail() {
    const params = new URLSearchParams(window.location.search);
    const slug = params.get("slug");
    const article = manifest.blogs.find((item) => item.slug === slug);

    const titleTarget = document.getElementById("detail-title");
    const dateTarget = document.getElementById("detail-date");
    const tagsTarget = document.getElementById("detail-tags");
    const markdownTarget = document.getElementById("detail-markdown");
    const cover = document.getElementById("detail-cover");
    const coverImage = document.getElementById("detail-cover-image");

    if (!article) {
        document.title = "Article Not Found | AzathothLXL";
        titleTarget.textContent = "没有找到这篇文章";
        dateTarget.textContent = "NOT FOUND";
        markdownTarget.innerHTML = "<p>这篇文章暂时不可访问，请返回首页查看已有内容。</p>";
        renderTagPills(tagsTarget, []);
        return;
    }

    const articleUrl = `${SITE_URL}/blog.html?slug=${encodeURIComponent(article.slug)}`;
    updatePageMeta({
        title: `${article.title} | AzathothLXL`,
        description: article.summary || "AzathothLXL 的技术文章。",
        url: articleUrl,
        image: article.coverImage || "src/pictures/avatar.png"
    });
    titleTarget.textContent = article.title;
    dateTarget.textContent = formatDate(article.date);
    renderTagPills(tagsTarget, article.tags, article.category);
    markdownTarget.innerHTML = renderMarkdown(article.markdown);

    if (article.coverImage) {
        cover.hidden = false;
        coverImage.src = article.coverImage;
        coverImage.alt = article.title;
    }
}

updateYear();
setupSmoothScroll();
setupReveal();

if (pageType === "home") {
    renderHomePage();
}

if (pageType === "blog-detail") {
    renderBlogDetail();
}

setupImageFullscreen();
