const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const pageType = document.body.dataset.page || "home";
const manifest = window.CONTENT_MANIFEST || { blogs: [], album: [] };

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

function renderTagPills(container, tags) {
    if (!container) {
        return;
    }

    if (!tags || !tags.length) {
        container.innerHTML = "";
        return;
    }

    container.innerHTML = tags
        .map((tag) => `<span>${escapeHtml(tag)}</span>`)
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
        <a class="blog-card" href="blog.html?slug=${encodeURIComponent(item.slug)}" target="_blank" rel="noreferrer">
            <h4 class="blog-card-title">${escapeHtml(item.title)}</h4>
            <div class="blog-meta">
                <span>${formatDate(item.date)}</span>
                <span>新窗口打开</span>
            </div>
            <p class="blog-summary">${escapeHtml(item.summary)}</p>
            ${renderMiniTags(item.tags)}
        </a>
    `;
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
        blogsList.innerHTML = manifest.blogs.length
            ? manifest.blogs.map(createBlogCard).join("")
            : '<div class="empty-state">还没有发现博客内容。后续在 `blogs/` 下添加文章文件夹后，再运行一次内容生成脚本即可。</div>';
    }

    if (albumList) {
        albumList.innerHTML = manifest.album.length
            ? manifest.album.map(createAlbumCard).join("")
            : '<div class="empty-state">还没有发现相册内容。后续在 `album/` 下添加图片目录后，再运行一次内容生成脚本即可。</div>';
    }

    setupAlbumModal();
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

    const openModal = (item) => {
        image.src = item.image;
        image.alt = item.title;
        date.textContent = formatDate(item.date);
        title.textContent = item.title;
        renderTagPills(tags, item.tags);
        markdown.innerHTML = renderMarkdown(item.markdown);
        modal.hidden = false;
        if (pageStack) {
            pageStack.style.overflow = "hidden";
        }

        if (image.complete) {
            setModalLayout();
        }
    };

    image.addEventListener("load", setModalLayout);

    albumList.addEventListener("click", (event) => {
        const trigger = event.target.closest("[data-album-slug]");
        if (!trigger) {
            return;
        }

        const item = manifest.album.find((entry) => entry.slug === trigger.dataset.albumSlug);
        if (item) {
            openModal(item);
        }
    });

    modal.querySelectorAll("[data-close-album]").forEach((element) => {
        element.addEventListener("click", closeModal);
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && !modal.hidden) {
            closeModal();
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
        markdownTarget.innerHTML = "<p>请确认文章存在，并在更新内容后重新运行一次内容生成脚本。</p>";
        renderTagPills(tagsTarget, []);
        return;
    }

    document.title = `${article.title} | AzathothLXL`;
    titleTarget.textContent = article.title;
    dateTarget.textContent = formatDate(article.date);
    renderTagPills(tagsTarget, article.tags);
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
