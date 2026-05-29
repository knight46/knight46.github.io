import fs from "node:fs/promises";
import path from "node:path";

const rootDir = process.cwd();
const outputFile = path.join(rootDir, "content-manifest.js");
const siteUrl = "https://knight46.github.io";
const rssFile = path.join(rootDir, "rss.xml");
const sitemapFile = path.join(rootDir, "sitemap.xml");
const robotsFile = path.join(rootDir, "robots.txt");

const IMAGE_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif"]);

async function getDirectories(baseDir) {
    try {
        const entries = await fs.readdir(baseDir, { withFileTypes: true });
        return entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name);
    } catch {
        return [];
    }
}

function parseFrontMatter(markdown) {
    if (!markdown.startsWith("---\n")) {
        return {
            attributes: {},
            body: markdown
        };
    }

    const closingIndex = markdown.indexOf("\n---\n", 4);
    if (closingIndex === -1) {
        return {
            attributes: {},
            body: markdown
        };
    }

    const rawFrontMatter = markdown.slice(4, closingIndex).trim();
    const body = markdown.slice(closingIndex + 5).trim();
    const attributes = {};

    rawFrontMatter.split(/\r?\n/).forEach((line) => {
        const separatorIndex = line.indexOf(":");
        if (separatorIndex === -1) {
            return;
        }

        const key = line.slice(0, separatorIndex).trim();
        const value = line.slice(separatorIndex + 1).trim();
        attributes[key] = value;
    });

    return {
        attributes,
        body
    };
}

function toUrlPath(filePath) {
    const relativePath = path.relative(rootDir, filePath);
    return relativePath
        .split(path.sep)
        .map((segment) => encodeURIComponent(segment))
        .join("/");
}

function normalizeTags(rawTags) {
    if (!rawTags) {
        return [];
    }

    return rawTags
        .split(",")
        .map((tag) => tag.trim())
        .filter(Boolean);
}

function xmlEscape(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&apos;");
}

function sitePath(pathname = "") {
    return `${siteUrl}/${pathname.replace(/^\/+/, "")}`;
}

function blogUrl(blog) {
    return sitePath(`blog.html?slug=${encodeURIComponent(blog.slug)}`);
}

function stripMarkdown(markdown) {
    return markdown
        .replace(/```[\s\S]*?```/g, " ")
        .replace(/`([^`]+)`/g, "$1")
        .replace(/!\[[^\]]*\]\([^)]+\)/g, " ")
        .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
        .replace(/^#{1,6}\s+/gm, "")
        .replace(/^>\s?/gm, "")
        .replace(/^[-*+]\s+/gm, "")
        .replace(/^\d+\.\s+/gm, "")
        .replace(/[*_~]/g, "")
        .replace(/\r?\n+/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function extractSummary(markdown, fallbackTitle) {
    const cleaned = stripMarkdown(markdown);
    if (!cleaned) {
        return `${fallbackTitle} 的内容预览将在这里显示。`;
    }

    return cleaned.length > 120 ? `${cleaned.slice(0, 117)}...` : cleaned;
}

function rewriteRelativeMarkdownPaths(markdown, folderPath) {
    return markdown.replace(/(!?\[[^\]]*\]\()([^)]+)(\))/g, (match, prefix, target, suffix) => {
        const value = target.trim();
        if (
            value.startsWith("http://") ||
            value.startsWith("https://") ||
            value.startsWith("mailto:") ||
            value.startsWith("#") ||
            value.startsWith("/")
        ) {
            return match;
        }

        const absoluteTarget = path.resolve(folderPath, value);
        return `${prefix}${toUrlPath(absoluteTarget)}${suffix}`;
    });
}

async function readBlog(folderName) {
    const folderPath = path.join(rootDir, "blogs", folderName);
    const entries = await fs.readdir(folderPath, { withFileTypes: true });
    const markdownEntry = entries.find((entry) => entry.isFile() && entry.name.toLowerCase().endsWith(".md"));
    const picDir = entries.find((entry) => entry.isDirectory() && entry.name === "pic");

    if (!markdownEntry) {
        return null;
    }

    const markdownPath = path.join(folderPath, markdownEntry.name);
    const rawMarkdown = await fs.readFile(markdownPath, "utf8");
    const { attributes, body } = parseFrontMatter(rawMarkdown);
    const content = rewriteRelativeMarkdownPaths(body, folderPath);

    let coverImage = "";
    if (picDir) {
        const picEntries = await fs.readdir(path.join(folderPath, picDir.name), { withFileTypes: true });
        const imageEntry = picEntries.find(
            (entry) => entry.isFile() && IMAGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase())
        );
        if (imageEntry) {
            coverImage = toUrlPath(path.join(folderPath, picDir.name, imageEntry.name));
        }
    }

    return {
        slug: folderName,
        title: attributes.title || folderName,
        date: attributes.date || new Date().toISOString().slice(0, 10),
        category: attributes.category || "未分类",
        summary: attributes.summary || extractSummary(content, folderName),
        tags: normalizeTags(attributes.tags),
        coverImage,
        markdown: content
    };
}

async function readAlbum(folderName) {
    const folderPath = path.join(rootDir, "album", folderName);
    const entries = await fs.readdir(folderPath, { withFileTypes: true });
    const markdownEntry = entries.find((entry) => entry.isFile() && entry.name.toLowerCase().endsWith(".md"));
    const imageEntry = entries.find(
        (entry) => entry.isFile() && IMAGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase())
    );

    if (!markdownEntry || !imageEntry) {
        return null;
    }

    const markdownPath = path.join(folderPath, markdownEntry.name);
    const rawMarkdown = await fs.readFile(markdownPath, "utf8");
    const { attributes, body } = parseFrontMatter(rawMarkdown);
    const content = rewriteRelativeMarkdownPaths(body, folderPath);

    return {
        slug: folderName,
        title: attributes.title || folderName,
        date: attributes.date || new Date().toISOString().slice(0, 10),
        summary: attributes.summary || extractSummary(content, folderName),
        tags: normalizeTags(attributes.tags),
        image: toUrlPath(path.join(folderPath, imageEntry.name)),
        markdown: content
    };
}

function sortByDateDescending(left, right) {
    const leftTime = new Date(left.date).getTime();
    const rightTime = new Date(right.date).getTime();
    return rightTime - leftTime;
}

function buildRss(blogs) {
    const latestDate = blogs[0]?.date || new Date().toISOString().slice(0, 10);
    const items = blogs
        .map((blog) => {
            const url = blogUrl(blog);
            const categories = [blog.category, ...(blog.tags || [])]
                .filter(Boolean)
                .map((tag) => `<category>${xmlEscape(tag)}</category>`)
                .join("");

            return `        <item>
            <title>${xmlEscape(blog.title)}</title>
            <link>${xmlEscape(url)}</link>
            <guid>${xmlEscape(url)}</guid>
            <pubDate>${new Date(blog.date).toUTCString()}</pubDate>
            <description>${xmlEscape(blog.summary)}</description>
            ${categories}
        </item>`;
        })
        .join("\n");

    return `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>AzathothLXL Professional Articles</title>
        <link>${siteUrl}/</link>
        <description>高性能计算、GPU 系统、深度学习系统和相关研究笔记。</description>
        <language>zh-CN</language>
        <lastBuildDate>${new Date(latestDate).toUTCString()}</lastBuildDate>
${items}
    </channel>
</rss>
`;
}

function buildSitemap(blogs) {
    const staticUrls = [
        {
            loc: sitePath(""),
            lastmod: new Date().toISOString().slice(0, 10),
            priority: "1.0"
        }
    ];

    const blogUrls = blogs.map((blog) => ({
        loc: blogUrl(blog),
        lastmod: blog.date,
        priority: "0.8"
    }));

    const urls = [...staticUrls, ...blogUrls]
        .map((entry) => `    <url>
        <loc>${xmlEscape(entry.loc)}</loc>
        <lastmod>${xmlEscape(entry.lastmod)}</lastmod>
        <priority>${entry.priority}</priority>
    </url>`)
        .join("\n");

    return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls}
</urlset>
`;
}

function buildRobots() {
    return `User-agent: *
Allow: /

Sitemap: ${sitePath("sitemap.xml")}
`;
}

async function buildManifest() {
    const blogFolders = await getDirectories(path.join(rootDir, "blogs"));
    const albumFolders = await getDirectories(path.join(rootDir, "album"));

    const blogs = (await Promise.all(blogFolders.map(readBlog))).filter(Boolean).sort(sortByDateDescending);
    const album = (await Promise.all(albumFolders.map(readAlbum))).filter(Boolean).sort(sortByDateDescending);

    const manifest = {
        generatedAt: new Date().toISOString(),
        blogs,
        album
    };

    const output = `window.CONTENT_MANIFEST = ${JSON.stringify(manifest, null, 4)};\n`;
    await fs.writeFile(outputFile, output, "utf8");
    await fs.writeFile(rssFile, buildRss(blogs), "utf8");
    await fs.writeFile(sitemapFile, buildSitemap(blogs), "utf8");
    await fs.writeFile(robotsFile, buildRobots(), "utf8");

    console.log(
        `Generated ${path.basename(outputFile)}, ${path.basename(rssFile)}, ${path.basename(sitemapFile)}, and ${path.basename(robotsFile)} with ${blogs.length} blog(s) and ${album.length} album item(s).`
    );
}

buildManifest().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
