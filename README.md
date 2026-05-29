# AzathothLXL Personal Website

A lightweight static personal website for technical writing, research notes, photo essays, and profile information.

Live site: <https://knight46.github.io>

## Overview

This repository hosts the source code for AzathothLXL's personal website. The site is built with plain HTML, CSS, and JavaScript, without a frontend framework. It is designed to stay simple, fast, and easy to maintain while still providing a polished reading experience.

The current site includes:

- **Intro**: a personal profile and research focus.
- **Professional Articles**: technical blog posts about high performance computing, CUDA programming, GPU systems, distributed communication, computer vision, and related topics.
- **Photo Essays**: short life notes paired with images.
- **Resume**: a compact research and engineering profile with links to articles, GitHub, and email.
- **Contact**: email, GitHub, and Bilibili links.
- **Discovery**: article search, category filters, RSS, sitemap, and robots metadata for better navigation and indexing.

## Content Structure

Blog posts are stored under:

```text
blogs/<Post Name>/article.md
```

Each article can include a `pic/` directory for diagrams or images:

```text
blogs/<Post Name>/pic/
```

Photo essay entries are stored under:

```text
album/<Entry Name>/note.md
album/<Entry Name>/<image-file>
```

The browser reads generated content from:

```text
content-manifest.js
```

The build script also generates:

```text
rss.xml
sitemap.xml
robots.txt
```

## Adding Or Updating Content

After adding or editing Markdown content, regenerate the manifest:

```bash
node scripts/build-content.mjs
```

Then run a basic syntax check:

```bash
node --check script.js
node --check scripts/build-content.mjs
```

## Tech Stack

- HTML
- CSS
- Vanilla JavaScript
- Markdown-based content folders
- GitHub Pages deployment

## Repository Goal

The goal of this repository is to maintain a quiet, readable, and long-lived personal knowledge space. Professional articles focus on technical understanding and method notes, while photo essays keep a lighter record of everyday life.
