---
id: TASK-2
title: Build base Jinja2 UI template with System180 branding
status: To Do
assignee: []
created_date: '2026-03-10 21:31'
labels:
  - frontend
  - ui
milestone: m-0
dependencies:
  - TASK-1
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the shared HTML/CSS base template that both the Inventarization and Gamification apps extend. All UI work must work fully offline — no CDN dependencies.

**Stack:**
- Jinja2 templates (via FastAPI's `Jinja2Templates`)
- HTMX (downloaded, served from `static/js/htmx.min.js`)
- Tailwind CSS (built once with the standalone Tailwind CLI, output committed to `static/css/tailwind.css`)

**Template structure:**
```
src/demonstrator/
  templates/
    base.html           # Shared layout: header, nav slot, content slot, footer
    inventarization/
      base.html         # Extends base.html, inventarization-specific nav
    gamification/
      base.html         # Extends base.html, gamification-specific nav
  static/
    js/
      htmx.min.js
    css/
      tailwind.css      # Pre-built Tailwind output (committed)
    images/
      system180-logo.jpg   # copied from old/
      dfki-logo.png        # copied from old/
```

**`base.html` must include:**
- System180 logo + wordmark in the header
- A responsive top navigation bar with slots for app-specific links
- A `{% block content %}` main area
- HTMX and Tailwind loaded from local static files only
- A minimal health/status indicator in the header (e.g. "3 cameras connected" — static placeholder for now)

**FastAPI integration:**
- Mount `static/` directory at `/static`
- Add a `GET /` route in each app stub that renders `inventarization/index.html` (or gamification equivalent) as a basic index page
- No camera logic yet — just the layout with placeholder content

Depends on: task-1 (package skeleton must exist before templates can be wired to FastAPI routes).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Running the inventarization app and visiting `/` renders a page with the System180 logo and correct layout — no broken asset links
- [ ] #2 All assets (HTMX, Tailwind, logos) are served from `/static/`, zero requests to external URLs
- [ ] #3 Both app base templates (`inventarization/base.html`, `gamification/base.html`) extend `base.html` and override the nav slot correctly
- [ ] #4 Tailwind pre-built CSS is committed and the page is styled without running any build tool at runtime
- [ ] #5 Page is responsive: layout does not break at mobile, tablet, and desktop widths
- [ ] #6 HTMX is loaded and functional — a simple `hx-get` test request on the index page confirms it works offline
<!-- AC:END -->
