"""
IUB Website Scraper — scraper.py
Handles: all pages, sublinks, JavaScript, popups, PDFs, dropdowns
Run: python scraper.py
"""

import os
import asyncio
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import fitz  # PyMuPDF for PDF extraction

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASE_URL        = "https://www.iub.edu.pk"
MAX_PAGES       = 500          # max HTML pages to scrape
MAX_PDFS        = 100          # max PDFs to download and extract
REQUEST_DELAY   = 1.5          # seconds between requests (be polite)
OUTPUT_DIR      = Path("scraped_data")
PDF_DIR         = Path("scraped_pdfs")
# ──────────────────────────────────────────────────────────────────────────────

VISITED_URLS  = set()
VISITED_PDFS  = set()
ALL_PAGE_DATA = []
ALL_PDF_DATA  = []

def is_same_domain(url: str) -> bool:
    try:
        return urlparse(url).netloc == urlparse(BASE_URL).netloc
    except:
        return False

def is_pdf_url(url: str) -> bool:
    return url.lower().endswith('.pdf') or 'pdf' in urlparse(url).path.lower()

def is_skippable(url: str) -> bool:
    skip_exts = ('.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico',
                 '.css', '.js', '.woff', '.woff2', '.ttf', '.mp4',
                 '.zip', '.rar', '.exe', '.ppt', '.pptx', '.xls', '.xlsx')
    return any(url.lower().endswith(ext) for ext in skip_exts)

def clean_html_text(soup: BeautifulSoup) -> str:
    """Remove noise and extract clean readable text from HTML."""
    for tag in soup(['script', 'style', 'nav', 'footer',
                     'header', 'iframe', 'noscript', 'aside',
                     'form', 'button', 'input']):
        tag.decompose()

    lines = []
    for elem in soup.find_all(['h1','h2','h3','h4','h5','h6','p','li','td','th','span','div']):
        text = elem.get_text(separator=' ', strip=True)
        if text and len(text) > 20:
            lines.append(text)

    # Deduplicate while preserving order
    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    return '\n'.join(unique_lines)

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(str(pdf_path))
        text_parts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{text.strip()}")
        doc.close()
        return '\n\n'.join(text_parts)
    except Exception as e:
        print(f"  PDF extraction failed: {e}")
        return ""

def download_pdf(url: str) -> str:
    """Download a PDF and return local path."""
    try:
        PDF_DIR.mkdir(exist_ok=True)
        filename = urlparse(url).path.split('/')[-1]
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        # sanitize filename
        filename = "".join(c for c in filename if c.isalnum() or c in '._-')
        local_path = PDF_DIR / filename

        if local_path.exists():
            return str(local_path)

        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return str(local_path)
    except Exception as e:
        print(f"  PDF download failed: {url} — {e}")
        return ""

async def scrape_single_page(page, url: str) -> dict:
    """Scrape one HTML page — handles JS, popups, dropdowns, lazy loading."""
    try:
        # Navigate and wait for full load
        await page.goto(url, wait_until='networkidle', timeout=30000)
        await page.wait_for_timeout(2000)

        # Close popups/modals by pressing Escape
        await page.keyboard.press('Escape')
        await page.wait_for_timeout(500)

        # Click away any overlay
        try:
            await page.mouse.click(10, 10)
            await page.wait_for_timeout(300)
        except:
            pass

        # Hover over navigation menus to reveal dropdown links
        nav_items = await page.query_selector_all('nav a, .navbar a, .menu a, .nav-item')
        for item in nav_items[:20]:
            try:
                await item.hover()
                await page.wait_for_timeout(200)
            except:
                pass

        # Slow scroll to trigger lazy-loaded content
        await page.evaluate("""
            async () => {
                await new Promise(resolve => {
                    let scrolled = 0;
                    const step = 400;
                    const delay = 150;
                    const timer = setInterval(() => {
                        window.scrollBy(0, step);
                        scrolled += step;
                        if (scrolled >= document.body.scrollHeight) {
                            clearInterval(timer);
                            resolve();
                        }
                    }, delay);
                });
            }
        """)
        await page.wait_for_timeout(1000)

        # Scroll back to top
        await page.evaluate("window.scrollTo(0, 0)")

        # Get fully rendered HTML
        html = await page.content()
        soup = BeautifulSoup(html, 'html.parser')

        title = soup.title.string.strip() if soup.title else url
        content = clean_html_text(soup)

        # Collect all links from this page
        links = []
        pdf_links = []

        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href', '').strip()
            if not href or href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
                continue

            full_url = urljoin(url, href)

            if not is_same_domain(full_url):
                continue

            if is_pdf_url(full_url):
                if full_url not in VISITED_PDFS:
                    pdf_links.append(full_url)
            elif not is_skippable(full_url):
                if full_url not in VISITED_URLS:
                    links.append(full_url)

        return {
            "success": True,
            "title": title,
            "content": content,
            "links": list(set(links)),
            "pdf_links": list(set(pdf_links))
        }

    except Exception as e:
        print(f"  FAILED: {url} — {e}")
        return {"success": False, "links": [], "pdf_links": []}

async def run_scraper():
    print("=" * 60)
    print("IUB Website Scraper — Full Coverage")
    print(f"Base URL: {BASE_URL}")
    print(f"Max pages: {MAX_PAGES} | Max PDFs: {MAX_PDFS}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(exist_ok=True)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-blink-features=AutomationControlled'
            ]
        )

        context = await browser.new_context(
            user_agent=(
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
            viewport={'width': 1280, 'height': 800}
        )

        # Block images, fonts, media to speed up scraping
        await context.route(
            "**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2,ttf,mp4,webp}",
            lambda route: route.abort()
        )

        page = await context.new_page()
        page.set_default_timeout(30000)

        queue = [BASE_URL]
        pdf_queue = []

        # ── SCRAPE HTML PAGES ──────────────────────────────────────────────
        print("\n[PHASE 1] Scraping HTML pages...\n")

        while queue and len(VISITED_URLS) < MAX_PAGES:
            url = queue.pop(0)

            if url in VISITED_URLS:
                continue

            VISITED_URLS.add(url)
            count = len(VISITED_URLS)
            print(f"[{count}/{MAX_PAGES}] {url[:80]}")

            result = await scrape_single_page(page, url)

            if result["success"] and result.get("content"):
                content = result["content"]
                if len(content) > 300:
                    ALL_PAGE_DATA.append({
                        "type": "webpage",
                        "url": url,
                        "title": result["title"],
                        "content": content,
                        "scraped_at": datetime.now().isoformat()
                    })
                    print(f"  OK — {result['title'][:60]} ({len(content)} chars)")
                else:
                    print(f"  SKIP — too little content")

            # Add new links to queue
            for link in result.get("links", []):
                if link not in VISITED_URLS:
                    queue.append(link)

            # Add PDF links to PDF queue
            for pdf_link in result.get("pdf_links", []):
                if pdf_link not in VISITED_PDFS:
                    pdf_queue.append(pdf_link)

            await asyncio.sleep(REQUEST_DELAY)

        await browser.close()

    # ── DOWNLOAD AND EXTRACT PDFs ──────────────────────────────────────────
    print(f"\n[PHASE 2] Processing {len(pdf_queue)} PDFs found...\n")

    for i, pdf_url in enumerate(pdf_queue):
        if len(VISITED_PDFS) >= MAX_PDFS:
            print(f"Reached PDF limit ({MAX_PDFS})")
            break

        if pdf_url in VISITED_PDFS:
            continue

        VISITED_PDFS.add(pdf_url)
        print(f"[{i+1}/{len(pdf_queue)}] Downloading PDF: {pdf_url[-60:]}")

        local_path = download_pdf(pdf_url)
        if not local_path:
            continue

        text = extract_pdf_text(Path(local_path))
        if len(text) > 200:
            filename = Path(local_path).name
            ALL_PDF_DATA.append({
                "type": "pdf",
                "url": pdf_url,
                "title": filename.replace('.pdf', '').replace('_', ' ').replace('-', ' '),
                "content": text,
                "local_file": local_path,
                "scraped_at": datetime.now().isoformat()
            })
            print(f"  OK — {len(text)} chars extracted")
        else:
            print(f"  SKIP — PDF has no readable text (may be scanned image)")

        time.sleep(1)

    # ── SAVE ALL DATA ──────────────────────────────────────────────────────
    print("\n[PHASE 3] Saving all data...\n")

    all_data = ALL_PAGE_DATA + ALL_PDF_DATA
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    # Save combined JSON
    combined_file = OUTPUT_DIR / f"iub_all_data_{timestamp}.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    # Save summary report
    summary = {
        "scrape_date": datetime.now().isoformat(),
        "total_pages": len(ALL_PAGE_DATA),
        "total_pdfs": len(ALL_PDF_DATA),
        "total_items": len(all_data),
        "output_file": str(combined_file)
    }
    summary_file = OUTPUT_DIR / f"summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("SCRAPING COMPLETE")
    print(f"  HTML pages scraped : {len(ALL_PAGE_DATA)}")
    print(f"  PDFs extracted     : {len(ALL_PDF_DATA)}")
    print(f"  Total items        : {len(all_data)}")
    print(f"  Saved to           : {combined_file}")
    print("=" * 60)
    return str(combined_file)

if __name__ == "__main__":
    asyncio.run(run_scraper())
