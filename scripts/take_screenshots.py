#!/usr/bin/env python3
"""Take screenshots of the GGUF Knowledge Extractor web UI."""
import asyncio
from playwright.async_api import async_playwright
from pathlib import Path

OUT_DIR = Path("/home/z/my-project/download/screenshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "http://127.0.0.1:8101"


async def take_screenshots():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=2,
        )
        page = await context.new_page()

        # Dashboard
        print("Screenshot: Dashboard...")
        await page.goto(f"{BASE_URL}/", wait_until="networkidle")
        await page.wait_for_timeout(2000)
        await page.screenshot(path=str(OUT_DIR / "dashboard.png"), full_page=False)
        print(f"  Saved: {OUT_DIR / 'dashboard.png'}")

        # Extract view
        print("Screenshot: Extract view...")
        await page.click('[data-view="extract"]')
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(OUT_DIR / "extract.png"), full_page=False)
        print(f"  Saved: {OUT_DIR / 'extract.png'}")

        # Models (HF Hub browser)
        print("Screenshot: Models / HF Hub...")
        await page.click('[data-view="models"]')
        await page.wait_for_timeout(1500)
        # Type a search query
        await page.fill('#models-search-query', 'llama 3.2')
        await page.click('#btn-models-search')
        await page.wait_for_timeout(2000)
        await page.screenshot(path=str(OUT_DIR / "models_hub.png"), full_page=False)
        print(f"  Saved: {OUT_DIR / 'models_hub.png'}")

        # Surgery view
        print("Screenshot: Surgery view...")
        await page.click('[data-view="surgery"]')
        await page.wait_for_timeout(1000)
        await page.click('#btn-surgery-add-op')
        await page.wait_for_timeout(500)
        await page.screenshot(path=str(OUT_DIR / "surgery.png"), full_page=False)
        print(f"  Saved: {OUT_DIR / 'surgery.png'}")

        # Transplant view
        print("Screenshot: Transplant view...")
        await page.click('[data-view="transplant"]')
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(OUT_DIR / "transplant.png"), full_page=False)
        print(f"  Saved: {OUT_DIR / 'transplant.png'}")

        # Quantize view
        print("Screenshot: Quantize view...")
        await page.click('[data-view="quantize"]')
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(OUT_DIR / "quantize.png"), full_page=False)
        print(f"  Saved: {OUT_DIR / 'quantize.png'}")

        await browser.close()
        print("\nAll screenshots saved to:", OUT_DIR)


if __name__ == "__main__":
    asyncio.run(take_screenshots())
