"""One-off dev utility: capture real screenshots of the running Streamlit
app (app.py) for the README, using Playwright's isolated Chromium (does
NOT touch the user's real browser profile).

Prereqs: `streamlit run app.py` already running on localhost:8501, and
`pip install playwright && playwright install chromium`.

Usage:
    python scripts/capture_webapp_screenshots.py
"""
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
APP_URL = "http://localhost:8501"
CROP_BOX = (0, 0, 900, 710)  # trims the empty space below the app content


def main():
    ASSETS_DIR.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 900, "height": 1000})
        page.goto(APP_URL, wait_until="networkidle")
        page.wait_for_selector("text=Spam Email Classifier")
        page.add_style_tag(content="header[data-testid='stHeader'] { visibility: hidden; }")
        time.sleep(1)

        dont_show = page.get_by_text("Don't show again")
        if dont_show.count() > 0:
            dont_show.first.click()
            time.sleep(0.3)

        page.screenshot(path=str(ASSETS_DIR / "webapp_landing.png"))
        print("Saved webapp_landing.png")

        page.get_by_role("button", name="Spam example").click()
        time.sleep(0.3)
        page.get_by_role("button", name="Classify", exact=True).click()
        time.sleep(0.5)
        page.screenshot(path=str(ASSETS_DIR / "webapp_spam_result.png"))
        print("Saved webapp_spam_result.png")

        page.reload(wait_until="networkidle")
        page.wait_for_selector("text=Spam Email Classifier")
        page.add_style_tag(content="header[data-testid='stHeader'] { visibility: hidden; }")
        time.sleep(0.5)

        page.get_by_role("button", name="Ham example").click()
        time.sleep(0.3)
        page.get_by_role("button", name="Classify", exact=True).click()
        time.sleep(0.5)
        page.screenshot(path=str(ASSETS_DIR / "webapp_ham_result.png"))
        print("Saved webapp_ham_result.png")

        browser.close()

    for name in ["webapp_landing.png", "webapp_spam_result.png", "webapp_ham_result.png"]:
        path = ASSETS_DIR / name
        Image.open(path).convert("RGB").crop(CROP_BOX).save(path)

    _build_demo_gif()


def _build_demo_gif():
    frame_names = ["webapp_landing.png", "webapp_spam_result.png", "webapp_ham_result.png"]
    frames = [Image.open(ASSETS_DIR / name).convert("RGB").crop(CROP_BOX) for name in frame_names]
    durations = [2200, 2600, 2600]
    out = ASSETS_DIR / "webapp_demo.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"Saved {out.name}")


if __name__ == "__main__":
    main()
