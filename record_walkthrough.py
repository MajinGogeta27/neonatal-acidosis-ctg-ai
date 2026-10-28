import os
import time
from pathlib import Path
from playwright.sync_api import sync_playwright
import imageio.v3 as iio

OUTPUT_DIR = Path("recordings")
OUTPUT_DIR.mkdir(exist_ok=True)

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(CHROME_PATH):
    CHROME_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

print(f"Using browser executable: {CHROME_PATH}")

frames = []

def capture(page, label, pause=1.0, steps=2):
    for _ in range(steps):
        time.sleep(pause / steps)
        img_bytes = page.screenshot(type="jpeg", quality=90)
        frames.append(iio.imread(img_bytes))
    print(f"  [Captured] {label} (frames so far: {len(frames)})")

def smooth_scroll(page, start_y, end_y, steps=10, pause=0.1):
    for s in range(1, steps + 1):
        curr_y = start_y + (end_y - start_y) * (s / steps)
        page.evaluate(f"window.scrollTo(0, {curr_y})")
        time.sleep(pause)
        img_bytes = page.screenshot(type="jpeg", quality=90)
        frames.append(iio.imread(img_bytes))

with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path=CHROME_PATH,
        headless=True,
        args=["--start-maximized", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"]
    )
    context = browser.new_context(
        viewport={"width": 1366, "height": 850},
        record_video_dir=str(OUTPUT_DIR),
        record_video_size={"width": 1366, "height": 850}
    )
    page = context.new_page()

    print("Navigating to Streamlit app at http://localhost:8501 ...")
    page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
    page.wait_for_selector(".main-header", timeout=15000)
    time.sleep(2.0)

    # ==========================================
    # 1. EXECUTIVE SUMMARY
    # ==========================================
    print("\n--- Section 1: Executive Summary ---")
    capture(page, "Executive Summary Header & KPIs", pause=2.0, steps=4)
    smooth_scroll(page, 0, 480, steps=10, pause=0.15)
    capture(page, "Outcome Distribution & Acidemia Breakdown", pause=3.0, steps=6)
    smooth_scroll(page, 480, 0, steps=8, pause=0.1)
    time.sleep(0.5)

    # ==========================================
    # 2. INTERACTIVE CTG INSPECTOR
    # ==========================================
    print("\n--- Section 2: Interactive CTG Inspector ---")
    page.locator("text=Interactive CTG Inspector").click()
    time.sleep(2.0)
    capture(page, "CTG Inspector - Record 1001 Overview", pause=2.0, steps=4)
    smooth_scroll(page, 0, 520, steps=10, pause=0.15)
    capture(page, "CTG Inspector - 30min Waveform Tracing (FHR & UC)", pause=3.0, steps=6)
    smooth_scroll(page, 520, 0, steps=8, pause=0.1)

    # Select severe acidemia record 1002 (pH 7.00)
    print("  Selecting Record 1002 (Severe Acidemia, pH=7.00)")
    inp = page.get_by_label("Select record ID:")
    inp.click()
    time.sleep(0.5)
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")
    time.sleep(2.0)
    capture(page, "CTG Inspector - Record 1002 High Risk Metrics", pause=2.0, steps=4)
    smooth_scroll(page, 0, 520, steps=10, pause=0.15)
    capture(page, "CTG Inspector - Record 1002 Waveform with Severe Decelerations", pause=3.0, steps=6)
    smooth_scroll(page, 520, 0, steps=8, pause=0.1)

    # ==========================================
    # 3. CONTINUOUS pH REGRESSION BENCHMARK
    # ==========================================
    print("\n--- Section 3: Continuous pH Regression Benchmark ---")
    page.locator("text=Continuous pH Regression Benchmark").click()
    time.sleep(2.0)
    capture(page, "Continuous Regression Benchmark Summary Table", pause=2.5, steps=5)
    smooth_scroll(page, 0, 650, steps=10, pause=0.15)
    capture(page, "Predicted vs Actual pH Scatter Plots", pause=3.0, steps=6)
    smooth_scroll(page, 650, 0, steps=8, pause=0.1)

    # ==========================================
    # 4. THRESHOLD CLASSIFICATION BENCHMARK
    # ==========================================
    print("\n--- Section 4: Threshold Classification Benchmark ---")
    page.locator("text=Threshold Classification Benchmark").click()
    time.sleep(2.0)
    capture(page, "Significant Acidemia (pH < 7.10) Table", pause=2.5, steps=5)
    smooth_scroll(page, 0, 680, steps=10, pause=0.15)
    capture(page, "Significant Acidemia (pH < 7.10) ROC Curves", pause=3.0, steps=6)
    smooth_scroll(page, 680, 0, steps=8, pause=0.1)

    # Switch to Severe Acidemia Tab (pH < 7.05)
    print("  Switching to Severe (pH < 7.05) Tab")
    try:
        tabs = page.locator("[role='tab']")
        if tabs.count() >= 2:
            tabs.nth(1).click()
        else:
            page.locator("text=Severe").click()
    except Exception as e:
        print(f"  Warning on tab click: {e}")
    time.sleep(1.5)
    capture(page, "Severe Acidemia (pH < 7.05) Table", pause=2.5, steps=5)
    smooth_scroll(page, 0, 680, steps=10, pause=0.15)
    capture(page, "Severe Acidemia (pH < 7.05) ROC Curves", pause=3.0, steps=6)
    smooth_scroll(page, 680, 0, steps=8, pause=0.1)

    # ==========================================
    # 5. EXPLAINABILITY (SHAP AND SALIENCY)
    # ==========================================
    print("\n--- Section 5: Explainability (SHAP & Saliency) ---")
    page.locator("text=Explainability (SHAP and Saliency)").click()
    time.sleep(2.0)
    capture(page, "Explainability Overview", pause=2.0, steps=4)
    smooth_scroll(page, 0, 520, steps=10, pause=0.15)
    capture(page, "SHAP Feature Importance & Deep Temporal Saliency", pause=4.0, steps=8)
    smooth_scroll(page, 520, 0, steps=8, pause=0.1)

    print("\nWalkthrough completed successfully. Finalizing video recording...")
    page.close()
    context.close()
    browser.close()

# Save compiled MP4 video from captured frames
mp4_path = OUTPUT_DIR / "ctg_acidosis_ai_walkthrough.mp4"
print(f"\nWriting {len(frames)} frames to MP4 video: {mp4_path} ...")
iio.imwrite(str(mp4_path), frames, fps=6, codec="libx264")
print(f"  --> MP4 video saved: {mp4_path} ({os.path.getsize(mp4_path)/(1024*1024):.2f} MB)")

# Rename native Playwright webm
webm_files = list(OUTPUT_DIR.glob("*.webm"))
if webm_files:
    latest_webm = max(webm_files, key=os.path.getctime)
    renamed_webm = OUTPUT_DIR / "ctg_acidosis_ai_walkthrough.webm"
    if latest_webm != renamed_webm:
        if renamed_webm.exists():
            renamed_webm.unlink()
        latest_webm.rename(renamed_webm)
    print(f"  --> WebM video saved: {renamed_webm} ({os.path.getsize(renamed_webm)/(1024*1024):.2f} MB)")

print("\nALL RECORDINGS CREATED SUCCESSFULLY!")
