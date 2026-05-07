"""
百度百科 + Wikipedia 育儿条目补充爬虫 (Playwright 真人模拟版)

策略：优先 Wikipedia 中文（反爬弱，内容权威），备选百度百科移动版。
所有行为模拟真实用户：随机延迟、鼠标移动、滚动节奏变化。

用法:
  python crawlers/crawl_baike.py
  python crawlers/crawl_baike.py --source wiki     # 仅 Wikipedia
  python crawlers/crawl_baike.py --source baike    # 仅百度百科
  python crawlers/crawl_baike.py --max 30
"""

import asyncio
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.common_schema import (
    KnowledgeItem, guess_age_range, guess_category,
    is_valid_knowledge, chinese_char_ratio
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "baike_articles.jsonl")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "baike_progress.json")

# === Wikipedia 中文育儿关键词 ===
SEED_KEYWORDS_WIKI = [
    "育儿", "婴儿护理", "婴幼儿喂养", "早期教育", "母乳喂养",
    "辅食", "新生儿", "儿童发育", "儿童心理", "家庭教育",
    "婴幼儿腹泻", "小儿发热", "婴儿湿疹", "儿童哮喘", "小儿肺炎",
    "手足口病", "水痘", "麻疹", "新生儿黄疸", "儿童过敏",
    "孕期保健", "产前检查", "产后护理", "坐月子", "胎教",
    "儿童营养", "配方奶粉", "辅食添加",
    "儿童生长发育", "儿童身高", "儿童体重", "儿童语言发育",
    "儿童疫苗接种", "百白破疫苗", "脊髓灰质炎疫苗",
    "儿童安全", "婴儿猝死综合征", "儿童意外伤害",
    "注意力缺陷多动障碍", "自闭症谱系障碍", "儿童焦虑",
    "如厕训练", "感觉统合", "儿童睡眠障碍",
    "妊娠期糖尿病", "产后抑郁", "母乳",
    "维生素D", "儿童补钙", "儿童补锌", "儿童肥胖",
    "肺炎球菌疫苗", "流感疫苗", "乙肝疫苗",
    "儿科", "小儿外科", "儿童保健", "儿童医院",
    "自然分娩", "剖宫产", "无痛分娩",
    "蒙台梭利教育法", "正面管教",
]

# === 百度百科备选关键词（Wikipedia 没有的条目） ===
SEED_KEYWORDS_BAIKE = [
    "丁香妈妈", "育学园", "春雨医生",
    "中国儿童发展纲要", "新生儿疾病筛查",
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
]


async def human_delay(min_s: float = 0.5, max_s: float = 3.0):
    await asyncio.sleep(random.uniform(min_s, max_s))


async def simulate_scroll(page, scrolls: int = None):
    if scrolls is None:
        scrolls = random.randint(3, 6)
    for _ in range(scrolls):
        delta = random.randint(200, 500)
        await page.evaluate(f"window.scrollBy(0, {delta})")
        await asyncio.sleep(random.uniform(0.2, 0.8))
    if random.random() < 0.3:
        delta = random.randint(-200, -50)
        await page.evaluate(f"window.scrollBy(0, {delta})")
        await asyncio.sleep(random.uniform(0.2, 0.5))


async def simulate_mouse(page):
    for _ in range(random.randint(1, 3)):
        x = random.randint(100, 800)
        y = random.randint(100, 600)
        await page.mouse.move(x, y, steps=random.randint(3, 10))
        await asyncio.sleep(random.uniform(0.05, 0.2))


async def goto_with_retry(page, url: str, max_retries: int = 2) -> bool:
    for attempt in range(max_retries):
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=25000)
            await human_delay(1.0, 2.5)
            title = await page.title()
            if any(bad in title for bad in ["验证", "403", "404", "captcha"]):
                print(f"    [BLOCKED/404] {title[:60]}")
                if attempt < max_retries - 1:
                    await human_delay(5.0, 10.0)
                    continue
                return False
            return True
        except Exception as e:
            print(f"    [RETRY] {type(e).__name__}")
            if attempt < max_retries - 1:
                await human_delay(3.0, 6.0)
    return False


# ================== Wikipedia 中文 ==================

async def crawl_wikipedia_entry(context, keyword: str) -> KnowledgeItem | None:
    """爬取 Wikipedia 中文条目"""
    url = f"https://zh.wikipedia.org/wiki/{keyword}"

    page = await context.new_page()
    try:
        if not await goto_with_retry(page, url):
            return None

        # 等待正文加载
        try:
            await page.wait_for_selector("#bodyContent, #mw-content-text, .mw-parser-output",
                                          timeout=8000)
        except Exception:
            pass

        await simulate_mouse(page)
        await human_delay(0.3, 1.0)
        await simulate_scroll(page, random.randint(3, 6))
        await human_delay(0.5, 1.0)

        # 提取标题和正文
        extracted = await page.evaluate("""
            () => {
                // 标题
                let title = '';
                const h1 = document.querySelector('#firstHeading, h1');
                if (h1) title = h1.textContent.trim();

                // 正文
                const contentEl = document.querySelector('#mw-content-text, .mw-parser-output, #bodyContent');
                if (!contentEl) return null;

                // 移除引用、导航、信息框
                const clone = contentEl.cloneNode(true);
                const removes = clone.querySelectorAll(
                    '.reflist, .references, .navbox, .infobox, .mw-editsection, ' +
                    '.thumb, .toc, .sidebar, .noprint, .metadata, script, style, ' +
                    '.hatnote, .shortdescription, sup.reference, .mw-cite-backlink'
                );
                removes.forEach(el => el.remove());

                const text = clone.textContent.replace(/\\s+/g, ' ').trim();
                return { title, content: text, url: window.location.href };
            }
        """)

        if not extracted or not extracted.get("content"):
            return None

        content = extracted["content"]
        if len(content) < 100 or chinese_char_ratio(content) < 0.3:
            return None

        # 只保留前 3000 字
        content = content[:3000]

        item = KnowledgeItem(
            source="wikipedia_zh",
            type="article",
            title=extracted["title"],
            content=content,
            age_range=guess_age_range(extracted["title"] + content),
            category=guess_category(extracted["title"] + content),
            url=extracted.get("url", url),
            quality_score=0.75,
        )

        if not is_valid_knowledge(item):
            return None

        return item

    except Exception as e:
        print(f"    [ERROR] {type(e).__name__}: {e}")
        return None
    finally:
        await page.close()


# ================== 百度百科（WAP 版） ==================

async def crawl_baike_wap_entry(context, keyword: str) -> KnowledgeItem | None:
    """爬取百度百科 WAP 版条目"""
    import urllib.parse
    kw_encoded = urllib.parse.quote(keyword)
    # WAP 版 URL
    url = f"https://baike.baidu.com/item/{kw_encoded}?view=home"

    page = await context.new_page()
    try:
        # 用移动端 User-Agent 和 viewport
        await page.set_viewport_size({"width": 414, "height": 896})
        if not await goto_with_retry(page, url):
            return None

        # 等待正文
        try:
            await page.wait_for_selector(".J-lemma-content, .lemma-text, .para, .main-info",
                                          timeout=8000)
        except Exception:
            pass

        await simulate_scroll(page, random.randint(3, 5))
        await human_delay(0.5, 1.0)

        extracted = await page.evaluate("""
            () => {
                let title = '';
                const h1 = document.querySelector('h1, .lemma-title');
                if (h1) title = h1.textContent.trim();

                const paras = [];
                const els = document.querySelectorAll('.J-lemma-content .para, .lemma-text .para, .para');
                for (const el of els) {
                    const t = el.textContent.trim();
                    if (t && t.length > 15) paras.push(t);
                }
                if (paras.length === 0) {
                    // fallback: any paragraph-like elements
                    const allPs = document.querySelectorAll('p, div.para, [class*="para"]');
                    for (const el of allPs) {
                        const t = el.textContent.trim();
                        if (t && t.length > 15) paras.push(t);
                    }
                }

                const content = paras.join('\\n\\n');
                return { title, content, url: window.location.href };
            }
        """)

        if not extracted or not extracted.get("content"):
            return None

        content = extracted["content"]
        content = re.sub(r'\[\d+(?:[-,]\d+)*\]', '', content)
        content = re.sub(r'\s+', ' ', content).strip()

        if len(content) < 100 or chinese_char_ratio(content) < 0.3:
            return None

        content = content[:3000]

        item = KnowledgeItem(
            source="baike",
            type="article",
            title=extracted["title"],
            content=content,
            age_range=guess_age_range(extracted["title"] + content),
            category=guess_category(extracted["title"] + content),
            url=extracted.get("url", url),
            quality_score=0.8,
        )

        if not is_valid_knowledge(item):
            return None

        return item

    except Exception as e:
        print(f"    [ERROR] {type(e).__name__}: {e}")
        return None
    finally:
        await page.close()


async def main_async(source: str = "all", max_entries: int = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载进度
    done_keywords = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            done_keywords = set(json.load(f))
        print(f"[INFO] Previously crawled: {len(done_keywords)} keywords")

    # 加载已有条目
    existing_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_count = sum(1 for _ in f)
    print(f"[INFO] Existing articles: {existing_count}")

    # 构建爬取任务列表
    tasks = []
    if source in ("all", "wiki"):
        for kw in SEED_KEYWORDS_WIKI:
            if f"wiki:{kw}" not in done_keywords:
                tasks.append(("wiki", kw))
    if source in ("all", "baike"):
        for kw in SEED_KEYWORDS_BAIKE:
            if f"baike:{kw}" not in done_keywords:
                tasks.append(("baike", kw))

    if max_entries:
        tasks = tasks[:max_entries]

    if not tasks:
        print("[INFO] All keywords already crawled.")
        return

    print(f"[INFO] Tasks to crawl: {len(tasks)} (wiki: {sum(1 for t in tasks if t[0]=='wiki')}, "
          f"baike: {sum(1 for t in tasks if t[0]=='baike')})")

    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-blink-features=AutomationControlled",
            ]
        )

        context = await browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport={"width": random.randint(1280, 1440), "height": random.randint(800, 960)},
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
        )

        # Anti-detection
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en'] });
        """)

        new_items = 0
        for i, (src, kw) in enumerate(tasks):
            print(f"\n[{i + 1}/{len(tasks)}] [{src}] {kw}")

            if src == "wiki":
                item = await crawl_wikipedia_entry(context, kw)
            else:
                item = await crawl_baike_wap_entry(context, kw)

            task_key = f"{src}:{kw}"
            if item:
                new_items += 1
                done_keywords.add(task_key)

                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")
                with open(PROGRESS_FILE, "w", encoding="utf-8") as fp:
                    json.dump(list(done_keywords), fp, ensure_ascii=False)

                print(f"    [OK] {len(item.content)} chars, age={item.age_range}, cat={item.category}")
            else:
                done_keywords.add(task_key)
                with open(PROGRESS_FILE, "w", encoding="utf-8") as fp:
                    json.dump(list(done_keywords), fp, ensure_ascii=False)

            await human_delay(2.0, 6.0)

            if (i + 1) % 15 == 0:
                rest = random.uniform(5.0, 10.0)
                print(f"\n  === Rest {rest:.0f}s ===")
                await asyncio.sleep(rest)

        await browser.close()

    print(f"\n[DONE] New articles: {new_items}")
    print(f"[DONE] Total in output: {existing_count + new_items}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="百科育儿条目爬虫")
    parser.add_argument("--source", default="all", choices=["all", "wiki", "baike"])
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()

    asyncio.run(main_async(source=args.source, max_entries=args.max))


if __name__ == "__main__":
    main()
