"""
一键登录状态保存工具

同时打开 2 个浏览器窗口，手动登录后按 Enter 保存所有登录状态。
保存的 cookies/localStorage 供后续并发爬虫使用。

用法:
  python crawlers/save_auth.py
  python crawlers/save_auth.py --site zhihu,dxy  # 仅指定站点
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

AUTH_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "auth")

SITES = [
    {
        "key": "dxy",
        "name": "丁香妈妈/丁香园 (专业医学，权重 1.0)",
        "url": "https://dxy.com",
        "login_hint": "请使用微信扫码或手机号登录丁香园",
        "crawl_targets": [
            ("育儿知识栏目", "https://dxy.com/column/304"),
            ("母婴健康栏目", "https://dxy.com/column/303"),
        ],
    },
    {
        "key": "zhihu",
        "name": "知乎 (高质量问答，权重 0.7)",
        "url": "https://www.zhihu.com",
        "login_hint": "请使用手机号/微信扫码登录知乎",
        "crawl_targets": [
            ("育儿话题", "https://www.zhihu.com/topic/19550931/hot"),
            ("母婴话题", "https://www.zhihu.com/topic/19634995/hot"),
        ],
    },
]


async def save_auth_for_site(browser, site: dict) -> dict:
    """为单个站点打开浏览器，等待用户手动登录，返回认证状态"""
    context = await browser.new_context(
        viewport={"width": 1280, "height": 900},
        locale="zh-CN",
        timezone_id="Asia/Shanghai",
    )

    # 注入 anti-detection
    await context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'chrome', { get: () => ({ runtime: {} }) });
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en'] });
    """)

    page = await context.new_page()
    print(f"\n  [{site['name']}] 正在打开 {site['url']}")
    print(f"  [{site['name']}] {site['login_hint']}")
    print(f"  [{site['name']}] 请在浏览器窗口中完成登录...")

    try:
        await page.goto(site["url"], wait_until="domcontentloaded", timeout=30000)
    except Exception as e:
        print(f"  [{site['name']}] 页面加载警告: {e}")

    return {
        "site": site,
        "context": context,
        "page": page,
    }


async def main_async(site_filter: list[str] = None):
    os.makedirs(AUTH_DIR, exist_ok=True)

    # 筛选站点
    targets = SITES
    if site_filter:
        targets = [s for s in SITES if s["key"] in site_filter]
        if not targets:
            print(f"[ERROR] 未找到匹配站点: {site_filter}")
            print(f"[INFO] 可用站点: {[s['key'] for s in SITES]}")
            return

    print("=" * 60)
    print("  育儿数据源登录状态保存工具")
    print("=" * 60)
    print(f"\n将同时打开 {len(targets)} 个浏览器窗口：")
    for s in targets:
        print(f"  - {s['name']} ({s['key']}): {s['url']}")
    print(f"\n请在每个窗口中手动登录。")
    print(f"全部登录完成后，回到此处按 Enter 保存所有登录状态。\n")

    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
        )

        # 并发打开所有站点
        tasks = [save_auth_for_site(browser, s) for s in targets]
        results = await asyncio.gather(*tasks)

        # 等待用户确认
        input("\n全部登录完成后，按 Enter 保存状态...")

        # 保存每个站点的认证状态
        saved_count = 0
        for r in results:
            site = r["site"]
            context = r["context"]
            page = r["page"]

            try:
                # 获取 cookies
                cookies = await context.cookies()

                # 获取 localStorage
                local_storage = await page.evaluate("""
                    () => {
                        const items = {};
                        for (let i = 0; i < localStorage.length; i++) {
                            const key = localStorage.key(i);
                            items[key] = localStorage.getItem(key);
                        }
                        return items;
                    }
                """)

                auth_data = {
                    "site": site["key"],
                    "name": site["name"],
                    "url": site["url"],
                    "cookies": cookies,
                    "local_storage": local_storage,
                    "crawl_targets": site.get("crawl_targets", []),
                }

                filepath = os.path.join(AUTH_DIR, f"{site['key']}_auth.json")
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(auth_data, f, ensure_ascii=False, indent=2)

                print(f"  [{site['name']}] 已保存: {len(cookies)} cookies, "
                      f"{len(local_storage)} localStorage keys → {filepath}")
                saved_count += 1

            except Exception as e:
                print(f"  [{site['name']}] 保存失败: {e}")
            finally:
                await context.close()

        await browser.close()

    print(f"\n[DONE] 成功保存 {saved_count}/{len(targets)} 个站点的登录状态")
    print(f"[DONE] 状态文件目录: {AUTH_DIR}")
    print(f"[NEXT] 运行: python crawlers/crawl_authenticated.py")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="保存网站登录状态")
    parser.add_argument("--site", type=str, default=None,
                        help="指定站点，逗号分隔 (如: zhihu,dxy)")
    args = parser.parse_args()

    site_filter = None
    if args.site:
        site_filter = [s.strip() for s in args.site.split(",")]

    asyncio.run(main_async(site_filter))


if __name__ == "__main__":
    main()
