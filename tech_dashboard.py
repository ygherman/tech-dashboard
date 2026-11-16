#!/usr/bin/env python3
"""Daily technical-screen dashboard: S&P 500 + large TASE stocks.

Downloads one year of daily bars from Yahoo Finance, computes signals
(52-week breakouts, extreme RSI, gaps, daily movers, SMA50/200 crosses)
and writes a single self-contained HTML page.

Usage:  python tech_dashboard.py [--out site/index.html]
"""
import argparse
import datetime as dt
import html
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

SP500_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

# Largest TASE stocks (Yahoo symbol without ".TA" -> display name). Edit freely.
TASE = {
    "TEVA": "טבע", "LUMI": "לאומי", "POLI": "הפועלים", "ESLT": "אלביט מערכות",
    "MZTF": "מזרחי טפחות", "PHOE": "הפניקס", "AZRG": "עזריאלי", "DSCT": "דיסקונט",
    "HARL": "הראל", "MMHD": "מנורה מבטחים", "ENLT": "אנלייט", "OPCE": "OPC אנרגיה",
    "CLIS": "כלל ביטוח", "FIBI": "הבינלאומי", "NICE": "נייס", "BEZQ": "בזק",
    "MGDL": "מגדל", "ICL": "כיל", "NWMD": "ניו-מד", "MLSR": "מליסרון",
    "MGOR": "מגה אור", "NXSN": "נקסטוויז'ן", "BIG": "ביג", "DLEKG": "קבוצת דלק",
    "SPEN": "שפיר", "NVPT": "נאוויטס", "STRS": "שטראוס", "DORL": "דוראל",
    "ELAL": "אל על", "ENRG": "אנרג'יקס", "TASE": "הבורסה לני\"ע", "FTAL": "פתאל",
    "FIBIH": "פיבי אחזקות", "SKBN": "שיכון ובינוי", "SAE": "שופרסל", "MTAV": "מיטב",
    "MTRX": "מטריקס", "AMOT": "עמות", "PAZ": "פז", "MVNE": "מבנה",
}

RSI_HIGH, RSI_LOW = 70, 30
GAP_MIN = 3.0          # % open vs previous close
TOP_N = 10             # rows per table

# Three-candle favicon (green, red, green), URL-encoded for a data: URI
FAVICON = (
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' rx='7' fill='%23111827'/%3E"
    "%3Cg stroke-width='2' stroke-linecap='round'%3E"
    "%3Cpath d='M8 9v16' stroke='%2322c55e'/%3E%3Crect x='5' y='13' width='6' height='8' rx='1' fill='%2322c55e'/%3E"
    "%3Cpath d='M16 6v18' stroke='%23ef4444'/%3E%3Crect x='13' y='9' width='6' height='10' rx='1' fill='%23ef4444'/%3E"
    "%3Cpath d='M24 4v15' stroke='%2322c55e'/%3E%3Crect x='21' y='6' width='6' height='9' rx='1' fill='%2322c55e'/%3E"
    "%3C/g%3E%3C/svg%3E"
)


def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


def cross(close, sma):
    """'up'/'down' if today's close crossed the moving average, else None."""
    before, now = close.iloc[-2] - sma.iloc[-2], close.iloc[-1] - sma.iloc[-1]
    if before < 0 < now:
        return "up"
    if before > 0 > now:
        return "down"
    return None


def metrics(d):
    c, prev = d["Close"], d["Close"].iloc[-2]
    return {
        "date": d.index[-1].date(),
        "close": c.iloc[-1],
        "chg": (c.iloc[-1] / prev - 1) * 100,
        "gap": (d["Open"].iloc[-1] / prev - 1) * 100,
        "rsi": rsi(c).iloc[-1],
        "relvol": d["Volume"].iloc[-1] / d["Volume"].iloc[-21:-1].mean(),
        "hi52": d["High"].iloc[-253:-1].max(),
        "lo52": d["Low"].iloc[-253:-1].min(),
        "x50": cross(c, c.rolling(50).mean()),
        "x200": cross(c, c.rolling(200).mean()),
        "spark": c.iloc[-30:].tolist(),
    }


def signals(r):
    s = []
    if r.close > r.hi52: s.append(("שיא 52 ש׳", "up"))
    if r.close < r.lo52: s.append(("שפל 52 ש׳", "down"))
    if r.rsi >= RSI_HIGH: s.append((f"RSI {r.rsi:.0f}", "up"))
    if r.rsi <= RSI_LOW: s.append((f"RSI {r.rsi:.0f}", "down"))
    if r.gap >= GAP_MIN: s.append(("גאפ למעלה", "up"))
    if r.gap <= -GAP_MIN: s.append(("גאפ למטה", "down"))
    for col, label in (("x50", "ממוצע 50"), ("x200", "ממוצע 200")):
        if isinstance(r[col], str): s.append((f"חצה {label} {'↑' if r[col] == 'up' else '↓'}", r[col]))
    return s


def scan(names):
    """names: {yahoo_symbol: display_name} -> DataFrame of metrics + signals."""
    data = yf.download(list(names), period="13mo", interval="1d", group_by="ticker",
                       auto_adjust=True, threads=True, progress=False)
    rows = []
    for sym, name in names.items():
        try:
            d = data[sym].dropna()
        except KeyError:
            d = pd.DataFrame()
        if len(d) < 60:
            print(f"skip {sym}: {len(d)} bars")
            continue
        rows.append({"sym": sym, "name": name, **metrics(d)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["signals"] = df.apply(signals, axis=1)
    return df


# ---------- HTML ----------

def spark(values):
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    pts = " ".join(f"{i * 80 / (len(values) - 1):.1f},{22 - (v - lo) / span * 20:.1f}"
                   for i, v in enumerate(values))
    cls = "up" if values[-1] >= values[0] else "down"
    return f'<svg class="spark {cls}" viewBox="0 0 80 24"><polyline points="{pts}"/></svg>'


def pct(x):
    return f'<span class="{"up" if x >= 0 else "down"}">{x:+.1f}%</span>'


def table(title, hint, df, chart_url):
    if df.empty:
        return f'<section><h3>{title}</h3><p class="empty">אין מניות שעונות על התנאי היום.</p></section>'
    body = "".join(
        f'<tr><td><a href="{chart_url(r.sym)}" target="_blank" rel="noopener">'
        f'<b>{html.escape(r.sym.removesuffix(".TA"))}</b></a><small>{html.escape(r["name"])}</small></td>'
        f'<td class="num">{r.close:,.2f}</td><td class="num">{pct(r.chg)}</td>'
        f'<td class="num">{pct(r.gap)}</td><td class="num">{r.rsi:.0f}</td>'
        f'<td class="num">×{r.relvol:.1f}</td><td>{spark(r.spark)}</td>'
        f'<td>{"".join(f"<i class={c}>{t}</i>" for t, c in r.signals)}</td></tr>'
        for _, r in df.head(TOP_N).iterrows())
    return f"""<section><h3>{title}</h3><p class="hint">{hint}</p><div class="scroll"><table>
<thead><tr><th>מניה</th><th>שער</th><th>שינוי</th><th>גאפ</th><th>RSI</th><th>מחזור יחסי</th><th>30 יום</th><th>אותות</th></tr></thead>
<tbody>{body}</tbody></table></div></section>"""


def market(key, title, df, chart_url, unit):
    if df.empty:
        return f'<section id="{key}" class="market"><h2>{title}</h2><p class="empty">לא התקבלו נתונים.</p></section>'
    last = df["date"].max()
    df = df[df["date"] == last]  # drop symbols that did not trade on the last session
    n = df["signals"].str.len()
    t = lambda *a: table(*a, chart_url)
    parts = [
        t("שווה לשים לב", "מניות עם שני אותות ומעלה, לפי מספר האותות ומחזור יחסי",
          df[n >= 2].assign(n=n).sort_values(["n", "relvol"], ascending=False)),
        t("פריצה לשיא 52 שבועות", "סגירה מעל השיא של השנה האחרונה. מחזור יחסי גבוה מחזק את הפריצה",
          df[df.close > df.hi52].sort_values("relvol", ascending=False)),
        t("שבירה לשפל 52 שבועות", "סגירה מתחת לשפל של השנה האחרונה",
          df[df.close < df.lo52].sort_values("relvol", ascending=False)),
        t(f"RSI מעל {RSI_HIGH}", "קנייה-יתר. מומנטום חזק, אבל גם סיכון לתיקון",
          df[df.rsi >= RSI_HIGH].sort_values("rsi", ascending=False)),
        t(f"RSI מתחת ל-{RSI_LOW}", "מכירת-יתר. מועמדות לריבאונד",
          df[df.rsi <= RSI_LOW].sort_values("rsi")),
        t(f"גאפים של {GAP_MIN:.0f}% ומעלה", "פתיחה רחוקה מסגירת אתמול, למעלה או למטה",
          df[df.gap.abs() >= GAP_MIN].sort_values("gap", key=abs, ascending=False)),
        t("העולות של היום", "", df.sort_values("chg", ascending=False)),
        t("היורדות של היום", "", df.sort_values("chg")),
        t("חציות ממוצע 50 / 200", "סגירה שחצתה את הממוצע הנע היום",
          df[df.x50.notna() | df.x200.notna()].sort_values("relvol", ascending=False)),
    ]
    return (f'<section id="{key}" class="market"><h2>{title}</h2>'
            f'<p class="meta">מסחר אחרון: {last:%d.%m.%Y} · {len(df)} מניות נסרקו · {unit}</p>'
            + "".join(parts) + "</section>")


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1b2330;--mute:#647084;--line:#e3e7ee;--accent:#2f5bd3;--up:#11804a;--down:#c33a32;--chip-up:#e3f4ea;--chip-down:#fbe7e5}
@media (prefers-color-scheme:dark){:root{--bg:#10141b;--card:#171c25;--fg:#e6eaf1;--mute:#8d98aa;--line:#273040;--accent:#7d9cff;--up:#4cc489;--down:#f0776e;--chip-up:#173527;--chip-down:#3d1d1b;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 "Rubik",system-ui,sans-serif;padding:0 max(16px,env(safe-area-inset-right)) 48px max(16px,env(safe-area-inset-left))}
header{position:sticky;top:0;background:var(--bg);padding:14px 0 10px;border-bottom:1px solid var(--line);z-index:2;display:flex;flex-wrap:wrap;gap:8px 20px;align-items:baseline}
h1{font-size:20px;margin:0}
nav{display:flex;gap:8px}
nav a{padding:4px 12px;border:1px solid var(--line);border-radius:99px;color:var(--fg);text-decoration:none;background:var(--card)}
.gen{color:var(--mute);font-size:13px;margin-inline-start:auto}
main{max-width:1100px;margin:0 auto}
h2{font-size:22px;margin:32px 0 2px}
h3{font-size:16px;margin:0 0 2px}
.meta,.hint,.empty{color:var(--mute);margin:0 0 10px;font-size:13px}
section section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:14px 0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th{font-weight:500;color:var(--mute);font-size:12px;text-align:start;padding:4px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap;vertical-align:middle}
tr:last-child td{border-bottom:0}
th:first-child,td:first-child{position:sticky;right:0;background:var(--card);z-index:1}
td small{display:block;color:var(--mute);font-size:12px;max-width:180px;overflow:hidden;text-overflow:ellipsis}
td a{color:var(--accent);text-decoration:none}
.num{direction:ltr;text-align:end}
.up{color:var(--up)}.down{color:var(--down)}
i{font-style:normal;font-size:12px;padding:1px 8px;border-radius:99px;margin-inline-end:4px}
i.up{background:var(--chip-up)}i.down{background:var(--chip-down)}
.spark{width:80px;height:24px;display:block}
.spark polyline{fill:none;stroke:currentColor;stroke-width:1.5}
footer{color:var(--mute);font-size:12px;max-width:1100px;margin:32px auto 0}
"""


def page(sections, generated):
    return f"""<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>סורק טכני יומי</title>
<link rel="icon" href="data:image/svg+xml,{FAVICON}">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Rubik:wght@400;500;700&display=swap">
<style>{CSS}</style></head><body>
<header><h1>סורק טכני יומי</h1><nav><a href="#us">ארה״ב</a><a href="#il">ת״א</a></nav>
<span class="gen">עודכן {generated:%d.%m.%Y} בשעה {generated:%H:%M}</span></header>
<main>{sections}</main>
<footer>נתונים: Yahoo Finance דרך yfinance, מחירים מתואמים לדיבידנדים ופיצולים. RSI לפי 14 ימים (Wilder).
מחזור יחסי = מחזור היום חלקי ממוצע 20 הימים הקודמים. הדף הוא כלי סינון ולא המלצת השקעה.</footer>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/index.html")
    out = Path(ap.parse_args().out)

    sp = pd.read_csv(SP500_CSV)
    us = dict(zip(sp.Symbol.str.replace(".", "-", regex=False), sp.Security))
    il = {f"{k}.TA": v for k, v in TASE.items()}

    sections = (
        market("us", "ארה״ב · S&P 500", scan(us),
               lambda s: f"https://www.tradingview.com/chart/?symbol={s.replace('-', '.')}", "מחירים בדולר")
        + market("il", "תל אביב · מניות גדולות", scan(il),
                 lambda s: f"https://www.tradingview.com/chart/?symbol=TASE:{s.removesuffix('.TA')}", "מחירים באגורות")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(sections, dt.datetime.now(ZoneInfo("Asia/Jerusalem"))), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
