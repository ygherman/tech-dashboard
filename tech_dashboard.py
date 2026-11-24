#!/usr/bin/env python3
"""Daily technical-screen dashboard: S&P 500 + large TASE stocks.

Downloads one year of daily bars from Yahoo Finance, computes signals
(52-week breakouts, extreme RSI, gaps, daily movers, SMA50/200 crosses),
an ATR-based stop for each stock, and writes a single self-contained HTML page.
Position size is computed in the browser from a portfolio value the viewer
types in, so it never appears in the repo or on the published page.

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
STOP_ATR = 2           # stop = close - STOP_ATR x ATR(14)

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


def atr(d, n=14):
    """Average True Range (Wilder): smoothed max(high-low, |high-prev close|, |low-prev close|)."""
    prev = d["Close"].shift()
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - prev).abs(), (d["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


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
        "atr": atr(d).iloc[-1],
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


# Column headers link to their glossary entry
COLUMNS = [("שער", "price"), ("שינוי", "chg"), ("גאפ", "gap"), ("RSI", "rsi"),
           ("מחזור יחסי", "relvol"), ("ATR", "atr"), ("סטופ", "stop"),
           ("כמות", "size"), ("סכום ₪", "size"), ("סיכון ₪", "size"), ("30 יום", "spark"), ("אותות", "signals")]
THEAD = "<th>מניה</th>" + "".join(f'<th><a href="#g-{k}">{t}</a></th>' for t, k in COLUMNS)


def row(r, chart_url, fx):
    stop = r.close - STOP_ATR * r.atr
    # data-c / data-s: close and stop in ILS per share, read by the sizing script
    sizing = f' data-c="{r.close * fx:.4f}" data-s="{stop * fx:.4f}"' if fx and stop > 0 else ""
    return (
        f'<tr{sizing}><td><a href="{chart_url(r.sym)}" target="_blank" rel="noopener">'
        f'<b>{html.escape(r.sym.removesuffix(".TA"))}</b></a><small>{html.escape(r["name"])}</small></td>'
        f'<td class="num">{r.close:,.2f}</td><td class="num">{pct(r.chg)}</td>'
        f'<td class="num">{pct(r.gap)}</td><td class="num">{r.rsi:.0f}</td>'
        f'<td class="num">×{r.relvol:.1f}</td><td class="num">{r.atr / r.close * 100:.1f}%</td>'
        f'<td class="num">{stop:,.2f}</td>' + '<td class="num sz">—</td>' * 3 +
        f'<td>{spark(r.spark)}</td>'
        f'<td>{"".join(f"<i class={c}>{t}</i>" for t, c in r.signals)}</td></tr>')


def table(title, hint, df, chart_url, fx):
    if df.empty:
        return f'<section><h3>{title}</h3><p class="empty">אין מניות שעונות על התנאי היום.</p></section>'
    body = "".join(row(r, chart_url, fx) for _, r in df.head(TOP_N).iterrows())
    return f"""<section><h3>{title}</h3><p class="hint">{hint}</p><div class="scroll"><table>
<thead><tr>{THEAD}</tr></thead><tbody>{body}</tbody></table></div></section>"""


def market(key, title, df, chart_url, unit, fx):
    """fx: ILS per one unit of the quote currency (None = no sizing)."""
    if df.empty:
        return f'<section id="{key}" class="market"><h2>{title}</h2><p class="empty">לא התקבלו נתונים.</p></section>'
    last = df["date"].max()
    df = df[df["date"] == last]  # drop symbols that did not trade on the last session
    df = df.assign(n=df["signals"].str.len())  # signal count, as a column so filters keep it aligned
    t = lambda *a: table(*a, chart_url, fx)
    parts = [
        t("שווה לשים לב", "מניות עם שני אותות ומעלה, לפי מספר האותות ומחזור יחסי",
          df[df.n >= 2].sort_values(["n", "relvol"], ascending=False)),
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
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:16px 0 0;display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center}
.panel label{display:flex;gap:6px;align-items:center;font-size:14px}
.panel input{width:7.5em;font:inherit;padding:3px 6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);direction:ltr}
.panel input[name=risk],.panel input[name=cap]{width:4.5em}
.panel small{color:var(--mute);flex-basis:100%}
th a{color:inherit;text-decoration:underline dotted}
.panel a{color:var(--accent)}
[id]{scroll-margin-top:70px}
#glossary dl{margin:0}
#glossary dt{font-weight:700;margin-top:14px}
#glossary dd{margin:2px 0 0}
#glossary p{margin:4px 0}
footer{color:var(--mute);font-size:12px;max-width:1100px;margin:32px auto 0}
"""


# Sizing inputs live only in the viewer's browser (localStorage), never in the repo or page.
PANEL = """<form class="panel" id="sizing" onsubmit="return false">
<label>שווי תיק ₪ <input name="pf" type="number" inputmode="decimal" min="0" step="1000"></label>
<label>סיכון לעסקה % <input name="risk" type="number" inputmode="decimal" min="0" step="0.1" value="1"></label>
<label>תקרה לפוזיציה % <input name="cap" type="number" inputmode="decimal" min="0" step="1" value="10"></label>
<small>הערכים נשמרים רק בדפדפן הזה ולא נשלחים לשום מקום. בטלפון ובמחשב צריך להקליד אותם פעם אחת בכל מכשיר.
<a href="#g-size">איך מחושב גודל העסקה</a></small></form>"""

SIZING_JS = """(() => {
  const f = document.getElementById('sizing'), KEY = 'sizing', names = ['pf', 'risk', 'cap'];
  try { const v = JSON.parse(localStorage.getItem(KEY)) || {}; names.forEach(k => { if (v[k]) f[k].value = v[k]; }); } catch (e) {}
  const fmt = n => Math.round(n).toLocaleString('he-IL');
  function calc() {
    try { localStorage.setItem(KEY, JSON.stringify(Object.fromEntries(names.map(k => [k, f[k].value])))); } catch (e) {}
    const pf = +f.pf.value, risk = pf * f.risk.value / 100, cap = pf * f.cap.value / 100;
    document.querySelectorAll('tr[data-c]').forEach(tr => {
      const c = +tr.dataset.c, d = c - tr.dataset.s, [qty, amt, rsk] = tr.querySelectorAll('.sz');
      if (!(pf > 0)) { qty.textContent = amt.textContent = rsk.textContent = '—'; return; }
      const n = Math.floor(Math.min(risk / d, cap / c));
      qty.textContent = n; amt.textContent = fmt(n * c); rsk.textContent = fmt(n * d);
    });
  }
  f.addEventListener('input', calc); calc();
})();"""

GLOSSARY = """<section id="glossary"><h2>מילון מושגים</h2>
<p class="meta">מה כל עמודה מודדת, מה היא אומרת על המגמה ואיך משתמשים בה. ההסברים מתארים שימוש מקובל בניתוח טכני ואינם המלצת השקעה.</p>
<section><dl>

<dt id="g-price">שער</dt><dd><p>מחיר הסגירה של יום המסחר האחרון. בארה״ב בדולרים, בתל אביב באגורות (100 אגורות = 1 ₪).
מחירי העבר מתואמים לדיבידנדים ולפיצולים, כדי שחלוקת דיבידנד לא תיראה בגרף כמו נפילה.</p></dd>

<dt id="g-chg">שינוי</dt><dd><p>אחוז השינוי מסגירת יום המסחר הקודם.</p>
<p><b>מה זה אומר על המגמה:</b> כמעט כלום לבד. יום אחד הוא רעש. תנועה חזקה מעניינת כשהיא מגיעה עם מחזור יחסי גבוה, או כשהיא פורצת רמה חשובה כמו שיא שנתי או ממוצע נע.</p></dd>

<dt id="g-gap">גאפ (פער)</dt><dd><p>הפער בין מחיר הפתיחה היום למחיר הסגירה אתמול. בדרך כלל נוצר מחדשות שהגיעו כשהבורסה סגורה, כמו דוחות כספיים, הודעות או אירועים בשווקים אחרים.</p>
<p><b>איך קוראים:</b> גאפ למעלה עם מחזור גבוה, שהמחיר לא חזר "לסגור" אותו (לרדת חזרה לסגירה של אתמול), נחשב סימן לביקוש אמיתי ולעיתים פותח מגמה. גאפ שנסגר עוד באותו יום מראה שהשוק ביטל את התגובה הראשונית.</p>
<p><b>סיכון:</b> גאפים הם הסיבה שסטופ לא תמיד עובד. אם מניה נפתחת מתחת לסטופ, המכירה תתבצע במחיר נמוך ממנו וההפסד יהיה גדול מהמתוכנן.</p></dd>

<dt id="g-rsi">RSI (מדד עוצמה יחסית)</dt><dd><p>Relative Strength Index, שפיתח ויילדר ב-1978. מדד בין 0 ל-100 שמשווה את גודל העליות לגודל הירידות ב-14 ימי המסחר האחרונים. 50 הוא איזון. מעל 70 נקרא "קנייה-יתר" ומתחת ל-30 "מכירת-יתר".</p>
<p><b>שתי קריאות הפוכות</b>, וחשוב לדעת באיזו משתמשים:</p>
<p>• <b>מומנטום:</b> RSI גבוה הוא סימן לכוח. במגמת עלייה חזקה RSI יכול להישאר מעל 70 שבועות, ומכירה רק בגלל "קנייה-יתר" מפספסת את המשך העלייה.</p>
<p>• <b>חזרה לממוצע:</b> RSI קיצוני מעיד שהמניה נמתחה רחוק מדי וצפוי תיקון.</p>
<p><b>שימוש מקובל לסוחר לונג:</b> RSI נמוך במניה שנמצאת במגמת עלייה ארוכה (מעל ממוצע 200) נחשב נקודת כניסה אפשרית בתיקון. RSI נמוך במניה במגמת ירידה יכול להמשיך לרדת ("לתפוס סכין נופלת").</p></dd>

<dt id="g-relvol">מחזור יחסי</dt><dd><p>מחזור המסחר היום חלקי ממוצע המחזור ב-20 ימי המסחר הקודמים. ×1 הוא יום רגיל, ×2 הוא פי שניים מהרגיל.</p>
<p><b>מה זה אומר:</b> זה מד ה"שכנוע" של התנועה. פריצה או גאפ עם מחזור גבוה (×1.5 ומעלה) משקפים השתתפות של גופים גדולים, ונוטים להחזיק יותר. פריצה עם מחזור נמוך חשודה כפריצת שווא.</p></dd>

<dt id="g-atr">ATR (טווח אמיתי ממוצע)</dt><dd><p>Average True Range: כמה המניה זזה בממוצע ביום, כולל גאפים. לכל יום לוקחים את הגדול מבין: גבוה פחות נמוך, המרחק מהגבוה לסגירה של אתמול, והמרחק מהנמוך לסגירה של אתמול. אחר כך ממצעים על פני 14 יום. בטבלה הוא מוצג כאחוז מהשער כדי שאפשר יהיה להשוות בין מניות. 1–2% זו מניה רגועה, 4% ומעלה זו מניה תנודתית.</p>
<p><b>מה זה אומר על המגמה:</b> שום דבר על הכיוון, רק על עוצמת התנודה. עלייה חדה ב-ATR מעידה שהתנודתיות מתגברת, לרוב סביב חדשות או בירידות. ה-ATR משמש כאן לקביעת הסטופ, ודרכו לגודל העסקה.</p></dd>

<dt id="g-stop">סטופ</dt><dd><p>המחיר שאם המניה יורדת אליו, יוצאים מהעסקה כדי להגביל את ההפסד. כאן: <b>שער הסגירה פחות 2×ATR</b>, באותו מטבע כמו השער.
המרווח של בערך שני ימי תנודה רגילה נועד למנוע יציאה בגלל רעש יומי.</p>
<p><b>מגבלות:</b> זו רמה מכנית שלא מתחשבת בתמיכות בגרף. כדאי לפתוח את הגרף (קישור בשם המניה) ולבדוק אם יש תמיכה מעט מתחת לסטופ. אם מזיזים את הסטופ אל מתחת לתמיכה, הכמות קטנה. הכפולה 2 היא מוסכמה מקובלת, לא ערך שהוכח כאופטימלי.</p></dd>

<dt id="g-size">כמות, סכום ₪, סיכון ₪ (גודל עסקה)</dt><dd>
<p>• <b>סיכון מקסימלי לעסקה</b> = שווי התיק × אחוז הסיכון. לדוגמה, 50,000 ₪ × 1% = 500 ₪.</p>
<p>• <b>כמות</b> = הקטן מבין (סיכון מקסימלי ÷ המרחק מהשער לסטופ) ו-(תקרה ÷ שער), מעוגל כלפי מטה.</p>
<p>• <b>סכום ₪</b> = כמות × שער, כלומר כמה כסף נכנס לעסקה.</p>
<p>• <b>סיכון ₪</b> = כמות × המרחק לסטופ, כלומר כמה תפסידי אם הסטופ ייפגע. כשהמספר קטן מהסיכון המקסימלי, התקרה היא שהגבילה את הכמות. זה קורה בכל מניה שהסטופ שלה קרוב מ-10% לשער.</p>
<p>הרעיון: כל עסקה מסכנת בערך את אותו סכום. במניה תנודתית קונים פחות מניות ובמניה רגועה יותר, כך שאף הפסד בודד לא פוגע בתיק בצורה משמעותית. עשרה הפסדים רצופים עולים כ-10% מהתיק.</p>
<p><b>מגבלות:</b> החישוב מניח כניסה בשער הסגירה. בפועל קונים בפתיחה הבאה ובמחיר אחר, אז כדאי לחשב שוב לפני הקנייה. לא כולל עמלות. במניות ארה״ב ההמרה לשקלים לפי שער הדולר היומי, והחישוב לא כולל את סיכון המטבע.</p></dd>

<dt id="g-spark">30 יום</dt><dd><p>גרף קו של מחירי הסגירה ב-30 ימי המסחר האחרונים. ירוק אם המניה עלתה בתקופה, אדום אם ירדה. נותן מבט מהיר על ההקשר: האם האות של היום הוא חלק מהמגמה או נגדה.</p></dd>

<dt id="g-signals">אותות</dt><dd>
<p>• <b>שיא 52 ש׳:</b> סגירה מעל המחיר הגבוה ביותר של השנה האחרונה. מעל הרמה הזו אין "מוכרים לכודים", כי כל מי שקנה בשנה האחרונה נמצא ברווח. מחקר אקדמי (George &amp; Hwang, 2004) מצא שמניות קרובות לשיא השנתי נטו בממוצע להמשיך ולהכות את השוק.</p>
<p>• <b>שפל 52 ש׳:</b> סגירה מתחת למחיר הנמוך ביותר של השנה. זה סימן חולשה. לסוחר לונג זה אות אזהרה, לא הזדמנות.</p>
<p>• <b>חצה ממוצע 50 / 200:</b> ממוצע נע הוא ממוצע מחירי הסגירה ב-50 או ב-200 הימים האחרונים. ממוצע 200 נחשב קו המגמה הארוכה: מחיר מעליו הוא מגמה עולה, ומחיר מתחתיו הוא מגמה יורדת. ממוצע 50 מייצג את המגמה הבינונית. חצייה כלפי מעלה (↑) היא שיפור, וחצייה כלפי מטה (↓) היא הרעה. ממוצעים מאחרים מטבעם, ובשוק שזז הצידה הם מייצרים הרבה אותות שווא.</p>
<p>• <b>RSI ו-גאפ:</b> כמו בעמודות למעלה. ירוק הוא אות חיובי ואדום הוא אות שלילי.</p>
<p>• <b>"שווה לשים לב"</b> מרכז מניות עם שני אותות או יותר באותו יום.</p></dd>

<dt id="g-together">איך קוראים את הכול יחד</dt><dd>
<p>אף אינדיקטור לבד לא אומר "לקנות". בניתוח טכני מחפשים כמה סימנים שמתיישבים זה עם זה, ובודקים בגרף. כמה תבניות מקובלות לסוחר לונג:</p>
<p>• <b>המשך מגמה (פריצה):</b> שיא 52 שבועות, מחזור יחסי ×1.5 ומעלה, המחיר מעל ממוצע 200, ו-RSI בין 50 ל-75.</p>
<p>• <b>תיקון בתוך מגמה עולה:</b> מניה מעל ממוצע 200 שה-RSI שלה ירד מתחת ל-30–40. נהוג לחכות לסימן היפוך, למשל יום עלייה עם מחזור, לפני כניסה.</p>
<p>• <b>ממה נהוג להימנע:</b> מניה בשפל שנתי ומתחת לממוצע 200, גם כשה-RSI נמוך.</p>
<p>הסטופ וגודל העסקה עובדים רק אם יוצאים בסטופ בפועל, בלי להזיז אותו כלפי מטה. כדאי לבדוק את השיטה על הנייר כמה שבועות לפני שמשקיעים כסף אמיתי.</p></dd>
</dl></section></section>"""

def page(sections, generated):
    return f"""<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>סורק טכני יומי</title>
<link rel="icon" href="data:image/svg+xml,{FAVICON}">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Rubik:wght@400;500;700&display=swap">
<style>{CSS}</style></head><body>
<header><h1>סורק טכני יומי</h1><nav><a href="#us">ארה״ב</a><a href="#il">ת״א</a><a href="#glossary">מילון</a></nav>
<span class="gen">עודכן {generated:%d.%m.%Y} בשעה {generated:%H:%M}</span></header>
<main>{PANEL}{sections}{GLOSSARY}</main>
<footer>נתונים: Yahoo Finance דרך yfinance, מחירים מתואמים לדיבידנדים ופיצולים. RSI לפי 14 ימים (Wilder).
מחזור יחסי = מחזור היום חלקי ממוצע 20 הימים הקודמים. הדף הוא כלי סינון ולא המלצת השקעה.</footer>
<script>{SIZING_JS}</script>
</body></html>"""


def usd_ils():
    try:
        return float(yf.Ticker("ILS=X").history(period="5d")["Close"].iloc[-1])
    except Exception as e:  # no rate -> US tables simply show no position size
        print(f"no USD/ILS rate: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/index.html")
    out = Path(ap.parse_args().out)

    sp = pd.read_csv(SP500_CSV)
    us = dict(zip(sp.Symbol.str.replace(".", "-", regex=False), sp.Security))
    il = {f"{k}.TA": v for k, v in TASE.items()}

    fx = usd_ils()
    sections = (
        market("us", "ארה״ב · S&P 500", scan(us),
               lambda s: f"https://www.tradingview.com/chart/?symbol={s.replace('-', '.')}",
               f"מחירים בדולר · דולר = {fx:.3f} ₪" if fx else "מחירים בדולר · אין שער דולר, גודל עסקה לא חושב", fx)
        + market("il", "תל אביב · מניות גדולות", scan(il),
                 lambda s: f"https://www.tradingview.com/chart/?symbol=TASE:{s.removesuffix('.TA')}",
                 "מחירים באגורות", 0.01)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(sections, dt.datetime.now(ZoneInfo("Asia/Jerusalem"))), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
