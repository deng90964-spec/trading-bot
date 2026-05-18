pythonimport os
import smtplib
import yfinance as yf
import pandas as pd
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from zoneinfo import ZoneInfo
from anthropic import Anthropic

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
EMAIL_FROM = os.environ["EMAIL_FROM"]
EMAIL_PASSWORD = os.environ["EMAIL_PASSWORD"]
EMAIL_TO = os.environ["EMAIL_TO"]
ACCOUNT_SIZE_USD = float(os.environ.get("ACCOUNT_SIZE", "5000"))
RISK_PER_TRADE_PCT = 1.0
ET = ZoneInfo("America/New_York")
STOP_BUFFER_PCT = 0.1

def fetch_data(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period="2d", interval="5m", prepost=True)
    if df.empty:
        raise RuntimeError(f"无法获取 {ticker} 行情")
    df.index = df.index.tz_convert(ET)
    return df

def find_premarket_range(df):
    today = datetime.now(ET).date()
    premarket = df[
        (df.index.date == today) &
        (((df.index.hour >= 4) & (df.index.hour < 9)) |
         ((df.index.hour == 9) & (df.index.minute < 30)))
    ]
    if premarket.empty:
        return None, None
    return premarket["High"].max(), premarket["Low"].min()

def check_breakout(df, pm_high, pm_low):
    today = datetime.now(ET).date()
    window = df[
        (df.index.date == today) &
        (((df.index.hour == 9) & (df.index.minute >= 35)) |
         (df.index.hour == 10) |
         ((df.index.hour == 11) & (df.index.minute == 0)))
    ]
    if window.empty:
        return None
    for idx, row in window.iterrows():
        if row["High"] > pm_high:
            entry = pm_high
            stop_loss = row["Low"] * (1 - STOP_BUFFER_PCT / 100)
            risk = entry - stop_loss
            return {
                "direction": "做多 LONG",
                "trigger_time": idx.strftime("%H:%M ET"),
                "entry": round(entry, 2),
                "stop_loss": round(stop_loss, 2),
                "tp1": round(entry + 1 * risk, 2),
                "tp2": round(entry + 2 * risk, 2),
                "tp3_trail": round(entry + 2.5 * risk, 2),
                "risk_pct": round(risk / entry * 100, 3),
            }
        if row["Low"] < pm_low:
            entry = pm_low
            stop_loss = row["High"] * (1 + STOP_BUFFER_PCT / 100)
            risk = stop_loss - entry
            return {
                "direction": "做空 SHORT",
                "trigger_time": idx.strftime("%H:%M ET"),
                "entry": round(entry, 2),
                "stop_loss": round(stop_loss, 2),
                "tp1": round(entry - 1 * risk, 2),
                "tp2": round(entry - 2 * risk, 2),
                "tp3_trail": round(entry - 2.5 * risk, 2),
                "risk_pct": round(risk / entry * 100, 3),
            }
    return None

def analyze_ticker(ticker):
    df = fetch_data(ticker)
    pm_high, pm_low = find_premarket_range(df)
    if pm_high is None:
        return {"ticker": ticker, "error": "无盘前数据"}
    current = df["Close"].iloc[-1]
    signal = check_breakout(df, pm_high, pm_low)
    return {
        "ticker": ticker,
        "pm_high": round(pm_high, 2),
        "pm_low": round(pm_low, 2),
        "range_pct": round((pm_high - pm_low) / pm_low * 100, 2),
        "current": round(current, 2),
        "signal": signal,
    }

def recommend_ticker(spy_result, qqq_result):
    spy_sig = spy_result.get("signal")
    qqq_sig = qqq_result.get("signal")
    if spy_sig and qqq_sig:
        if spy_result["range_pct"] < qqq_result["range_pct"]:
            return "SPY", "SPY区间更窄，假突破风险低，推荐新手/求稳选SPY"
        return "QQQ", "QQQ区间更宽，趋势行情潜力大，但假突破风险也高"
    if spy_sig:
        return "SPY", "仅SPY突破，QQQ未表态，可选SPY"
    if qqq_sig:
        return "QQQ", "仅QQQ突破，SPY未表态，可选QQQ"
    return None, "两个标的均未突破，今日跳过"

def get_claude_commentary(spy_result, qqq_result, recommended, reason):
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""你是一位严谨的期权交易顾问。今天的5分钟突破策略数据：

【SPY】盘前高:{spy_result.get('pm_high')} 盘前低:{spy_result.get('pm_low')} 当前价:{spy_result.get('current')} 信号:{spy_result.get('signal') or '未突破'}
【QQQ】盘前高:{qqq_result.get('pm_high')} 盘前低:{qqq_result.get('pm_low')} 当前价:{qqq_result.get('current')} 信号:{qqq_result.get('signal') or '未突破'}
【系统推荐】:{recommended or '无'} 【理由】:{reason}

请用中文150字以内给出：
1. 当前形态质量评估
2. 期权选择建议（1DTE首选，ATM或浅OTM）
3. 最重要的一条风险提醒

注意：技术分析参考，不是投资建议。"""
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

def format_signal_block(result):
    t = result["ticker"]
    if "error" in result:
        return f"\n【{t}】⚠️ {result['error']}\n"
    sig = result.get("signal")
    txt = f"""
━━━━━━━━━━━━━━━━━━━━━━
【{t}】
━━━━━━━━━━━━━━━━━━━━━━
盘前高: {result['pm_high']}
盘前低: {result['pm_low']}
区间宽: {result['range_pct']}%
当前价: {result['current']}
"""
    if sig:
        risk_dollars = ACCOUNT_SIZE_USD * RISK_PER_TRADE_PCT / 100
        txt += f"""
🎯 突破信号触发！
方向:     {sig['direction']}
触发时间: {sig['trigger_time']}
入场价:   {sig['entry']}
止损价:   {sig['stop_loss']} (含0.1%缓冲)
风险幅度: {sig['risk_pct']}%

📊 分批止盈：
  TP1(1:1) → {sig['tp1']}  卖50%
  TP2(2:1) → {sig['tp2']}  卖30%
  TP3(2.5:1) → {sig['tp3_trail']}  剩20%奔跑

💰 仓位建议(账户${ACCOUNT_SIZE_USD:.0f}, 风险{RISK_PER_TRADE_PCT}%):
  本次最大可亏: ${risk_dollars:.0f}
  期权预算: 单张约${risk_dollars/0.4:.0f}(止损40%反算)
  建议1DTE, ATM或浅OTM
"""
    else:
        txt += "\n⏸ 暂无突破信号\n"
    return txt

def send_email(subject, body):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.send_message(msg)
    print(f"✅ 邮件已发送: {subject}")

def main():
    today_str = datetime.now(ET).strftime("%Y-%m-%d %A")
    print(f"🔍 开始分析 SPY/QQQ - {today_str}")
    try:
        spy = analyze_ticker("SPY")
        qqq = analyze_ticker("QQQ")
        recommended, reason = recommend_ticker(spy, qqq)
        commentary = get_claude_commentary(spy, qqq, recommended, reason)
        body = f"""【SPY/QQQ 5分钟突破策略】每日分析
日期: {today_str}
账户: ${ACCOUNT_SIZE_USD:.0f} | 单笔风险: {RISK_PER_TRADE_PCT}%
{format_signal_block(spy)}
{format_signal_block(qqq)}
━━━━━━━━━━━━━━━━━━━━━━
🎯 系统推荐: {recommended or '今日跳过'}
理由: {reason}

━━━━━━━━━━━━━━━━━━━━━━
🤖 Claude点评:
{commentary}

━━━━━━━━━━━━━━━━━━━━━━
🚨 今日入场前确认:
□ 今天是否FOMC/重大事件? → 是则不做
□ 今天是否CPI/大股财报? → 是则减半
□ 止损位置已设好?
□ 单笔风险 < 账户1%?
□ 今天已有仓位? → 有则不再开

⚠️ 本报告为技术分析参考，不构成投资建议
⚠️ 下单前用券商行情核对价格
⚠️ 11:00 AM ET后不再入场
"""
        send_email(f"📈 SPY/QQQ分析 {today_str}", body)
    except Exception as e:
        send_email(f"❌ 脚本出错 {today_str}", f"错误:\n{str(e)}")
        raise

if __name__ == "__main__":
    main()
