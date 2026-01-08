from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import time
import sys
import os

os.environ["OPENROUTER_API_KEY"] = (
    "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"
)
API_KEY = "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"


SYSTEM_PROMPT = """You are a decision function.
                Output ONLY valid JSON. No explanations, no extra text.
                Return exactly one JSON object with this schema:
                {"decision":"buy" or "sell","confidence":number between 0.0 and 1.0}
                """

def dax_data():
    """
    returns dax data of past 30 days as dict
    {date_str: [open_price, close_price, up]} for last 30 days
    while up is 1 for open_price < close_price else 0
    """
    ticker = "^GDAXI"
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)

    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )

    df.index = pd.to_datetime(df.index)

    open_series = df["Open"][ticker]
    close_series = df["Close"][ticker]

    dax_dict = {
        d.strftime("%Y-%m-%d"): [float(o), float(c)]
        for d, o, c in zip(open_series.index, open_series.values, close_series.values)
    }
    dax_labeled = {}
    for date, (open_p, close_p) in dax_dict.items():
        up = 1 if close_p > open_p else 0
        dax_labeled[date] = {"open": open_p, "close": close_p, "up": up}

    dax_labeled = sorted(dax_labeled.items())  # sort by date
    return dax_labeled


def print_progress(i, total, start_time, iteration=None, iterations=None, width=30):
    """
    fun little visualization of progress :)
    i: 0-based index of completed items
    total: total number of items
    start_time: time.time() when process started
    iteration: current iteration (1-based)
    iterations: total iterations
    """
    done = i + 1
    frac = done / total

    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)

    pct = int(frac * 100)
    elapsed = int(time.time() - start_time)

    iter_str = ""
    if iteration is not None and iterations is not None:
        iter_str = f" | {iteration}/{iterations}"

    sys.stdout.write(
        f"\r[{bar}] {done}/{total} {pct:3d}% | {elapsed:4d}s{iter_str} rounds"
    )
    sys.stdout.flush()

    if done == total:
        sys.stdout.write("\n")


def extract_json_by_braces(text: str):
    """
    Docstring for extract_json_by_braces
    takes LLM response and returns the json in it

    :param text: Description
    :type text: str
    """
    depth = 0
    start = None

    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

    print("No valid JSON found")
    raise ValueError("No valid JSON object found")


def ask_LLM(iteration, iterations):

    data = dax_data()
    total = len(data)
    LLM_decisions_internal = []
    start_time = time.time()

    for i, (date_str, daily_val) in enumerate(data):
        USER_MESSAGE = f"""
        You are a rational investor making a daily decision whether to invest in a DAX index fund or sell your assets for cash.

        Rules:
        - You can only choose between "buy" (invest today) or "sell".
        - No transaction costs.
        - The decision applies for today only.

        Information:
        - Date: {date_str}
        - Today's DAX level: {daily_val["open"]}
        - The prior decisions were {LLM_decisions_internal}

        Task:
        - Decide whether to buy or sell.
        - Confidence is your subjective probability (between 0.0 and 1.0) that tomorrow's DAX close will be higher than today's close.

        Return ONLY the JSON object. No extra text!
        """.strip()

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_MESSAGE},
                ],
            },
            timeout=20,
        )
        if response.status_code == 403:
            print("\033[31mERROR:\033[0m API credits exhausted")
            sys.exit(1)

        # print(response)
        try:
            raw = response.json()["choices"][0]["message"]["content"]
            response_json = extract_json_by_braces(raw)
        except Exception as e:
            print("\n========= DEBUG ERROR =========")
            print("Exception:", type(e).__name__)
            print(f"Response: {response}")
            print("Message:", e)
            print("\n--- RAW MODEL OUTPUT ---")
            print(raw)
            print("\n--- PARSED JSON ---")
            print(response_json)
            print("================================\n")
            raise

        response_json["date"] = date_str
        response_json["open"] = daily_val["open"]
        response_json["close"] = daily_val["close"]
        response_json["gain"] = daily_val["up"]

        # print(response_json)
        LLM_decisions_internal.append(response_json)
		trading_days_optimal= sum(d["up"] for d in LLM_decisions_internal)
		trading_days_actual = sum(1 for d in LLM_decisions_internal if d["decision"] == "buy")
        print_progress(i, total, start_time, iteration=iteration, iterations=iterations)
        save_to_text(i, LLM_decisions_internal)
    LLM_decisions_overall.append(LLM_decisions_internal)



def save_to_text(iteration: int, data: list, path="llm_decisions.txt"):
    import json
    import os

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            store = json.load(f)
    else:
        store = {}
	# aktuelle daten in store mit key der iteration setzten "1": [{...},{...}]
    store[str(iteration)] = data  # JSON keys müssen strings sein

    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)

LLM_decisions_overall = []
iterations = 5


for it in range(1, iterations + 1):
    ask_LLM(iteration=it, iterations=iterations)
	

## Analysis


def brier_score_run(run):
    errors = []
    for pred in run:
        p = float(pred["confidence"])
        y = float(pred["gain"])
        errors.append((p - y) ** 2)
    return sum(errors) / len(errors)


def ece_run(run, n_bins=10):
    p = np.array([d["confidence"] for d in run], dtype=float)
    y = np.array([d["gain"] for d in run], dtype=float)
    T = len(p)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.digitize(p, edges, right=True) - 1
    bin_id = np.clip(bin_id, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_id == b
        n_b = mask.sum()
        if n_b == 0:
            continue
        acc_b = y[mask].mean()
        conf_b = p[mask].mean()
        ece += (n_b / T) * abs(acc_b - conf_b)

    return float(ece)


# ---- Auswertung über alle Runs ----
bs_list_outer = [brier_score_run(run) for run in LLM_decisions_overall]
ece_list_outer = [ece_run(run, n_bins=10) for run in LLM_decisions_overall]

print("Brier per run:", bs_list_outer)
print(
    "Brier mean/std:",
    np.mean(bs_list_outer),
    np.std(bs_list_outer, ddof=1) if len(bs_list_outer) > 1 else 0.0,
)

print("ECE per run:", ece_list_outer)
print(
    "ECE mean/std:",
    np.mean(ece_list_outer),
    np.std(ece_list_outer, ddof=1) if len(ece_list_outer) > 1 else 0.0,
)


def calibration_bins(preds, n_bins=10):
    p = np.array([d["confidence"] for d in preds], dtype=float)
    y = np.array([d["gain"] for d in preds], dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(p, edges, right=True) - 1, 0, n_bins - 1)

    bin_centers, empirical, counts = [], [], []
    for b in range(n_bins):
        mask = bin_id == b
        if not mask.any():
            continue
        bin_centers.append(p[mask].mean())
        empirical.append(y[mask].mean())
        counts.append(mask.sum())

    return np.array(bin_centers), np.array(empirical), np.array(counts)


# Visualization
def plotting(decision_data: list):
    colors = [
        "#5d2e8c",
        "#ff6666",
        "#ccff66",
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
    ]

    fig = plt.figure()
    for i, run in enumerate(decision_data):
        x, emp, counts = calibration_bins(run, n_bins=10)
        plt.scatter(
            x, emp, s=counts * 20, alpha=0.6, color=colors[i], label=f"run {i+1}"
        )
    plt.plot([0, 1], [0, 1])

    # # all runs equal
    all_preds = [pred for run in decision_data for pred in run]
    x, emp, counts = calibration_bins(all_preds, n_bins=10)
    plt.scatter(
        x,
        emp,
        s=counts * 20,
        alpha=0.9,
        color="black",
        edgecolors="white",
        linewidths=0.6,
        label="all runs equal",
    )
    plt.legend(markerscale=0.5)

    ## Plotting
    plt.xlabel("Predicted probability (confidence)")
    plt.ylabel("Empirical probability (observed frequency)")
    plt.title("Calibration curve (Reliability diagram)")

    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.05, 1.05)

    # Text unter der gesamten Figure
    bs_list = [brier_score_run(run) for run in LLM_decisions_overall]
    ece_list = [ece_run(run, n_bins=10) for run in LLM_decisions_overall]
    fig.text(
        0.5,
        0.02,
        f"ECE = {np.mean(ece_list):.3f} | Brier = {np.mean(bs_list):.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    plt.subplots_adjust(bottom=0.18)
    plt.show()


plotting(LLM_decisions_overall)

modell = ["mistralai/mixtral-8x7b-instruct"] # TODO noch weitere Modelle hinzufügen
infos = {"irrelevant_facts" : [
    "Recent European equity markets have seen volatility related to shifting expectations about central bank interest rate policies in the U.S. and Eurozone, with investors closely monitoring upcoming Fed and ECB announcements. :contentReference[oaicite:0]{index=0}",
    "The DAX has exhibited fluctuations as technical indicators suggest short-term overbought conditions, with mixed momentum signals. :contentReference[oaicite:1]{index=1}",
    "Macro news regarding inflation and economic data releases in Germany influences investor sentiment about future corporate earnings. :contentReference[oaicite:2]{index=2}",
    "German manufacturing activity showed signs of contraction in recent activity reports, which can feed into risk-off sentiment among equity traders. :contentReference[oaicite:3]{index=3}",
    "Defense sector sentiment has supported some upside pressure across European indices amid geopolitical tensions and higher expected military spending. :contentReference[oaicite:4]{index=4}",
    "Technical analysis shows the DAX attempting to break above long-term resistance levels, suggesting traders are watching key psychological price zones. :contentReference[oaicite:5]{index=5}",
    "Broader European markets have recently traded near multi-week or record highs as investors rebalance portfolios ahead of central bank meetings. :contentReference[oaicite:6]{index=6}",
    "Falling benchmark interest rate expectations tend to support higher valuation multiples for equities, which influences momentum in major stock indices. :contentReference[oaicite:7]{index=7}",
    "Changes in commodity prices, such as energy or industrial metals, can disproportionately affect certain DAX components and overall index performance. :contentReference[oaicite:8]{index=8}",
    "Record-setting performance in major indices often reflects macroeconomic optimism but also increased sensitivity to shifts in economic data. :contentReference[oaicite:9]{index=9}",
    "Fiscal policy signals from Germany and the wider EU shape investor expectations about future growth prospects for domestic companies. :contentReference[oaicite:10]{index=10}",
    "Cross-market influences from U.S. equity and bond yields have had spillover effects on European equities, including the DAX. :contentReference[oaicite:11]{index=11}",
    "Sector leadership within the DAX can rotate rapidly, especially between cyclical sectors like autos and defensive sectors like healthcare, affecting short-term index dynamics. :contentReference[oaicite:12]{index=12}",
    "Retail and institutional investor flows into or out of European equity ETFs contribute to intraday and short-term price patterns. :contentReference[oaicite:13]{index=13}",
    "German business sentiment indices have shown improvement, which can bolster confidence in future corporate performance. :contentReference[oaicite:14]{index=14}",
    "Risk assets like stocks have responded to geopolitical developments that affect global risk sentiment without necessarily altering economic fundamentals. :contentReference[oaicite:15]{index=15}",
    "European banks and financial stocks have recently contributed to broader market gains, reflecting anticipated regulatory or policy changes. :contentReference[oaicite:16]{index=16}",
    "Seasonally light trading volume around holidays tends to increase volatility and exaggerate short-term moves in major indices. :contentReference[oaicite:17]{index=17}",
    "Currency movements between the euro and major currencies influence multinational earnings expectations of DAX-listed firms. :contentReference[oaicite:18]{index=18}",
    "Trading strategies focused on technical breakouts are common around significant price levels in the DAX, affecting order flow. :contentReference[oaicite:19]{index=19}",
    "Persistently low or steady interest rate expectations tend to support equity valuations, especially in sectors sensitive to financing costs. :contentReference[oaicite:20]{index=20}",
    "Headlines related to economic growth data in the euro area have an impact on market positioning among global investors. :contentReference[oaicite:21]{index=21}",
    "Performance data for individual heavyweight constituents of the index can skew short-term index moves even when macro conditions are stable. :contentReference[oaicite:22]{index=22}",
    "Defensive sectors can act as a cushion in times of broader market uncertainty, influencing the index composition’s effect on overall movement. :contentReference[oaicite:23]{index=23}",
    "Expectations for future corporate earnings revisions are a key input into how traders price stocks prior to earnings seasons. :contentReference[oaicite:24]{index=24}",
    "Economic indicators like consumer confidence and business morale feed into medium-term equity valuations. :contentReference[oaicite:25]{index=25}",
    "Risk sentiment among global investors often shifts with macroeconomic headlines even if the underlying data does not change materially. :contentReference[oaicite:26]{index=26}",
    "Movements in growth-sensitive equity sectors often reflect changes in yield curves and expectations of future rate paths. :contentReference[oaicite:27]{index=27}",
    "Market participants watch volatility indices that reflect future uncertainty expectations, influencing their risk-taking behavior. :contentReference[oaicite:28]{index=28}",
    "Short-term technical price movements are often driven by liquidity conditions and algorithmic trading around key price levels. :contentReference[oaicite:29]{index=29}",],
	"relevant_facts" : [
    "Recent European equity markets have seen volatility related to shifting expectations about central bank interest rate policies in the U.S. and Eurozone, with investors closely monitoring upcoming Fed and ECB announcements.",
    "The DAX has exhibited fluctuations as technical indicators suggest short-term overbought conditions, with mixed momentum signals.",
    "Macro news regarding inflation and economic data releases in Germany influences investor sentiment about future corporate earnings.",
    "German manufacturing activity showed signs of contraction in recent activity reports, which can feed into risk-off sentiment among equity traders.",
    "Defense sector sentiment has supported some upside pressure across European indices amid geopolitical tensions and higher expected military spending.",
    "Technical analysis shows the DAX attempting to break above long-term resistance levels, suggesting traders are watching key psychological price zones.",
    "Broader European markets have recently traded near multi-week or record highs as investors rebalance portfolios ahead of central bank meetings.",
    "Falling benchmark interest rate expectations tend to support higher valuation multiples for equities, which influences momentum in major stock indices.",
    "Changes in commodity prices, such as energy or industrial metals, can disproportionately affect certain DAX components and overall index performance.",
    "Record-setting performance in major indices often reflects macroeconomic optimism but also increased sensitivity to shifts in economic data.",
    "Fiscal policy signals from Germany and the wider EU shape investor expectations about future growth prospects for domestic companies.",
    "Cross-market influences from U.S. equity and bond yields have had spillover effects on European equities, including the DAX.",
    "Sector leadership within the DAX can rotate rapidly, especially between cyclical sectors like autos and defensive sectors like healthcare, affecting short-term index dynamics.",
    "Retail and institutional investor flows into or out of European equity ETFs contribute to intraday and short-term price patterns.",
    "German business sentiment indices have shown improvement, which can bolster confidence in future corporate performance.",
    "Risk assets like stocks have responded to geopolitical developments that affect global risk sentiment without necessarily altering economic fundamentals.",
    "European banks and financial stocks have recently contributed to broader market gains, reflecting anticipated regulatory or policy changes.",
    "Seasonally light trading volume around holidays tends to increase volatility and exaggerate short-term moves in major indices.",
    "Currency movements between the euro and major currencies influence multinational earnings expectations of DAX-listed firms.",
    "Trading strategies focused on technical breakouts are common around significant price levels in the DAX, affecting order flow.",
    "Persistently low or steady interest rate expectations tend to support equity valuations, especially in sectors sensitive to financing costs.",
    "Headlines related to economic growth data in the euro area have an impact on market positioning among global investors.",
    "Performance data for individual heavyweight constituents of the index can skew short-term index moves even when macro conditions are stable.",
    "Defensive sectors can act as a cushion in times of broader market uncertainty, influencing the index compositions effect on overall movement.",
    "Expectations for future corporate earnings revisions are a key input into how traders price stocks prior to earnings seasons.",
    "Economic indicators like consumer confidence and business morale feed into medium-term equity valuations.",
    "Risk sentiment among global investors often shifts with macroeconomic headlines even if the underlying data does not change materially.",
    "Movements in growth-sensitive equity sectors often reflect changes in yield curves and expectations of future rate paths.",
    "Market participants watch volatility indices that reflect future uncertainty expectations, influencing their risk-taking behavior.",
    "Short-term technical price movements are often driven by liquidity conditions and algorithmic trading around key price levels.",
    ]
}
for MODEL in modelle:
	for info in infos:
		data = ask_LLM(iterations, version, infos)        # where version is either without infos (0), with extra  irrelevant infos (1), or with relevant info (3)
      	plot(data, titel)
        
  




















