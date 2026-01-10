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

API_KEY = os.getenv("OPENROUTER_API_KEY")  # <-- KEY NICHT hardcoden!
if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set in environment variables.")

SYSTEM_PROMPT = (
    "You are a decision function.\n"
    "Output ONLY valid JSON. No explanations, no extra text.\n"
    "Return exactly one JSON object with this schema:\n"
    '{"decision":"buy" or "sell","confidence":number between 0.0 and 1.0}'
)


# -------------------- DATA --------------------
def dax_data(days=30):
    ticker = "^GDAXI"
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

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

    return sorted(dax_labeled.items())


# -------------------- UTIL --------------------
def print_progress(i, total, start_time, iteration=None, iterations=None, width=30):
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

    raise ValueError("No valid JSON object found")


# -------------------- SCORING --------------------
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


# -------------------- PLOTTING --------------------
def plotting(decision_data, title: str):
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]

    fig = plt.figure()

    for i, run in enumerate(decision_data):
        x, emp, counts = calibration_bins(run, n_bins=10)
        plt.scatter(
            x,
            emp,
            s=counts * 20,
            alpha=0.6,
            color=colors[i % len(colors)],
            label=f"run {i+1}",
        )

    plt.plot([0, 1], [0, 1])

    # all runs pooled
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
        label="all runs pooled",
    )

    plt.legend(markerscale=0.7)
    plt.xlabel("Predicted probability (confidence)")
    plt.ylabel("Empirical probability (observed frequency)")
    plt.title(f"Calibration curve — {title}")
    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.05, 1.05)

    bs_list = [brier_score_run(run) for run in decision_data]
    ece_list = [ece_run(run, n_bins=10) for run in decision_data]
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


# -------------------- CORE EXPERIMENT --------------------
def build_user_message(date_str, open_price, prior_decisions, extra_info=None):
    extra_block = ""
    if extra_info is not None:
        extra_block = f"\n- Today's additional information: {extra_info}"

    return f"""
You are a rational investor making a daily decision whether to invest in a DAX index fund or sell your assets for cash.

Rules:
- You can only choose between "buy" (invest today) or "sell".
- No transaction costs.
- The decision applies for today only.

Information:
- Date: {date_str}
- Today's DAX level: {open_price}
- The prior decisions were {prior_decisions}{extra_block}

Task:
- Decide whether to buy or sell.
- Confidence is your subjective probability (between 0.0 and 1.0) that tomorrow's DAX close will be higher than today's close.

Return ONLY the JSON object. No extra text!
""".strip()


def ask_llm_run(model, iteration, iterations, treatment_facts=None):
    data = dax_data(days=30)
    total = len(data)
    decisions_internal = []
    start_time = time.time()

    for i, (date_str, daily_val) in enumerate(data):
        extra = None
        if treatment_facts is not None:
            extra = treatment_facts[i % len(treatment_facts)]

        user_message = build_user_message(
            date_str=date_str,
            open_price=daily_val["open"],
            prior_decisions=decisions_internal,
            extra_info=extra,
        )

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            },
            timeout=20,
        )
        # print(response)
        if response.status_code == 403:
            print("\033[31mERROR:\033[0m API credits exhausted")
            sys.exit(1)

        raw = response.json()["choices"][0]["message"]["content"]
        response_json = extract_json_by_braces(raw)

        response_json["date"] = date_str
        response_json["open"] = daily_val["open"]
        response_json["close"] = daily_val["close"]
        response_json["gain"] = daily_val["up"]

        decisions_internal.append(response_json)
        print_progress(i, total, start_time, iteration=iteration, iterations=iterations)

    return decisions_internal


def run_experiment(model, iterations, treatment_facts=None):
    runs = []
    for it in range(1, iterations + 1):
        run = ask_llm_run(
            model=model,
            iteration=it,
            iterations=iterations,
            treatment_facts=treatment_facts,
        )
        runs.append(run)
    return runs


# -------------------- TREATMENT FACTS --------------------
irrelevant_facts = [
    "The DAX was first introduced in 1988 with a base value of 1,000 points.",
    "The name DAX stands for German Stock Index and is a registered trademark.",
    "The index is calculated in Frankfurt and updated continuously during Xetra trading hours.",
    "The DAX is a performance index that reinvests dividends for calculation purposes.",
    "The composition of the DAX is reviewed regularly according to predefined rules.",
    "The index consists of exactly 40 companies after being expanded from 30 in 2021.",
    "Many DAX-listed companies generate a large share of their revenue outside Germany.",
    "The DAX is often described as a barometer of the German economy, although it represents only a subset.",
    "The index is calculated in euros and based on free-float market capitalization.",
    "Trading in DAX derivatives is possible both on exchanges and over the counter.",
    "The DAX is one of the most well-known stock indices in Europe and followed worldwide.",
    "Historically, the DAX has undergone several structural reforms in its rulebook.",
    "The index is administered by Deutsche Börse AG.",
    "Some DAX companies were founded in the nineteenth century.",
    "The DAX is reported daily in financial media through tables and summaries.",
    "DAX index levels are often used as reference points in economic retrospectives.",
    "The DAX is part of a broader index family including MDAX, SDAX, and TecDAX.",
    "The visual presentation of the DAX differs across financial news platforms.",
    "The DAX is frequently used as an example of a stock index in academic textbooks.",
    "Historical DAX data is publicly available and often used for academic research.",
    "Many retail investors recognize the DAX without being able to name its constituents.",
    "The DAX is often mentioned alongside international benchmark indices in news reports.",
    "The DAX serves as the underlying asset for structured products and certificates.",
    "Some DAX companies are headquartered outside Frankfurt despite the index being calculated there.",
    "The DAX is replicated by numerous investment funds and exchange-traded funds.",
    "Index membership is determined by transparent and publicly available criteria.",
    "Over its history, the DAX has reached multiple new record highs.",
    "DAX levels are commonly displayed in charts using linear or logarithmic scales.",
    "The DAX is a permanent fixture of economic reporting in Germany.",
    "Outside finance, the term DAX is often used as a synonym for the German stock market.",
]
relevant_facts = [
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


# -------------------- MAIN --------------------
if __name__ == "__main__":
    models = ["mistralai/mixtral-8x7b-instruct"]
    iterations = 1

    treatments = [
        ("without information", None),
        ("with irrelevant information", irrelevant_facts),
        ("with relevant information", relevant_facts),
    ]
    all_results = {}

    for model in models:
        all_results[model] = {}

        for title, facts in treatments:
            print(f"\nRunning: {model} | {title}")
            decision_data = run_experiment(
                model=model,
                iterations=iterations,
                treatment_facts=facts,
            )
            all_results[model][title] = decision_data

    for model, model_results in all_results.items():
        for title, decision_data in model_results.items():
            plotting(decision_data, f"{model} — {title}")
