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


SYSTEM_PROMPT = """You are a decision function.
                Output ONLY valid JSON. No explanations, no extra text.
                Return exactly one JSON object with this schema:
                {"decision":"buy" or "sell","confidence":number between 0.0 and 1.0}
                """
MODEL = "mistralai/mixtral-8x7b-instruct"
API_KEY = "APIKEY here"


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
        print_progress(i, total, start_time, iteration=iteration, iterations=iterations)
    LLM_decisions_overall.append(LLM_decisions_internal)


LLM_decisions_overall = []
iterations = 5

for it in range(1, iterations + 1):
    ask_LLM(iteration=it, iterations=iterations)


## Data visualization


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
bs_list = [brier_score_run(run) for run in LLM_decisions_overall]
ece_list = [ece_run(run, n_bins=10) for run in LLM_decisions_overall]

print("Brier per run:", bs_list)
print(
    "Brier mean/std:",
    np.mean(bs_list),
    np.std(bs_list, ddof=1) if len(bs_list) > 1 else 0.0,
)

print("ECE per run:", ece_list)
print(
    "ECE mean/std:",
    np.mean(ece_list),
    np.std(ece_list, ddof=1) if len(ece_list) > 1 else 0.0,
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


# calibration per run

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
for i, run in enumerate(LLM_decisions_overall):
    x, emp, counts = calibration_bins(run, n_bins=10)
    plt.scatter(x, emp, s=counts * 20, alpha=0.6, color=colors[i], label=f"run {i+1}")
plt.plot([0, 1], [0, 1])

# # all runs equal
all_preds = [pred for run in LLM_decisions_overall for pred in run]
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
