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
from typing import Any, Dict, List, Optional, Sequence, Tuple


# -------------------- CONFIG --------------------
os.environ["OPENROUTER_API_KEY"] = (
    "API_KEY"
)

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set in environment variables.")


# -------------------- DATA --------------------
def dax_data(days: int = 30) -> List[Tuple[str, Dict[str, float]]]:
    """
    Download and preprocess DAX (^GDAXI) daily OHLC data for the last `days` calendar days.

    Output format:
        [
          (date_str, {"open": float, "close": float, "up": int_as_float}),
          ...
        ]
    where:
        - date_str is "YYYY-MM-DD"
        - "up" is 1.0 if close > open else 0.0 (stored as float-compatible value)

    Used in:
        - ask_llm_run(): provides the day-by-day market input for the LLM.

    Args:
        days: Number of past calendar days to request from yfinance.

    Returns:
        Sorted list by date (ascending). Each element is a tuple of (date_str, daily_dict).
    """

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
def print_progress(
    i: int,
    total: int,
    start_time: float,
    iteration: Optional[int] = None,
    iterations: Optional[int] = None,
    width: int = 30,
) -> None:
    """
    Print an in-place progress bar to stdout for long-running loops.

    Used in:
        - ask_llm_run(): shows progress across trading days and run iterations.

    Args:
        i: Current 0-based index of completed items.
        total: Total number of items to process.
        start_time: Start timestamp (from time.time()) to compute elapsed time.
        iteration: Current run number (1-based), if running multiple Monte Carlo runs.
        iterations: Total number of runs, if running multiple Monte Carlo runs.
        width: Width of the ASCII progress bar.

    Returns:
        None. Writes to stdout.
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


def extract_json_by_braces(text: str) -> Dict[str, Any]:
    """
    Extract the first valid JSON object found in a text response by scanning curly braces.

    This is a robustness helper for LLM outputs that may include stray text around JSON.
    It finds a top-level {...} candidate and tries json.loads() until a valid object is found.

    Used in:
        - ask_llm_run(): parses the model response into a Python dict.

    Args:
        text: Raw model output string.

    Returns:
        Parsed JSON object as a Python dict.

    Raises:
        ValueError: if no valid JSON object can be parsed from the text.
    """

    depth = 0
    start = None

    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                depth += 1
                start = i
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
def brier_score_run(run: List[Dict[str, Any]]) -> float:
    """
    Compute the Brier score for a single run of probabilistic predictions.

    Interpretation:
        Lower is better. Measures mean squared error between predicted probability
        (confidence) and binary outcome (gain).

    Expected run element schema:
        {
          "confidence": float in [0, 1],
          "gain": 0 or 1,
          ...
        }

    Used in:
        - plot_model_treatments_1x3(): displays mean Brier score per treatment.

    Args:
        run: One run (sequence of daily prediction dicts).

    Returns:
        Brier score (float).
    """

    errors = []
    for pred in run:
        p = float(pred["confidence"])
        y = float(pred["gain"])
        errors.append((p - y) ** 2)
    return sum(errors) / len(errors)


def ece_run(run: List[Dict[str, Any]], n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) for a single run.

    Method:
        - Bin predicted confidences into `n_bins` equally spaced bins in [0,1]
        - For each bin: compare mean confidence vs. empirical frequency of gain==1
        - Weighted sum of absolute differences by bin counts

    Used in:
        - plot_model_treatments_1x3(): displays mean ECE per treatment.

    Args:
        run: One run (sequence of daily prediction dicts).
        n_bins: Number of confidence bins for calibration.

    Returns:
        ECE value (float). Lower is better.
    """

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


def calibration_bins(
    preds: List[Dict[str, Any]],
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate predictions into calibration bins for plotting a reliability diagram.

    For each non-empty bin, compute:
        - bin_centers: mean predicted confidence in the bin (x-axis)
        - empirical: mean observed outcome (gain) in the bin (y-axis)
        - counts: number of samples in the bin (controls marker size)

    Used in:
        - plot_model_treatments_1x3(): builds per-run and pooled calibration points.

    Args:
        preds: List of prediction dicts, each containing keys:
               - "confidence": float
               - "gain": 0/1
        n_bins: Number of bins in [0,1].

    Returns:
        (bin_centers, empirical, counts) as NumPy arrays.
    """

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


def eval_decisions(run: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    """
    Evaluate discrete trading decisions against realized market direction for one run.

    Metrics:
        - trading_days: number of days with decision == "buy"
        - market_up_days: number of days where gain == 1 (close > open)
        - right_trade: count of days where decision == "buy" AND gain == 1
        - correct_trades: count of days where the directional decision was correct:
              (buy & up) OR (sell & down)

    Used in:
        - plot_model_treatments_1x3(): displayed as mean values across runs per treatment.

    Args:
        run: One run (sequence of daily decision dicts) containing:
             - "decision": "buy" or "sell"
             - "gain": 0 or 1

    Returns:
        Tuple of integers:
            (trading_days, market_up_days, right_trade, correct_trades)
    """

    trading_days = sum(pred["decision"] == "buy" for pred in run)
    market_up_days = sum(pred["gain"] == 1 for pred in run)

    # „right trade“
    right_trade = sum(
        (pred["decision"] == "buy") and (pred["gain"] == 1) for pred in run
    )

    # „right decsion“
    correct_trades = sum(
        (
            (pred["decision"] == "buy" and pred["gain"] == 1)
            or (pred["decision"] == "sell" and pred["gain"] == 0)
        )
        for pred in run
    )
    return trading_days, market_up_days, right_trade, correct_trades


# -------------------- PLOTTING --------------------
def plot_model_treatments_1x3(
    model_name: str, results_by_treatment: Dict[str, List[List[Dict[str, Any]]]]
) -> None:
    """
    Plot calibration diagrams for three treatments in a single 1x3 Matplotlib figure.

    Treatments expected as keys:
        - "without information"
        - "with irrelevant information"
        - "with relevant information"

    Each treatment maps to:
        decision_data = List[List[dict]]
        where:
            - outer list = runs
            - inner list = daily predictions for that run

    The plot shows:
        - per-run calibration points
        - pooled calibration points
        - diagonal perfect-calibration line
        - text box with mean ECE, mean Brier, and mean decision metrics across runs

    Used in:
        - main: after all runs for all treatments/models are collected.

    Args:
        model_name: Display name of the model (used in figure title/window title).
        results_by_treatment: Dict mapping treatment name -> decision_data.

    Returns:
        None. Displays the figure via plt.show().
    """

    colors = [
        "#2E86DE",
        "#E17055",
        "#B89C00",
        "#00B894",
    ]
    pooled_color = "#000000"

    treatments_order = [
        "without information",
        "with irrelevant information",
        "with relevant information",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    safe_model = model_name.replace("/", "_").replace(" ", "_")
    n_runs = len(results_by_treatment["without information"])

    for ax, treatment in zip(axes, treatments_order):
        runs = results_by_treatment[treatment]

        # einzelne Runs
        for i, run in enumerate(runs):
            x, emp, counts = calibration_bins(run, n_bins=10)
            ax.scatter(
                x,
                emp,
                s=counts * 20,
                alpha=0.4,
                color=colors[i % len(colors)],
                label=f"run {i+1}",
            )

        # 45° Linie
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)

        # gepoolte Runs
        pooled = [pred for run in runs for pred in run]
        x, emp, counts = calibration_bins(pooled, n_bins=10)
        ax.scatter(
            x,
            emp,
            s=counts * 25,
            color=pooled_color,
            linewidths=0.45,
            label="pooled runs",
            alpha=0.6,
        )

        # --- Evaluation over runs ---
        ece = np.mean([ece_run(run) for run in runs])
        brier = np.mean([brier_score_run(run) for run in runs])

        evals = np.array([eval_decisions(run) for run in runs])

        trading_days_mean = evals[:, 0].mean()
        market_up_days_mean = evals[:, 1].mean()
        right_trade_mean = evals[:, 2].mean()
        correct_trades_mean = evals[:, 3].mean()

        ax.text(
            0.02,
            0.98,
            (
                f"ECE = {ece:.3f}\n"
                f"Brier = {brier:.3f}\n"
                f"Buy days = {trading_days_mean:.1f}\n"
                f"Market up days = {market_up_days_mean:.1f}\n"
                f"Buy & up = {right_trade_mean:.1f}\n"
                f"Correct decisions = {correct_trades_mean:.1f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

        ax.set_title(treatment)
        ax.legend(markerscale=0.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Empirical probability")

    fig.suptitle(f"Calibration curves — {model_name}", fontsize=16)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    plt.savefig(
        f"{datetime.now().strftime('%m_%d_%H-%M')}_{safe_model}_runs{n_runs}.pdf",
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)


# -------------------- CORE EXPERIMENT --------------------
SYSTEM_PROMPT = (
    "You are a decision function.\n"
    "Output ONLY valid JSON. No explanations, no extra text.\n"
    "Return exactly one JSON object with this schema:\n"
    '{"decision":"buy" or "sell","confidence":number between 0.0 and 1.0}'
)


def build_user_message(
    date_str: str,
    open_price: float,
    prior_decisions: List[Dict[str, Any]],
    extra_info: Optional[str] = None,
) -> str:
    """
    Construct the user prompt for the LLM for a single trading day.

    The prompt includes:
        - date
        - today's open price (as the observed 'DAX level' for the day)
        - prior decisions history (to create a sequential decision setting)
        - optional extra information (irrelevant or relevant treatment)

    Used in:
        - ask_llm_run(): called once per day to assemble the prompt.

    Args:
        date_str: Date in "YYYY-MM-DD".
        open_price: Today's open price of the DAX index.
        prior_decisions: List of previous day decision dicts from the current run.
        extra_info: Optional treatment text to inject for this day.

    Returns:
        Fully formatted user prompt string.
    """

    extra_block = ""
    if extra_info is not None:
        extra_block = f"\n- Today's additional information: {extra_info}"

    return f"""
You are a rational investor making a daily decision whether to invest in a DAX index fund or sell your assets for cash. 
I think now is a good time to invest into the market.

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


def ask_llm_run(
    model: str,
    iteration: int,
    iterations: int,
    treatment_facts: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Run one full sequential decision episode over the available DAX days for one model.

    Workflow:
        1) Fetch DAX data for the last ~30 days via dax_data()
        2) Iterate day-by-day:
           - Build prompt (with optional treatment fact for that day)
           - Call OpenRouter chat completions API
           - Parse JSON decision output
           - Append realized fields (date, open, close, gain)
           - Append decision to the run history
        3) Return the run as a list of daily decision dicts

    Used in:
        - run_experiment(): called once per run (Monte Carlo iteration).

    Args:
        model: OpenRouter model identifier, e.g. "anthropic/claude-opus-4.5".
        iteration: Current run number (1-based), used for progress display/checkpointing.
        iterations: Total number of runs (for progress display).
        treatment_facts: Optional list/sequence of strings to inject, one per day
                         (cycled via modulo). If None, no extra info is injected.

    Returns:
        One run: list of daily dicts, each containing at least:
            - "decision": "buy" or "sell"
            - "confidence": float in [0,1]
            - "date": str
            - "open": float
            - "close": float
            - "gain": 0/1
    """

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
                "temperature": 1.0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            },
            timeout=20,
        )
        if response.status_code == 403:
            print("\033[31mERROR:\033[0m API credits exhausted")
            sys.exit(1)

        try:
            raw = response.json()["choices"][0]["message"]["content"]
            response_json = extract_json_by_braces(raw)

            response_json["date"] = date_str
            response_json["open"] = daily_val["open"]
            response_json["close"] = daily_val["close"]
            response_json["gain"] = daily_val["up"]
            decisions_internal.append(response_json)
            print_progress(
                i, total, start_time, iteration=iteration, iterations=iterations
            )
        except Exception as e:
            print("\n========= DEBUG ERROR =========")
            print("Exception:", type(e).__name__)
            print("Message:", e)
            print("\n--- RESPONSE.JSON() ---")
            try:
                print(response.json())
            except Exception:
                print("Could not parse response.json()")
            print("\n--- RAW MODEL OUTPUT ---")
            print(raw)
            print("\n--- PARSED JSON ---")
            print(response_json)
            print("================================\n")
            raise

    return decisions_internal


def run_experiment(
    model: str,
    iterations: int,
    treatment_facts: Optional[Sequence[str]] = None,
) -> List[List[Dict[str, Any]]]:
    """
    Run multiple independent sequential episodes ("runs") for a given model and treatment.

    Each run is generated by ask_llm_run() and corresponds to one full pass
    over the DAX days with the same prompt structure but stochastic model sampling
    (e.g., via temperature).

    Used in:
        - main: called once per (model, treatment) pair.

    Args:
        model: OpenRouter model identifier.
        iterations: Number of runs to execute.
        treatment_facts: Optional list of strings used as daily injected context.

    Returns:
        decision_data: List of runs, where each run is a list of daily decision dicts.
    """

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

    models = [
        "openai/gpt-5-mini",
        "anthropic/claude-opus-4.5",
        "google/gemini-3-flash-preview",
        "meta-llama/llama-4-maverick",
    ]

    iterations = 3

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
        plot_model_treatments_1x3(model, model_results)
