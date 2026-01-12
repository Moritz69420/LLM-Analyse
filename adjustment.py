
Dax funktion so anpassen, dass sie auch wirklich 30 Handelstage läd.
```python
def dax_data(trading_days=30):
    ticker = "^GDAXI"
    # großzügig mehr Kalendertage laden
    end_date = datetime.today()
    start_date = end_date - timedelta(days=int(trading_days * 1.7))

    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )

    # Index sicher als Datetime
    df.index = pd.to_datetime(df.index)

    # Nur die letzten N Handelstage
    df = df.tail(trading_days)

    open_series = df["Open"][ticker]
    close_series = df["Close"][ticker]

    dax_labeled = {
        d.strftime("%Y-%m-%d"): {
            "open": float(o),
            "close": float(c),
            "up": int(c > o),
        }
        for d, o, c in zip(df.index, open_series.values, close_series.values)
    }

    return list(dax_labeled.items())
    
    
    
    --------
    
text = “You are a rational investor making a daily decision whether to invest in a DAX index fund or sell your assets for cash.“ 
# Antwort Priming
version1 = „I think now is a good time to invest into the markte“
version2= „I think now is not a good time to invest into the markte“
# Zielfokus
version3 = “You are a rational investor who wants to make as much money as possible making a daily decision whether to invest in a DAX index fund or sell your assets for cash.“ 
# Bullish bias
version4 = “You are a rational bullish investor making a daily decision whether to invest in a DAX index fund or sell your assets for cash.“ 
version5 = “You are a rational bearish making a daily decision whether to invest in a DAX index fund or sell your assets for cash.“ 
# Overconfidence out of past investment
version6 = „You have recently made several successful investment decisions and trust your judgment strongly.“
# autority bias
version7 = „You are a renowned market expert whose opinions are widely respected and rarely questioned.“




Rules:
- You can only choose between "buy" (invest today) or "sell".
…
```
```python
base_prompts = [
    ("neutral", "You are a rational investor making a daily decision whether to invest in a DAX index fund or sell your assets for cash."),
    ("profit_max", "You are a rational investor who wants to make as much money as possible making a daily decision whether to invest in a DAX index fund or sell your assets for cash."),
    ("bullish", "You are a rational bullish investor making a daily decision whether to invest in a DAX index fund or sell your assets for cash."),
]

def build_user_message(
    date_str,
    open_price,
    prior_decisions,
    extra_info=None,
    base_prompt="You are a rational investor making a daily decision whether to invest in a DAX index fund or sell your assets for cash.",
):
    """
    base_prompt: the role/identity prompt you want to vary across runs.
    """
    extra_block = ""
    if extra_info is not None:
        extra_block = f"\n- Today's additional information: {extra_info}"

    return f"""
{base_prompt}

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



if __name__ == "__main__":
    models = ["mistralai/mixtral-8x7b-instruct"]
    iterations = 1

    treatments = [
        ("without information", None),
        ("with irrelevant information", irrelevant_facts),
        ("with relevant information", relevant_facts),
    ]

    base_prompts = [
        ("neutral", "You are a rational investor making a daily decision whether to invest in a DAX index fund or sell your assets for cash."),
        ("profit_max", "You are a rational investor who wants to make as much money as possible making a daily decision whether to invest in a DAX index fund or sell your assets for cash."),
        ("bullish", "You are a rational bullish investor making a daily decision whether to invest in a DAX index fund or sell your assets for cash."),
    ]

    all_results = {}

    for model in models:
        all_results[model] = {}

        for prompt_id, base_prompt in base_prompts:
            all_results[model][prompt_id] = {}

            for title, facts in treatments:
                print(f"\nRunning: {model} | prompt={prompt_id} | {title}")

                decision_data = run_experiment(
                    model=model,
                    iterations=iterations,
                    treatment_facts=facts,
                    base_prompt=base_prompt,   # <--- neu
                )
                all_results[model][prompt_id][title] = decision_data

    # Plotten: pro Modell + Prompt-ID ein 1x3 Plot über Treatments
    for model, model_results in all_results.items():
        for prompt_id, prompt_results in model_results.items():
            plot_model_treatments_1x3(
                model_name=f"{model} | {prompt_id}",
                results_by_treatment=prompt_results,
            )
