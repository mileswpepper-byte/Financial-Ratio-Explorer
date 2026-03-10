import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Metrics that should be interpreted as percentages
PERCENT_METRICS = [
    "Gross Margin",
    "Operating Margin",
    "Net Profit Margin",
    "Operating CF Margin",
    "Free Cash Flow Margin",
    "ROIC",
    "ROE",
    "ROA",
    "DuPont ROE",
    "Free Cash Flow Yield",
]


def _safe_series(index_like) -> pd.Series:
    """Return a NaN series for missing line items."""
    return pd.Series(index=index_like, dtype="float64")


def _get_row(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """
    Try several possible row labels and return the first match.
    If none exist, return a NaN series so ratios just show NaN instead of crashing.
    """
    for label in candidates:
        if label in df.index:
            return df.loc[label]
    return _safe_series(df.columns)


def _compute_ratios_numeric(ticker: str) -> tuple[pd.DataFrame, float]:
    stock = yf.Ticker(ticker)

    income_statement = stock.financials
    balance_sheet = stock.balance_sheet
    cf = stock.cash_flow

    # Make sure we actually got data back
    if income_statement.empty or balance_sheet.empty:
        raise ValueError("No financial data available for this ticker.")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    op_income = _get_row(income_statement, ["Operating Income", "OperatingIncome"])
    gross_profit = _get_row(income_statement, ["Gross Profit", "GrossProfit"])
    revenue = _get_row(income_statement, ["Total Revenue", "TotalRevenue"])
    net_income = _get_row(income_statement, ["Net Income", "NetIncome"])
    current_assets = _get_row(
        balance_sheet, ["Current Assets", "Total Current Assets"]
    )
    current_liabilities = _get_row(
        balance_sheet, ["Current Liabilities", "Total Current Liabilities"]
    )
    eff_tax_rate = _get_row(income_statement, ["Tax Rate For Calcs"])
    ebitda = _get_row(income_statement, ["EBITDA"])
    invested_capital = _get_row(balance_sheet, ["Invested Capital"])
    avg_invested_cap = invested_capital.replace(0, np.nan).mean()
    ebit = _get_row(income_statement, ["EBIT"])
    total_debt = _get_row(balance_sheet, ["Total Debt"])
    total_equity = _get_row(balance_sheet, ["Common Stock Equity", "Total Equity"])
    cogs = _get_row(income_statement, ["Cost Of Revenue", "Cost of Goods Sold"])
    inventory = _get_row(balance_sheet, ["Inventory"])
    receivables = _get_row(balance_sheet, ["Receivables", "Accounts Receivable"])
    payables = _get_row(balance_sheet, ["Payables", "Accounts Payable"])
    cash_cashequivalents = _get_row(
        balance_sheet, ["Cash And Cash Equivalents", "Cash And Cash Equivalents"]
    )
    total_assets = _get_row(balance_sheet, ["Total Assets"])
    operating_cash_flow = _get_row(
        cf,
        [
            "Total Cash From Operating Activities",
            "Operating Cash Flow",
            "Cash From Operating Activities",
        ],
    )
    capex = _get_row(cf, ["Capital Expenditure", "Capital Expenditures"])

    operating_margin = op_income / revenue
    gross_margin = gross_profit / revenue
    net_profit_margin = net_income / revenue
    current_ratio = current_assets / current_liabilities
    roic = (ebitda * (1 - eff_tax_rate)) / avg_invested_cap
    debt_to_ebitda = total_debt / ebitda
    debt_to_capital = total_debt / total_equity

    inventory_turnover = cogs / inventory
    receivables_turnover = revenue / receivables
    payables_turnover = cogs / payables

    days_inv_outstanding = 365 / inventory_turnover
    days_sales_oustanding = 365 / receivables_turnover
    days_payable_outstanding = 365 / payables_turnover
    ccc = days_inv_outstanding + days_sales_oustanding - days_payable_outstanding

    quick_ratio = (cash_cashequivalents + receivables) / current_liabilities
    roe = net_income / total_equity
    roa = net_income / total_assets
    asset_turnover = revenue / total_assets
    equity_multiplier = total_assets / total_equity
    debt_to_equity = total_debt / total_equity
    eps = income_statement.loc["Basic EPS"]
    capex_sales = capex / revenue

    # Cash-flow based metrics
    ocf_margin = operating_cash_flow / revenue
    free_cash_flow = operating_cash_flow + capex  # capex is usually negative
    fcf_margin = free_cash_flow / revenue

    # Valuation ratios based on latest price
    history = stock.history(period="1d")
    if history.empty:
        latest_price = np.nan
    else:
        latest_price = history["Close"][0]

    shares_outstanding = _get_row(balance_sheet, ["Share Issued"])
    market_cap = latest_price * shares_outstanding

    pe = market_cap / net_income
    ps = market_cap / revenue
    pb = market_cap / total_equity

    free_cash_flow_yield = free_cash_flow / market_cap

    evtoebitda = (market_cap + total_debt - cash_cashequivalents) / ebitda

    # DuPont-style ROE decomposition (should roughly equal ROE)
    dupont_roe = net_profit_margin * asset_turnover * equity_multiplier

    ratios_df = pd.DataFrame(
        {
            "Gross Margin": gross_margin,
            "Operating Margin": operating_margin,
            "Net Profit Margin": net_profit_margin,
            "Operating CF Margin": ocf_margin,
            "Free Cash Flow Margin": fcf_margin,
            "Current Ratio": current_ratio,
            "ROIC": roic,
            "Asset Turnover": asset_turnover,
            "Equity Multiplier": equity_multiplier,
            "Inventory Turnover": inventory_turnover,
            "Receivables Turnover": receivables_turnover,
            "Payables Turnover": payables_turnover,
            "Debt / EBITDA": debt_to_ebitda,
            "Debt / Capital": debt_to_capital,
            "Cash Conversion Cycle": ccc,
            "Quick Ratio": quick_ratio,
            "ROE": roe,
            "ROA": roa,
            "Debt / Equity": debt_to_equity,
            "Basic EPS": eps,
            "CapEx / Sales": capex_sales,
            "P / E": pe,
            "P / S": ps,
            "P / B": pb,
            "Free Cash Flow Yield": free_cash_flow_yield,
            "EV / EBITDA": evtoebitda,
            "DuPont ROE": dupont_roe,
        }
    )

    # Index is currently full period dates (e.g. 2024-12-31) – convert to just the year
    years = []
    for idx in ratios_df.index:
        try:
            years.append(idx.year)
        except AttributeError:
            years.append(idx)
    ratios_df.index = years

    # Drop 2021 explicitly if present, and any all-NaN rows
    if 2021 in ratios_df.index:
        ratios_df = ratios_df.drop(index=[2021])
    ratios_df = ratios_df.dropna(axis=0, how="all")

    # Keep only the most recent 4 years if more are available
    unique_years = sorted(y for y in ratios_df.index if isinstance(y, int))
    if len(unique_years) > 4:
        last_four = unique_years[-4:]
        ratios_df = ratios_df.loc[last_four]

    # Convert selected metrics from decimals to percentage values (e.g. 0.23 -> 23.0)
    for col in PERCENT_METRICS:
        if col in ratios_df.columns:
            ratios_df[col] = ratios_df[col] * 100

    return ratios_df, latest_price


def fetch_ratios(ticker: str) -> tuple[pd.DataFrame, float]:
    """
    Fetch ratios and format them as strings for display in the single-company view.
    """
    ratios_df, latest_price = _compute_ratios_numeric(ticker)

    # Present with years as columns and metrics as rows
    result = ratios_df.T

    # Add % sign to percentage-based rows for display, and round everything to 2 decimals
    percent_rows = PERCENT_METRICS
    for row in percent_rows:
        if row in result.index:
            result.loc[row] = result.loc[row].apply(
                lambda v: "" if pd.isna(v) else f"{v:.2f}%"
            )

    # For all other rows, format as numbers with two decimal places
    for row in result.index:
        if row in percent_rows:
            continue
        result.loc[row] = result.loc[row].apply(
            lambda v: "" if pd.isna(v) else f"{float(v):.2f}"
        )

    return result, latest_price


st.set_page_config(page_title="Financial Ratio Explorer", layout="wide")

# Make tables more legible with slightly larger font
st.markdown(
    """
    <style>
    div[data-testid="stDataFrame"] div[role="gridcell"],
    div[data-testid="stDataFrame"] div[role="columnheader"] {
        font-size: 1.1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Financial Ratio Explorer")

mode = st.radio(
    "Select view",
    ["Single company", "Company comparison"],
    horizontal=True,
)

if mode == "Single company":
    ticker = st.text_input("Enter a stock ticker (e.g. AAPL, MSFT):", value="ISRG")

    if st.button("Get ratios"):
        if not ticker:
            st.error("Please enter a ticker symbol.")
        else:
            try:
                symbol = ticker.strip().upper()
                ratios, latest_price = fetch_ratios(symbol)
            except Exception as e:
                st.error(f"Could not fetch ratios for {ticker}: {e}")
            else:
                st.success(f"Showing ratios for {symbol}")
                if not np.isnan(latest_price):
                    st.subheader(f"Current {symbol} Share Price: ${latest_price:,.2f}")
                st.dataframe(ratios, use_container_width=True, height=600)

else:
    col1, col2 = st.columns(2)
    with col1:
        ticker_a = st.text_input("First ticker", value="AAPL")
    with col2:
        ticker_b = st.text_input("Second ticker", value="MSFT")

    compare_clicked = st.button("Compare")

    if compare_clicked:
        if not ticker_a or not ticker_b:
            st.error("Please enter both tickers.")
        else:
            sym_a = ticker_a.strip().upper()
            sym_b = ticker_b.strip().upper()
            try:
                ratios_a, price_a = _compute_ratios_numeric(sym_a)
                ratios_b, price_b = _compute_ratios_numeric(sym_b)
            except Exception as e:
                st.error(f"Could not fetch data: {e}")
            else:
                common_years = sorted(
                    set(ratios_a.index).intersection(set(ratios_b.index))
                )
                if not common_years:
                    st.error("No overlapping years of data for these tickers.")
                else:
                    year = st.selectbox(
                        "Select year for comparison",
                        common_years,
                        index=len(common_years) - 1,
                    )

                    available_ratios = list(ratios_a.columns)
                    selected_ratios = st.multiselect(
                        "Ratios to compare",
                        available_ratios,
                        default=[r for r in ["ROE", "ROA", "ROIC"] if r in available_ratios],
                    )

                    if selected_ratios:
                        data = {}
                        for metric in selected_ratios:
                            val_a = ratios_a.loc[year, metric]
                            val_b = ratios_b.loc[year, metric]
                            data[metric] = {sym_a: val_a, sym_b: val_b}

                        comparison_df = pd.DataFrame(data).T

                        # Format percentages and numbers similarly to single-company view
                        for metric in comparison_df.index:
                            if metric in PERCENT_METRICS:
                                comparison_df.loc[metric] = comparison_df.loc[
                                    metric
                                ].apply(
                                    lambda v: ""
                                    if pd.isna(v)
                                    else f"{float(v):.2f}%"
                                )
                            else:
                                comparison_df.loc[metric] = comparison_df.loc[
                                    metric
                                ].apply(
                                    lambda v: ""
                                    if pd.isna(v)
                                    else f"{float(v):.2f}"
                                )

                        st.subheader(f"Comparison for {year}")
                        st.dataframe(
                            comparison_df,
                            use_container_width=True,
                        )
                    else:
                        st.info("Select at least one ratio to compare.")
