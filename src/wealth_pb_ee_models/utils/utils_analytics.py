## VERSION 4th DECEMBER 2025
## AUTHOR Luis Sigcha

## Load the required packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce


def minutes_summary(df, encoding_dict, ACTIVITY_ORDER, analytics="activity", plot=True):
    df = df.copy()
    # ---- time handling ----
    if pd.api.types.is_datetime64_any_dtype(df["Time"]):
        # already datetime, do nothing (or ensure tz-naive)
        df["Time"] = pd.to_datetime(df["Time"])
    else:
        # try numeric Excel serial days first
        if pd.api.types.is_numeric_dtype(df["Time"]):
            df["Time"] = pd.to_datetime(df["Time"], unit="d", origin="1899-12-30")
        else:
            # fallback: parse strings
            df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    
    # ---- time to data----
    df['date'] = df['Time'].dt.date
    # ---- aggregation ----
    window_size_in_minutes = 10 / 60
    minutes = (
        pd.crosstab(df['date'], df[analytics])
        .astype(float)
        * window_size_in_minutes)
    # ---- ensure ALL activities exist ----
    expected_cols = list(encoding_dict.values())
    minutes = minutes.reindex(columns=expected_cols, fill_value=0)
    # ---- optional plot ----
    if plot:
        ax = minutes.plot(kind="bar", stacked=True, figsize=(14, 6))
        ax.set_ylabel("Minutes")
        ax.set_title("Daily activity minutes (fixed sampling rate)")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.show()
    # ---- total analyzed minutes ----
    minutes["daily_count"] = minutes.sum(axis=1)
    # ---- tidy output ----
    minutes = minutes.reset_index()
    minutes.columns.name = None
    return minutes

def minutes_summary_CREA_AP( df, encoding_dict, ACTIVITY_ORDER, analytics = "activity", sampling_rate = 20, plot: bool = True):
    """    Compute daily activity minutes from a sample-wise dataframe
    sampled at a fixed rate (e.g., 20 Hz).
    Parameters
    ----------
    df : pd.DataFrame        Sample-wise dataframe.
    encoding_dict : dict        Mapping of activity codes to activity names.
    ACTIVITY_ORDER : list        Ordered list of activities.
    analytics : str        Column with activity labels.
    sampling_rate : int        Sampling rate in Hz (default = 20).
    plot : bool        Whether to produce a stacked bar plot.
    Returns
    -------
    pd.DataFrame        Daily minutes per activity + total daily_count.
    """
    df = df.copy()
    # ---- time handling ----
    df["Time"] = pd.to_datetime(df["Time"], unit="d", origin="1899-12-30")
    df["date"] = df["Time"].dt.date
    # ---- minutes per sample ----
    minutes_per_sample = 1 / (sampling_rate * 60)
    # ---- aggregation (sample-wise) ----
    minutes = (
        pd.crosstab(df["date"], df[analytics])
        .astype(float)
        * minutes_per_sample    )
    # ---- ensure all activities exist ----
    expected_cols = list(encoding_dict.values())
    minutes = minutes.reindex(columns=expected_cols, fill_value=0)
    # ---- enforce activity order ----
    minutes = minutes.reindex(columns=ACTIVITY_ORDER, fill_value=0)
    # ---- optional plot ----
    if plot:
        ax = minutes.plot(kind="bar", stacked=True, figsize=(14, 6))
        ax.set_ylabel("Minutes")
        ax.set_title("Daily activity minutes (20 Hz sampling)")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.show()
    # ---- total analyzed minutes ----
    minutes["daily_count"] = minutes.sum(axis=1)
    # ---- tidy output ----
    minutes = minutes.reset_index()
    minutes.columns.name = None
    return minutes


def filter_valid_days(df, wear_time_col= "daily_count", min_wear_hours= 8.0):
    """Filter valid days according to a minimum wear-time threshold.
    Parameters
    ----------
    df : Input dataframe (daily-level).
    wear_time_col : Column with wear time expressed in minutes.
    min_wear_hours : Minimum wear time in hours (default = 8h).

    Returns
    -------
    pd.DataFrame: Filtered dataframe containing valid days only.
    """
    min_wear_minutes = min_wear_hours * 60
    df = df.copy()
    df = df[df[wear_time_col] >= min_wear_minutes]
    return df


def filter_minimum_days(
    df,
    min_weekdays=2,
    min_weekend_days=1,
    weekdays_col="n_weekdays",
    weekend_col="n_weekend_days"):
    """
    Filter participants based on minimum number of observed weekdays and weekend days.

    Parameters
    ----------
    df : pd.DataFrame        Summary DataFrame containing day-count columns.
    min_weekdays : int, default=2        Minimum required number of weekday days.
    min_weekend_days : int, default=1        Minimum required number of weekend days.
    weekdays_col : str, default="n_weekdays"        Column name for weekday counts.
    weekend_col : str, default="n_weekend_days"        Column name for weekend day counts.

    Returns
    -------
    pd.DataFrame        Filtered DataFrame.
    """
    mask = ((df[weekdays_col] >= min_weekdays) & (df[weekend_col] >= min_weekend_days))

    return df.loc[mask].reset_index(drop=True)


def summarize_weekday_weekend(df, pb_cols,ee_cols, id_col= "ID", date_col = "date"):
    """Summarise weekday vs weekend activity per participant
    (mean ± SD), returning a single row per ID and including
    the number of valid weekdays and weekends.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Weekday vs weekend
    df["day_type"] = np.where( df[date_col].dt.weekday < 5, "Weekday", "Weekend")
    vars_to_sum = pb_cols + ee_cols

    # ---- statistics (mean ± SD) ----
    stats = ( df.groupby([id_col, "day_type"])[vars_to_sum].agg(["mean", "std"]) )

    # Pivot weekday/weekend to columns
    stats = stats.unstack("day_type")

    # Flatten column names -> e.g. Sitting_Weekday_mean
    stats.columns = [
        f"{var}_{day}_{stat}"
        for var, stat, day in stats.columns
    ]

    # ---- counts of valid days ----
    counts = (
        df
        .groupby([id_col, "day_type"])
        .size()
        .unstack("day_type", fill_value=0)
        .rename(columns={
            "Weekday": "n_Weekday",
            "Weekend": "n_Weekend"
        })
    )

    # ---- combine ----
    summary = stats.join(counts).reset_index()

    return summary


#to erase
def summarize_weekday_weekend_weighted_V1(df, pb_cols, ee_cols,
                                       id_col="ID", date_col="date",
                                       w_weekday=5/7, w_weekend=2/7):
    """    Summarise activity per participant using a weighted average
    of weekday and weekend statistics.
    Weighting:
        - Weekdays: 5/7
        - Weekends: 2/7
    Returns one row per ID with weighted mean and SD per variable.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Weekday vs weekend
    df["day_type"] = np.where(df[date_col].dt.weekday < 5, "Weekday", "Weekend")
    vars_to_sum = pb_cols + ee_cols
    # ---- statistics (mean & SD) ----
    stats = (
        df
        .groupby([id_col, "day_type"])[vars_to_sum]
        .agg(["mean", "std"])
        .unstack("day_type")
    )

    frames = []
    for var in vars_to_sum:
        mean_wd = stats[(var, "mean", "Weekday")]
        mean_we = stats[(var, "mean", "Weekend")]
        std_wd  = stats[(var, "std", "Weekday")]
        std_we  = stats[(var, "std", "Weekend")]

        weighted_mean = w_weekday * mean_wd + w_weekend * mean_we
        weighted_std  = w_weekday * std_wd  + w_weekend * std_we

        frames.append(
            pd.DataFrame({
                id_col: weighted_mean.index,
                f"{var}_mean": weighted_mean.values,
                f"{var}_std": weighted_std.values
            })
        )

    # ---- merge all variables ----
    summary = reduce(
        lambda left, right: pd.merge(left, right, on=id_col),
        frames )

    return summary

def summarize_weekday_weekend_weighted(
    df,
    pb_cols,
    ee_cols,
    id_col="ID",
    date_col="date",
    w_weekday=5/7,
    w_weekend=2/7,
    include_day_counts=False):
    """
    Summarise activity per participant using a weighted average
    of weekday and weekend statistics.

    Weighting:
        - Weekdays: 5/7
        - Weekends: 2/7

    Returns one row per ID with weighted mean and SD per variable.
    Optionally appends:
        - n_weekdays
        - n_weekend_days
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # ---- weekday vs weekend ----
    df["day_type"] = np.where(df[date_col].dt.weekday < 5, "Weekday", "Weekend")

    vars_to_sum = pb_cols + ee_cols

    # ---- statistics (mean & SD) ----
    stats = (
        df
        .groupby([id_col, "day_type"])[vars_to_sum]
        .agg(["mean", "std"])
        .unstack("day_type")    )

    frames = []
    for var in vars_to_sum:
        mean_wd = stats[(var, "mean", "Weekday")]
        mean_we = stats[(var, "mean", "Weekend")]
        std_wd  = stats[(var, "std", "Weekday")]
        std_we  = stats[(var, "std", "Weekend")]

        weighted_mean = w_weekday * mean_wd + w_weekend * mean_we
        weighted_std  = w_weekday * std_wd  + w_weekend * std_we

        frames.append(
            pd.DataFrame({
                id_col: weighted_mean.index,
                f"{var}_mean": weighted_mean.values,
                f"{var}_std": weighted_std.values
            })
        )

    # ---- merge all variables ----
    summary = reduce(
        lambda left, right: pd.merge(left, right, on=id_col),
        frames
    )

    # ---- optional: count weekday / weekend days ----
    if include_day_counts:
        day_counts = (
            df
            .drop_duplicates([id_col, date_col])
            .groupby([id_col, "day_type"])[date_col]
            .count()
            .unstack("day_type")
            .fillna(0)
            .rename(columns={
                "Weekday": "n_weekdays",
                "Weekend": "n_weekend_days"
            })
            .reset_index()
        )

        summary = summary.merge(day_counts, on=id_col, how="left")

    return summary

def plot_multiple_weighted_summaries(
    summaries, variables,
    labels=None,
    ylabel="Minutes / day",
    title=None,
    capsize=5,
    error="ci",        # "ci" (default) or "sd"
    ci_level=0.95,
    show_table=False,
    table_round=2,
    axis_text_size=14
    ):
    """
    Plot grouped bar charts with uncertainty ranges from multiple weighted summaries,
    and optionally show a separate table with mean and SD/CI for each variable.

    Parameters
    ----------
    summaries : dict
        Dictionary of {name: summary_df}, where summary_df is the output of summarize_weekday_weekend_weighted.
    variables : list of str
        Variable base names (without _mean / _std).
    labels : list of str, optional
        Custom legend labels (same order as summaries.keys()).
        If None, dictionary keys are used.
    ylabel : str
        Y-axis label.
    title : str, optional
        Plot title.
    
    : int
        Error bar cap size.
    error : {"ci", "sd"}
        Type of uncertainty to plot:
        - "ci": confidence intervals (default)
        - "sd": standard deviation
    ci_level : float
        Confidence level used when error="ci" (default = 0.95).
    show_table : bool
        If True, displays a separate table with mean and SD/CI.
    table_round : int
        Rounding for table numeric columns.

    Returns
    -------
    table_df : pandas.DataFrame or None
        The summary table (returned if show_table=True), else None.
    """
    import scipy.stats as stats

    n_groups = len(variables)
    n_summaries = len(summaries)

    if labels is None:
        labels = list(summaries.keys())

    x = np.arange(n_groups)
    width = 0.8 / n_summaries
    plt.figure(figsize=(12, 6))
    z = stats.norm.ppf(1 - (1 - ci_level) / 2)

    # Collect rows for the output table
    table_rows = []

    for i, (label, df) in enumerate(zip(labels, summaries.values())):
        means = []
        errors = []

        for v in variables:
            col = f"{v}_mean"
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in summary DataFrame for '{label}'.")
            values = df[col].dropna().to_numpy()
            n = len(values)
            mean_val = float(np.mean(values)) if n > 0 else np.nan
            means.append(mean_val)

            if n <= 1:
                sd = np.nan
            else:
                sd = float(np.std(values, ddof=1))

            if error == "sd":
                err = sd
                # Table formatting fields
                ci_low = np.nan
                ci_high = np.nan
                uncertainty_label = "SD"
                uncertainty_value = err

            elif error == "ci":
                if n <= 1 or np.isnan(sd):
                    err = np.nan
                    ci_low = np.nan
                    ci_high = np.nan
                else:
                    err = float(z * sd / np.sqrt(n))
                    ci_low = mean_val - err
                    ci_high = mean_val + err
                uncertainty_label = f"CI{int(ci_level*100)}%"
                uncertainty_value = err

            else:
                raise ValueError("error must be 'ci' or 'sd'")
            errors.append(err)

            # Add a row to the table
            row = {"Group": label, "Variable": v, "n": n, "Mean": mean_val, "SD": sd, }

            if error == "ci":
                row.update({
                    "CI_level": ci_level,
                    "CI_halfwidth": uncertainty_value,
                    "CI_low": ci_low,
                    "CI_high": ci_high,
                })
            else:
                row.update({
                    "Uncertainty": uncertainty_label,
                    "Uncertainty_value": uncertainty_value,
                })

            table_rows.append(row)

        plt.bar(
            x + i * width - (n_summaries - 1) * width / 2,
            means,
            width,
            yerr=errors,
            capsize=capsize,
            label=label )

    plt.xticks(x, variables, rotation=40, ha="right", fontsize=axis_text_size)
    plt.yticks(fontsize=axis_text_size)
    
    plt.ylabel(ylabel, fontsize=15)

    if title is not None:
        plt.title(title)

    plt.legend()
    plt.tight_layout()
    plt.show()

    if not show_table:
        return None

    table_df = pd.DataFrame(table_rows)

    # Nicely formatted combined column for quick inspection
    if error == "ci":
        # e.g., "12.34 (10.12, 14.56)"
        table_df["Mean (CI)"] = table_df.apply(
            lambda r: (
                f"{r['Mean']:.{table_round}f} "
                f"({r['CI_low']:.{table_round}f}, {r['CI_high']:.{table_round}f})"
                if pd.notna(r["Mean"]) and pd.notna(r["CI_low"]) and pd.notna(r["CI_high"])
                else ""
            ),
            axis=1
        )
    else:
        # e.g., "12.34 ± 1.23"
        table_df["Mean ± SD"] = table_df.apply(
            lambda r: (
                f"{r['Mean']:.{table_round}f} ± {r['SD']:.{table_round}f}"
                if pd.notna(r["Mean"]) and pd.notna(r["SD"])
                else ""
            ),
            axis=1
        )
    # Round numeric columns for readability
    numeric_cols = table_df.select_dtypes(include=[np.number]).columns
    table_df[numeric_cols] = table_df[numeric_cols].round(table_round)
    # Display a separate table (in notebooks, this renders as a table)
    display(table_df)

    return table_df


#UTILS TO PLOT WEEKEND AND WEEKDAY SPLIT

def prepare_plot_data(summary_df,activities,id_col = "ID"):
    """ Prepare mean ± SD data for plotting from a single-row-per-subject
    weekday/weekend summary.

    Parameters
    ----------
    summary_df : pd.DataFrame        Output of summarize_weekday_weekend() (one row per ID).
    activities : list        Activity variables to plot.
    id_col : str        Participant identifier.

    Returns
    -------
    pd.DataFrame        Long-format dataframe aggregated across participants,suitable for bar plotting.
    """
    records = []
    for _, row in summary_df.iterrows():
        for act in activities:
            for day in ["Weekday", "Weekend"]:

                mean_col = f"{act}_{day}_mean"

                # Skip if participant lacks this day type
                if mean_col not in row or pd.isna(row[mean_col]):
                    continue

                records.append({
                    "ID": row[id_col],
                    "day_type": day,
                    "activity": act,
                    "mean": row[mean_col]
                })

    plot_df = pd.DataFrame(records)

    # Aggregate across participants (between-subject statistics)
    plot_df = (
        plot_df
        .groupby(["day_type", "activity"])
        .agg(
            mean=("mean", "mean"),
            std=("mean", "std")
        )
        .reset_index()
    )

    return plot_df


def plot_activity_bars(plot_df, activity_order, title="Plot activity", ylabel: str = "Minutes / day"):
    """    Bar plot of activities with mean ± SD intervals,
    plotted in a user-defined order.
    
    Parameters
    ----------
    plot_df : pd.DataFrame        Output of prepare_plot_data().
    activity_order : list        Ordered list of activity names (e.g., EE_COLS).
    title : str        Plot title.
    ylabel : str        Y-axis label.
    """
    # Enforce activity order and drop missing activities
    plot_df = plot_df[plot_df["activity"].isin(activity_order)].copy()
    plot_df["activity"] = pd.Categorical(plot_df["activity"],categories=activity_order, ordered=True )
    plot_df = plot_df.sort_values("activity")
    day_types = ["Weekday", "Weekend"]
    activities = activity_order
    x = range(len(activities))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, day in enumerate(day_types):
        subset = plot_df[plot_df["day_type"] == day].sort_values("activity")
        ax.bar(
            [p + i * width for p in x],
            subset["mean"].values,
            width=width,
            yerr=subset["std"].values,
            capsize=5,
            label=day )

    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(activities, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()
