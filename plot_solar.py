import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


TOWNS = ["ALBURGH", "SOUTH HERO", "GLOVER"]

def run():
    fname = 'data/PP12 Jan2025.xlsx'
    df = pd.read_excel(fname)
    df = clean_pv_data(df)
    add_pending(df)
    print_total_nums(df)
    plot(df)

# Clean up PV spreadsheet
def clean_pv_data(df):
    df = df[df["Unit/Fuel"] == "Solar"]
    df["City/Town"] = df["City/Town"].str.upper().str.strip()
    convert = {"ALBURG": "ALBURGH"}
    for k, v in convert.items():
        select = df["City/Town"] == k
        df.loc[select, "City/Town"] = v

    return df

def add_pending(df):
    print(df["Status"].unique())
    print("n pending", (df["Status"] == "Pending").sum())
    pending = df["Status"] == "Pending"
    for i in df.index[pending]:
        df.loc[i, "In-Service Date"] = pd.Timestamp(2025, 6, 1)

def print_total_nums(df):
    metername = "data/VEC_meter_number_data.parquet"
    meterdf = pd.read_parquet(metername)

    for sub, name in zip([28, 29, 43], ["ALBURGH", "SOUTH HERO", "GLOVER"]):
        tmp = df[df["City/Town"] == name]
        n_pv = (tmp["Status"] != "Cancelled").sum()
        nmeter = (meterdf["Substation"] == str(sub)).sum()
        print(f"Substation {sub} {name:<10} {n_pv} / {nmeter}")

def plot(df):
    #df = df[df["City/Town"].isin(TOWNS)]
    df = df[~df["In-Service Date"].isna()]
    df = df.sort_values(by="In-Service Date")

    sns.histplot(
        data=df,
        #hue="City/Town",
        x="In-Service Date",
        binwidth=365,
        multiple="dodge",
        shrink=0.8,
        kde=True,
        kde_kws=dict(bw_adjust=0.5),
    )
    plt.savefig("figures/solar_growth_total.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    run()
