"""可直接執行的 pandas 資料檢查、合併、缺值與聚合範例。"""

import pandas as pd


def main() -> None:
    orders = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4],
            "customer_id": pd.Series(["001", "002", "001", "003"], dtype="string"),
            "event_time": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"], utc=True
            ),
            "amount": [120.0, None, 80.0, 200.0],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": pd.Series(["001", "002", "003"], dtype="string"),
            "segment": ["A", "B", "A"],
        }
    )

    assert orders["order_id"].is_unique
    assert customers["customer_id"].is_unique
    print("缺值比例:\n", orders.isna().mean())

    joined = orders.merge(customers, on="customer_id", how="left", validate="m:1", indicator=True)
    assert len(joined) == len(orders)
    assert joined["_merge"].eq("both").all()
    joined["amount_filled"] = joined["amount"].fillna(joined["amount"].median())

    by_segment = (
        joined.groupby("segment", dropna=False)
        .agg(orders=("order_id", "size"), revenue=("amount_filled", "sum"))
        .reset_index()
    )
    daily = joined.set_index("event_time").resample("D")["amount_filled"].sum()
    print("分組彙總:\n", by_segment)
    print("每日金額:\n", daily)
    assert by_segment["orders"].sum() == len(orders)


if __name__ == "__main__":
    main()
