import pandas as pd


def convert_dfs_to_excel(
    dfs: list,
    excelFileName: str,
    nameListForSheet: list,
    freezePanes: tuple = (0, 0),
    addLength: int = 5,
    _index: bool = False,
):
    with pd.ExcelWriter(excelFileName, engine="xlsxwriter") as writer:
        for name, df in zip(nameListForSheet, dfs):
            df.to_excel(writer, sheet_name=name, index=_index, freeze_panes=freezePanes)
            worksheet = writer.sheets[name]

            for idx, col in enumerate(df):
                series = df[col]
                max_len = (
                    max(series.astype(str).map(len).max(), len(str(series.name)))
                    + addLength
                )
                worksheet.set_column(idx, idx, max_len)
