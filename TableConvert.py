import pandas as pd
import os

def escape_typst(text):
    if not isinstance(text, str):
        text = str(text)
    # 转义特殊字符，防止语法断裂
    chars = [('\\', r'\\'), ('#', r'\#'), ('$', r'\$'), ('*', r'\*'), 
             ('_', r'\_'), ('`', r'\`'), ('<', r'\<'), ('>', r'\>'), ('@', r'\@')]
    for c, r in chars:
        text = text.replace(c, r)
    return text

def generate_final_typst(input_xlsx):
    # --- 局部配置区 ---
    FONT_EN = "Times New Roman"
    FONT_ZH = "Microsoft YaHei" # 微软雅黑
    FONT_SIZE = "9pt"
    HIGHLIGHT_ROWS = "(11,)" # 高亮行号
    # ----------------

    try:
        # 使用 openpyxl 读取
        df = pd.read_excel(input_xlsx, engine='openpyxl')
        df = df.dropna(how='all').dropna(axis=1, how='all').fillna("")
        
        col_count = len(df.columns)
        
        # 1. 预处理所有单元格内容
        all_cells = []
        # 处理表头
        for col in df.columns:
            name = str(col).strip()
            if "Unnamed:" in name: name = ""
            all_cells.append(f"    [* {escape_typst(name)} *]")
        # 处理行数据
        for _, row in df.iterrows():
            for cell in row:
                all_cells.append(f"    [{escape_typst(str(cell))}]")
        
        cells_content = ",\n".join(all_cells)

        # 2. 生成“内容块”模式的 Typst 代码
        # #[ ... ] 是内容块，内部的 #set 只对块内生效，且内容会自动渲染
        should_flip = col_count > 6
        
        typst_code = f"""
#[
  #set text(font: ("{FONT_EN}", "{FONT_ZH}"), size: {FONT_SIZE})
  #let tbl = table(
    columns: (auto,) * {col_count},
    stroke: 0.5pt + black,
    inset: 6pt,
    align: horizon + center,
    fill: (x, y) => {{
      if y == 0 {{ gray.lighten(80%) }}
      else if y in {HIGHLIGHT_ROWS} {{ blue.lighten(90%) }}
    }},
{cells_content}
  )

  {"#page(flipped: true, margin: 1cm)[#tbl]" if should_flip else "#tbl"}
]
"""
        # 保存文件
        with open("output_snippet.typ", "w", encoding="utf-8") as f:
            f.write(typst_code.strip())
            
        print("--- 转换完成！ ---")
        print("请打开 output_snippet.typ，复制全部内容并粘贴。")

    except Exception as e:
        print(f"读取 Excel 出错: {e}")

if __name__ == "__main__":
    generate_final_typst(r"E:\TypstNote\datasheet\Calculated-3-cis-hexenal.xlsx")