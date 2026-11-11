"""把 docs/ 下的 Markdown 文档编译成 PDF（xelatex + ctex），公式按 LaTeX 排版。

运行（在 quasibound/ 下）：python3 docs/md2pdf.py [docs/ccrm_formulas.md]
输出与 .md 同名的 .pdf，中间文件放在临时目录。需要 TeX Live 的 xelatex、ctex、unicode-math、xltabular、
enumitem；字体 STIX Two（正文、公式）与 Fandol（中文）都随 TeX Live 提供，代码块在有 Menlo 时用 Menlo。
只处理本目录文档用到的 Markdown 子集：标题、段落、**粗体**、*斜体*、`行内代码`、$…$ 与 $$…$$ 公式、
有序和无序列表（可嵌套，列表项内可含公式块）、表格、图片、代码块。
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PREAMBLE = r"""\documentclass[UTF8,a4paper,11pt,fontset=fandol]{ctexart}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath}
\usepackage{unicode-math}
\setmainfont{STIXTwoText}[Extension=.otf, UprightFont=*-Regular, BoldFont=*-SemiBold,
                          ItalicFont=*-Italic, BoldItalicFont=*-SemiBoldItalic]
\setmathfont{STIXTwoMath-Regular.otf}
\IfFontExistsTF{Menlo}{\setmonofont{Menlo}[Scale=MatchLowercase]}{}
\xeCJKDeclareCharClass{Default}{"2070 -> "209F, "2190 -> "22FF, "25A0 -> "25FF}  % 箭头、数学符号、上下标用 STIX，不用中文字体
\usepackage{graphicx, booktabs, xltabular, array, enumitem, newunicodechar, needspace, fvextra, float}
\newunicodechar{≤}{\ensuremath{\leq}}\newunicodechar{≥}{\ensuremath{\geq}}\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{≲}{\ensuremath{\lesssim}}\newunicodechar{≳}{\ensuremath{\gtrsim}}
\newfontfamily\stixmath{STIXTwoMath-Regular.otf}
\newunicodechar{→}{\ensuremath{\rightarrow}}\newunicodechar{∎}{{\stixmath\symbol{"220E}}}  % STIX Two Text 缺这几个字形
\usepackage[hidelinks]{hyperref}
\ctexset{section={format=\Large\bfseries\raggedright}, subsection={format=\large\bfseries\raggedright}}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\setlist{itemsep=1pt, topsep=3pt, leftmargin=*}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\sloppy
\begin{document}
"""
LIST = re.compile(r"^( *)([-*]|\d+\.) +(.*)$")                      # 列表项：缩进、标记、正文
SPAN = re.compile(r"`[^`]*`|(?<![\\$])\$(?!\$)[^$\n]+?\$")           # 行内代码或行内公式（原样保留）
ESC = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_",
       "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "$": r"\$"}       # 公式已先占位，剩下的 $ 都是字面量


def esc(t):
    return "".join(ESC.get(c, c) for c in t)


def inline(text):
    """行内转换：公式与代码先占位，其余转义后处理粗体、斜体和链接，再放回占位。"""
    keep = []

    def stash(m):
        s = m.group(0)
        keep.append(r"\texttt{" + esc(s[1:-1]) + "}" if s[0] == "`" else s)
        return f"\ue000{len(keep) - 1}\ue001"
    t = esc(SPAN.sub(stash, text))
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"(?<!\*)\*(?![\s*])(.+?)(?<![\s*])\*(?!\*)", r"\\emph{\1}", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)                  # 链接只留文字
    return re.sub("\ue000(\\d+)\ue001", lambda m: keep[int(m.group(1))], t)


def plain(text):
    """书签用的纯文本：去掉公式与代码标记。"""
    return esc(re.sub(r"\\[A-Za-z]+|[$`{}^_*]", "", text))


def width(cell):
    """估计单元格的显示宽度（中文按 2，公式按源码长度的 1/3）。"""
    t = re.sub(r"\$([^$]*)\$", lambda m: "x" * max(2, len(m.group(1)) // 3), cell)
    return sum(2 if ord(c) > 0x2e80 else 1 for c in t)


def table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, body = cells[0], cells[2:]                                  # cells[1] 为分隔行
    w = [max(width(r[k]) for r in cells[:1] + body) for k in range(len(head))]
    spec = ["L" if x > 30 else "l" for x in w]
    if "L" not in spec:
        spec[w.index(max(w))] = "L"
    out = [r"\begin{xltabular}{\linewidth}{" + "".join(spec) + "}", r"\toprule",
           " & ".join(inline(c) for c in head) + r" \\ \midrule", r"\endhead"]
    out += [" & ".join(inline(c) for c in r) + r" \\" for r in body]
    return "\n".join(out + [r"\bottomrule", r"\end{xltabular}"])


def starts_block(line):
    s = line.lstrip()
    return bool(LIST.match(line)) or s.startswith(("#", "|", "```", "![", "$$")) or s.strip() == "---"


def items(lines, i):
    """从第 i 行起读一个列表，返回 (下一行号, LaTeX)。缩进不少于正文列的行都属于当前项。"""
    m = LIST.match(lines[i])
    base, ordered = len(m.group(1)), m.group(2)[0].isdigit()
    env = "enumerate" if ordered else "itemize"
    out = [rf"\begin{{{env}}}" + (f"[start={m.group(2)[:-1]}]" if ordered and m.group(2) != "1." else "")]
    while i < len(lines) and (m := LIST.match(lines[i])) and len(m.group(1)) == base:
        col = len(lines[i]) - len(m.group(3))                         # 该项正文所在列
        body, i = [m.group(3)], i + 1
        while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) >= col):
            body.append(lines[i][col:])
            i += 1
        out.append(r"\item " + blocks(body))
        if i < len(lines) and not lines[i].strip():
            i += 1
    return i, "\n".join(out + [rf"\end{{{env}}}"])


def blocks(lines):
    out, i = [], 0
    while i < len(lines):
        s = lines[i].strip()
        if not s or s == "---":
            i += 1
        elif s.startswith("```"):                                     # 代码块
            j = next(k for k in range(i + 1, len(lines)) if lines[k].strip().startswith("```"))
            out.append("\\begin{Verbatim}[fontsize=\\small, breaklines]\n" + "\n".join(lines[i + 1:j]) + "\n\\end{Verbatim}")  # 过长的行折行
            i = j + 1
        elif s == "$$":                                               # 行间公式（\tag 在 equation* 中可用）
            j = next(k for k in range(i + 1, len(lines)) if lines[k].strip() == "$$")
            out.append("\\begin{equation*}\n" + "\n".join(x.strip() for x in lines[i + 1:j]) + "\n\\end{equation*}")
            i = j + 1
        elif s.startswith("#"):                                       # 标题：# 为文档标题，## 节，### 小节
            level, title = len(s) - len(s.lstrip("#")), s.lstrip("#").strip()
            if level == 1:
                out.append(r"\begin{center}{\LARGE\bfseries " + inline(title) + r"}\end{center}")
            else:
                kind = "section" if level == 2 else "subsection"              # needspace：标题不落在页底
                out.append(rf"\needspace{{8\baselineskip}}\{kind}*{{{inline(title)}}}"
                           rf"\addcontentsline{{toc}}{{{kind}}}{{{plain(title)}}}")
            i += 1
        elif s.startswith("|"):
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            out.append(table(lines[i:j]))
            i = j
        elif s.startswith("!["):
            path = re.match(r"!\[.*?\]\((.*?)\)", s).group(1)
            out.append(r"\begin{figure}[H]\centering\includegraphics[width=0.85\linewidth,height=0.55\textheight,"
                       r"keepaspectratio]{" + path + r"}\end{figure}")                  # 不浮动：插图留在原处，紧接其图注
            i += 1
        elif LIST.match(lines[i]):
            i, tex = items(lines, i)
            out.append(tex)
        else:                                                         # 段落：到空行或下一个块为止
            j = i + 1
            while j < len(lines) and lines[j].strip() and not starts_block(lines[j]):
                j += 1
            out.append(inline(" ".join(x.strip() for x in lines[i:j])))
            i = j
    return "\n\n".join(out)


def main(md):
    md = Path(md).resolve()
    tex = PREAMBLE + r"\graphicspath{{" + md.parent.as_posix() + "/}}\n" + blocks(md.read_text("utf-8").split("\n")) \
        + "\n\\end{document}\n"
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "doc.tex").write_text(tex, "utf-8")
        for _ in range(2):                                            # 两遍：xltabular 列宽与书签
            run = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"], cwd=tmp,
                                 capture_output=True, text=True)
            if run.returncode:
                log = (Path(tmp) / "doc.log").read_text("utf-8", errors="replace")
                sys.exit("xelatex 失败：\n" + "\n".join(log.splitlines()[-30:]))
        shutil.copy(Path(tmp) / "doc.pdf", md.with_suffix(".pdf"))
        log = (Path(tmp) / "doc.log").read_text("utf-8", errors="replace")
    missing = sorted(set(re.findall(r"Missing character: There is no (.)", log)))
    print(f"写出 {md.with_suffix('.pdf')}" + (f"；缺字形：{''.join(missing)}" if missing else ""))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("ccrm_formulas.md"))
