import io, re, sys
BS = chr(92)
s = io.open("main.tex", encoding="utf-8").read()
start = s.index(BS + "section{" + sys.argv[1])
end = s.index(BS + "section{" + sys.argv[2])
body = s[start:end]
for para in body.split("\n\n"):
    lines = [l for l in para.split("\n") if not l.lstrip().startswith("%")]
    t = "\n".join(lines)
    t = re.sub(BS + BS + "[a-zA-Z]+", " ", t)
    for c in "{}$&~" + BS:
        t = t.replace(c, " ")
    w = t.split()
    if not w:
        continue
    print(len(w), "|", " ".join(w[:12]))
