import io, re
BS = chr(92)
s = io.open("main.tex", encoding="utf-8").read()
lines = s.split("\n")
tot = 0
for i, l in enumerate(lines):
    if l.startswith(BS + "label{fig:"):
        j = i - 1
        while not lines[j].lstrip().startswith(BS + "caption{"):
            j -= 1
        body = "\n".join(lines[j:i])
        t = re.sub(BS + BS + "[a-zA-Z]+", " ", body)
        for c in "{}$" + BS:
            t = t.replace(c, " ")
        n = len(t.split())
        tot += n
        print(l.strip(), n)
print("total", tot)
