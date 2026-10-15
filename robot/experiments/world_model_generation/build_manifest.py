import json, os, re, shutil, struct
SRC="/PHShome/zy992/Wilson/generation input"
OUT="/scratch/z/zy992/worldgen/inputs"
def nat(s): return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
def png_size(p):
    with open(p,"rb") as f: h=f.read(24)
    return struct.unpack(">II", h[16:24])
items=[]
def add(ds, name, frame, prompt, note=""):
    d=os.path.join(OUT, ds, name); os.makedirs(d, exist_ok=True)
    shutil.copyfile(frame, os.path.join(d,"first_frame.png")); shutil.copyfile(prompt, os.path.join(d,"prompt.txt"))
    w,h=png_size(frame); txt=open(prompt).read().strip()
    items.append(dict(id=ds+"__"+name, dataset=ds, name=name, first_frame=os.path.join(d,"first_frame.png"), prompt=txt, width=w, height=h, source_frame=frame, source_prompt=prompt, note=note))
# TAVP: matched frame+prompt pair; first_substep when present
t=os.path.join(SRC,"TAVP")
for n in sorted([x for x in os.listdir(t) if re.match(r"^\d{3}_",x)], key=nat)[:15]:
    g=os.path.join(t,n,"generation input"); sub=os.path.join(g,"first_substep")
    b=sub if os.path.isdir(sub) else g
    add("TAVP", n, os.path.join(b,"first_frame.png"), os.path.join(b,"prompt.txt"), "first_substep" if b==sub else "")
t=os.path.join(SRC,"Humanoid_Everyday")
for n in sorted([x for x in os.listdir(t) if "__" in x], key=nat)[:15]:
    g=os.path.join(t,n,"third_person","generation input")
    add("Humanoid_Everyday", n, os.path.join(g,"first_frame.png"), os.path.join(g,"actiongeneration_prompt.txt"), "third_person")
t=os.path.join(SRC,"NVIDIA_GR1_100")
for n in sorted([x for x in os.listdir(t) if x.isdigit()], key=nat)[:15]:
    g=os.path.join(t,n,"generation input")
    add("NVIDIA_GR1_100", n, os.path.join(g,"first_frame.png"), os.path.join(g,"actiongeneration_prompt.txt"))
json.dump(items, open(os.path.join(OUT,"manifest.json"),"w"), indent=1)
print(len(items))
from collections import Counter
print(Counter((i["dataset"],i["width"],i["height"]) for i in items))
print(max(len(i["prompt"]) for i in items), "max prompt chars")
for i in items[::7]: print(i["id"], i["note"], "|", i["prompt"][:110].replace(chr(10)," "))
