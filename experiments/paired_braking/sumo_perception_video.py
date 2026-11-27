from pathlib import Path
import json
import math
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import traci
import imageio_ffmpeg
from PIL import Image, ImageDraw
from cards import intro, results, text

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D.parents[1] / "bc"))
from study import Visual


trace = json.loads((D / "trace.json").read_text())
example = json.loads((D / "summary.json").read_text())["example"]
event = example["t"]
indices = range(event - 20, event + 16)
work = D / "runs" / "gui_perception"
work.mkdir(parents=True, exist_ok=True)
scheme = ET.parse(D / "seongsan.gui.xml")
scheme.getroot().find("delay").set("value", "0")
scheme.getroot().find("scheme/background").set("backgroundColor", "209,222,201")
scheme.write(work / "gui.xml")
start = traci.start
def start_gui(cmd, **kwargs):
    return start([*cmd, "--gui-settings-file", str(work / "gui.xml"), "--window-size", "1280,800", "--quit-on-end"], **kwargs)
traci.start = start_gui

def capture(j):
    env = Visual(work, example["seed"], 4, 0)
    env.gui = True
    try:
        env.reset(seed=example["seed"])
        gui = env.sim.gui
        view = gui.getIDList()[0]

        for i in range(1, indices.stop):
            r = trace[i]
            if i in indices:
                if i == indices.start: gui.setSchema(view, "seongsan")
                seen = set(r["seen"][j])
                for k in env.v.getIDList():
                    color = (241,66,67) if k == "ego" else (32,208,225) if k in seen else (148,151,157)
                    env.v.setColor(k, (*color,255))
                x,y = r["ego"]["x_m"], r["ego"]["y_m"] - 8
                gui.setBoundary(view,x-36,y-28,x+36,y+28)
                time.sleep(.06)
                gui.screenshot(view, str(work / f"{j}_{i}.png"),1600,900)
            if i in indices: time.sleep(.06)
            env.step([trace[i-1]["actions"][0]])
            if i in indices:
                for k,v in r["cars"].items():
                    xx,yy = env.v.getPosition(k)
                    assert math.hypot(xx-v["x_m"],yy-v["y_m"]) < .001
        time.sleep(.4)
    finally:
        env.close()

def frame(i):
    r = trace[i]
    im = Image.new("RGB",(1280,720),"#102332")
    d = ImageDraw.Draw(im)
    text(d,(28,20),"같은 주행 장면에서 AV가 인지한 차량",34,"white",True)
    text(d,(29,72),f"성산 SUMO-GUI   /   주행 {i*.2:.1f}초   /   자차 {r['ego']['speed_mps']*3.6:.1f} km/h",21,"#c1d3d9")
    for x,j,label in [(24,0,"맑음"),(648,2,"짙은 안개")]:
        text(d,(x,118),label,26,"white",True)
        act=["제동  -4 m/s²","유지  0 m/s²","가속  +3 m/s²"][r["actions"][j]//3]
        text(d,(x+385,120),act,23,"#ffb688",True)
        with Image.open(work / f"{j}_{i}.png") as shot:
            im.paste(shot.convert("RGB").crop((152,0,1448,900)).resize((608,422),Image.Resampling.LANCZOS),(x,163))
        seen = r["lead"] in r["seen"][j]
        text(d,(x,597),"같은 차로 선행차   "+("인지" if seen else "미인지"),23,"#96e6e5" if seen else "#ffc19c",True)
    text(d,(25,644),"빨강  자차 AV       청록  인지 차량       회색  미인지 차량",21,"white")
    note = "앞차 거리와 속도를 복원하면 짙은 안개에서도 제동 -4 m/s²" if i==event else "실제 주행은 맑음 행동을 적용 · 안개 쪽은 같은 시점에서 조회한 모델 출력"
    text(d,(25,683),note,19,"#cedfe4")
    return im

if __name__ == "__main__":
    if "--encode-only" not in sys.argv:
        for j in (0,2):
            capture(j)
    dest = D / "sumo_gui_perception.mp4"
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),"-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","1280x720","-r","10","-i","-","-an","-c:v","libx264","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",str(dest)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    def emit(im,n):
        raw=im.tobytes()
        for _ in range(n):
            proc.stdin.write(raw)
    emit(intro(),45)
    for i in indices:
        shot=frame(i)
        if i==event:
            shot.save(D/"sumo_gui_perception_keyframe.png")
        emit(shot,70 if i==event else 4)
    emit(results(),80)
    proc.stdin.close()
    assert proc.wait()==0
