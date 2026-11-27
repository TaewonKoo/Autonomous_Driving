from pathlib import Path
import json
from functools import lru_cache
from PIL import Image,ImageDraw,ImageFont

D=Path(__file__).resolve().parent
S=json.loads((D/'summary.json').read_text());ex=S['example'];event=ex['t']
W,H=1280,720
bg='#F5F6F3';ink='#132F38';muted='#46616A';teal='#087F7C';orange='#BB502D';line='#CCD9D7';white='#FFFFFF'
@lru_cache(None)
def font(size,bold):
 return ImageFont.truetype('C:/Windows/Fonts/'+('malgunbd.ttf' if bold else 'malgun.ttf'),size)
def text(d,xy,s,size=23,color=ink,bold=False):
 d.text(xy,s,font=font(size,bold),fill=color,anchor='lt')
def canvas():
 im=Image.new('RGB',(W,H),bg);return im,ImageDraw.Draw(im)
def head(d,tag,title):
 text(d,(45,27),tag,18,teal,True);text(d,(45,60),title,40,ink,True)
def footer(d,s):
 d.line((45,656,1235,656),fill=line,width=2);text(d,(45,673),s,18,muted)
def results():
 im,d=canvas();head(d,'성산  |  동일 장면 관측 비교','안개는 같은 순간의 제동 판단을 얼마나 바꿨나?')
 text(d,(46,125),'맑음에서 제동한 7,086개 시점 중, 안개 관측에서는 제동하지 않은 비율',23)
 for y,label,key,color in [(211,'옅은 안개','light',teal),(316,'짙은 안개','dense',orange)]:
  g=S[key];x=220;scale=15.6
  text(d,(48,y+4),label,29,ink,True)
  d.rounded_rectangle((x,y,x+40*scale,y+45),radius=8,fill='#E1E8E6')
  d.rounded_rectangle((x,y,x+100*g['drop_rate']*scale,y+45),radius=8,fill=color)
  text(d,(882,y-3),f"{100*g['drop_rate']:.1f}%",40,color,True)
  text(d,(1040,y+10),f"{g['drop']:,}개 시점",21,muted)
 d.rounded_rectangle((46,438,1233,624),radius=14,fill=ink)
 text(d,(75,461),'선행차가 누락된 짙은 안개 장면',23,'#9EDBD4',True)
 text(d,(75,507),'691개 중 528개에서 제동 복귀',40,white,True)
 text(d,(76,571),'안개 입력은 그대로 두고, 선행차 간격과 속도 두 값만 복원한 결과',22,'#E0EEED')
 footer(d,'100개 주행 · 94,459개 시점 · 정책과 교통 장면 고정 · 성산 BC의 행동 출력 비교')
 return im
def intro():
 im,d=canvas();head(d,'5tonomous   /   성산 저시정 실험','앞차를 놓치면 제동 판단도 달라질까?')
 d.rounded_rectangle((46,177,1233,530),radius=18,fill=ink)
 for y,n,label,detail in [(220,'01','같은 순간을 세 번 관측','맑음 · 옅은 안개 · 짙은 안개'),(324,'02','같은 BC에 각각 입력','세 조건 모두 같은 차량 위치와 속도'),(428,'03','앞차 정보만 복원','짙은 안개에서 제동 명령이 돌아오는지 확인')]:
  text(d,(78,y),n,27,'#86D7CB',True);text(d,(154,y),label,31,white,True);text(d,(636,y+3),detail,21,'#E1EDEC')
 text(d,(48,576),'실제 SUMO 궤적 재생  ·  차량 위치와 행동 명령을 수집 기록에서 재구성',22)
 footer(d,'차량은 맑음 정책으로 주행하며, 안개 행동은 같은 순간에 조회한다.')
 return im
