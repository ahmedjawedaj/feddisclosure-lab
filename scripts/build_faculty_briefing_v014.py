"""Build interim faculty-facing PPTX from verified research results, not projections."""
from pathlib import Path
from pptx import Presentation
import sys
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'reports/faculty-briefing-v0.14.0.pptx'
BG=RGBColor(13,25,40)
PANEL=RGBColor(25,44,62)
TEAL=RGBColor(90,211,194)
WHITE=RGBColor(235,244,252)
MUTED=RGBColor(164,188,207)
AMBER=RGBColor(240,194,112)

p=Presentation()
p.slide_width=Inches(13.333)
p.slide_height=Inches(7.5)
BLANK=p.slide_layouts[6]

def shape(s,x,y,w,h,fill=PANEL,radius=True):
    obj=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x),Inches(y),Inches(w),Inches(h))
    obj.fill.solid();obj.fill.fore_color.rgb=fill
    obj.line.fill.background()
    return obj

def text(s,txt,x,y,w,h,size=20,color=WHITE,bold=False,align=None):
    tb=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=tb.text_frame;tf.clear();tf.word_wrap=True
    tf.margin_left=Inches(0.04);tf.margin_top=Inches(0.02);tf.margin_right=Inches(0.04)
    par=tf.paragraphs[0];par.text=txt;par.space_after=Pt(0)
    if align: par.alignment=align
    for run in par.runs:
        run.font.size=Pt(size);run.font.name='Aptos';run.font.bold=bold;run.font.color.rgb=color
    return tb

def slide(title,subtitle=None):
    s=p.slides.add_slide(BLANK);s.background.fill.solid();s.background.fill.fore_color.rgb=BG
    shape(s,.55,.42,.075,.64,TEAL,False)
    text(s,title,.8,.37,11.9,.65,30,WHITE,True)
    if subtitle:text(s,subtitle,.82,1.10,11.6,.48,14,MUTED)
    num=len(p.slides)
    text(s,'FedDisclosure Lab  •  EXECUTED EVIDENCE  •  NOV 2026',.82,7.1,9,.18,9,MUTED)
    text(s,str(num),12.1,7.04,.5,.25,10,MUTED,align=PP_ALIGN.RIGHT)
    return s

def bullets(s,items,x=.9,y=1.7,w=11.5,step=.84,size=21):
    for i,item in enumerate(items):
        shape(s,x,y+i*step,.10,.10,TEAL,False)
        text(s,item,x+.25,y-.12+i*step,w,step-.04,size)

def card(s,title,value,detail,x,y,w=3.6):
    shape(s,x,y,w,1.7)
    text(s,title,x+.22,y+.18,w-.44,.33,13,MUTED)
    text(s,value,x+.22,y+.57,w-.44,.57,29,TEAL,True)
    text(s,detail,x+.22,y+1.26,w-.44,.31,12,MUTED)

def image(s,path,x,y,w,h):
    from PIL import Image

sys.path.insert(0,str(ROOT))
from dashboard.evidence import make_snapshot
V=make_snapshot()['v014']
if V is None:
    raise SystemExit('v0.14 evidence missing: refusing to build the briefing from projections')

def row(summary,partition,mode,eps=None):
    for r in summary:
        if r['partition']==partition and r['mode']==mode and (eps is None or r['target_epsilon']==eps):
            return r
    raise KeyError((partition,mode,eps))

# 1
s=p.slides.add_slide(BLANK);s.background.fill.solid();s.background.fill.fore_color.rgb=BG
shape(s,.75,1.35,.1,3.25,TEAL,False)
text(s,'FEDDISCLOSURE LAB',1.05,1.38,11.2,1.12,44,WHITE,True)
text(s,'Executed defenses: what held up and what did not',1.08,2.62,10.8,.85,24,TEAL)
text(s,'Real Docker + Flower training, OpenDP, Opacus and SecAgg+ evidence',1.08,4.04,10.8,.7,20,MUTED)
text(s,'Faculty briefing v0.14  |  21 November 2026  |  Independent research prototype',1.08,6.62,10.9,.45,14,MUTED)

# 2
s=slide('Question and what changed since v0.10','Earlier results were attacks on synthetic runs. Now the defenses actually ran.')
bullets(s,['Federated training ran in 8 real Docker containers on Flower 1.27 Deployment Runtime',
           'Three institutions each mounted their own dataset read-only',
           'OpenDP, Opacus and Flower SecAgg+ ran as real packages, not mock noise',
           'Formal guarantees and empirical attack results are reported separately'],size=21,step=1.0)

# 3
s=slide('Native Docker and Flower execution','Verified by receipts, parity against an independent PyTorch run, and docker inspect.')
card(s,'CONTAINERS','8','SuperLink, ServerApp, 3 SuperNodes, 3 ClientApps',.8,1.78)
card(s,'FULL MNIST','60,000','10 rounds, 3 clients, parity within 1e-5',4.85,1.78)
card(s,'GLOBAL ACCURACY','95.7%','Worst client 95.0%, 238 s',8.9,1.78)
bullets(s,['Three fault scenarios were correctly refused by the gate',
           'Fault handling waits for a timeout and is not fail-fast',
           'Hosted CI has not run because no authorised remote exists'],x=.95,y=4.18,size=18,step=.65)

# 4
sa=V['secagg']
s=slide('Flower SecAgg+ works but has clear limits','Upstream SecAggPlusWorkflow in Docker, 3 clients, threshold 2, matched plaintext control.')
card(s,'VS PLAINTEXT','%.1e'%sa['plaintext_diff'],'Max weight difference, within derived bound',.8,1.72)
card(s,'SERVER ONLY','%.1e'%sa['victim_error_server_only'],'Victim reconstruction error',4.85,1.72)
card(s,'SERVER + N-1','%.1e'%sa['victim_error_n_minus_1'],'Victim update recovered',8.9,1.72)
bullets(s,['Aggregate-only protection is void if the server colludes with n-1 clients',
           'With 3 clients and threshold 2, one dropout halted the protocol and no model was written',
           'The n of 4 or more explanation comes from code reading and is not yet tested'],x=.95,y=4.18,size=18,step=.7)

# 5
s=slide('OpenDP releases with a real budget ledger','Laplace counts, pure epsilon-DP, persistent accounting.')
bullets(s,['Four releases at epsilon 0.25 used the full budget and the fifth was refused with HTTP 409',
           'Retries return the cached release at no extra cost, ID reuse and cap increases are rejected',
           'Measured error matches Laplace theory, distinguishing advantage stayed under the DP bound',
           'At epsilon 0.05 the typical error is about 20, so tiny counts become unusable'],size=19,step=.95)

# 6
d=lambda part,mode,eps=None:row(V['digits_dp'],part,mode,eps)
s=slide('Opacus DP-FedAvg on Digits, 10 seeds','Achieved epsilon never exceeded the target. delta 1e-5. Accuracy is the price.')
card(s,'IID, NO DP','%.1f%%'%(100*d('iid','nonprivate')['test_accuracy']),'Test accuracy',.8,1.72)
card(s,'IID, EPS 4','%.1f%%'%(100*d('iid','opacus_dp',4.0)['test_accuracy']),'Achieved epsilon 3.99',4.85,1.72)
card(s,'LABEL SKEW, EPS 1','%.1f%%'%(100*d('label_skew','opacus_dp',1.0)['test_accuracy']),'Worst client %.0f%%'%(100*d('label_skew','opacus_dp',1.0)['worst_client_accuracy']),8.9,1.72)
bullets(s,['DP hurts the worst client far more than the average under label skew',
           'Membership AUC was about 0.5 with and without DP, so the attack shows no benefit to measure',
           'This is a negative result and says nothing about stronger attacks'],x=.95,y=4.18,size=18,step=.7)

# 7
m=lambda part,mode,eps=None:row(V['mnist_dp'],part,mode,eps)
s=slide('Full MNIST DP-FedAvg, 3 seeds','Small seed count, so intervals are wide. Treat as indicative.')
card(s,'IID, NO DP','%.1f%%'%(100*m('iid','nonprivate')['test_accuracy']),'Test accuracy',.8,1.72)
card(s,'IID, EPS 2','%.1f%%'%(100*m('iid','opacus_dp',2.0)['test_accuracy']),'Achieved epsilon 1.99',4.85,1.72)
card(s,'SKEW, EPS 2','%.1f%%'%(100*m('label_skew','opacus_dp',2.0)['test_accuracy']),'Worst client %.0f%%'%(100*m('label_skew','opacus_dp',2.0)['worst_client_accuracy']),8.9,1.72)
bullets(s,['DP simulation took about 9 times longer than non-private training on IID',
           'Communication volume is identical across modes',
           'Epsilon 2 and 8 are hard to tell apart at 10 rounds and 1 local epoch'],x=.95,y=4.18,size=18,step=.7)

# 8
s=slide('What this does and does not show','Keeping guarantees separate from attack success.')
bullets(s,['Epsilon and delta are properties of the mechanism, reported as achieved values',
           'Attack AUC and reconstruction error apply only to the tested attacker',
           'SecAgg+ is not differential privacy and does not bound what the final model leaks',
           'Nothing here is a privacy certification or a novelty claim'],size=20,step=.98)

# 9
s=slide('Still open','Hard gates for the next sprint.')
bullets(s,['Hosted CI on an authorised repository',
           'Fail-fast handling when a client disappears',
           'SecAgg+ with 4 or more clients to test dropout tolerance',
           'Native non-IID MNIST and multi-seed native runs',
           'Stronger attacks on protected and unprotected models, MNIST licence text check'],size=20,step=.9)

# 10
s=slide('Feedback requested','Which direction is thesis-worthy?')
bullets(s,['Is dropout tolerance under SecAgg+ a useful research question for small cross-silo federations?',
           'Is the worst-client cost of DP under label skew worth a focused study?',
           'Which stronger membership or inversion attack should be the standard evaluator?'],size=21,step=1.05)

OUTPUT.parent.mkdir(parents=True,exist_ok=True)
p.save(OUTPUT)
print(OUTPUT)
print('slide_count',len(p.slides))
