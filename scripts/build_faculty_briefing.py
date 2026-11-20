"""Build interim faculty-facing PPTX from verified research results, not projections."""
from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'reports/faculty-briefing-v0.10.0.pptx'
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
    text(s,'FedDisclosure Lab  •  INTERIM RESEARCH  •  NOV 2026',.82,7.1,9,.18,9,MUTED)
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
    im=Image.open(path)
    r=min(w/im.width,h/im.height)
    dw,dh=im.width*r,im.height*r
    s.shapes.add_picture(str(path),Inches(x+(w-dw)/2),Inches(y+(h-dh)/2),width=Inches(dw),height=Inches(dh))

# 1
s=p.slides.add_slide(BLANK);s.background.fill.solid();s.background.fill.fore_color.rgb=BG
shape(s,.75,1.35,.1,3.25,TEAL,False)
text(s,'FEDDISCLOSURE LAB',1.05,1.38,11.2,1.12,44,WHITE,True)
text(s,'When raw data stays local, what can still be inferred?',1.08,2.62,10.8,.85,24,TEAL)
text(s,'Research engineering, threat-model analysis and reproducible evidence',1.08,4.04,10.8,.7,20,MUTED)
text(s,'Interim faculty briefing  |  20 November 2026  |  Independent research prototype',1.08,6.62,10.9,.45,14,MUTED)

# 2
s=slide('Research question and scope','Two core tracks. A digital twin is an optional later extension, not claimed here.')
bullets(s,[
 'Can published aggregate answers reveal restricted subgroup counts?',
 'Can gradients, client updates or model outputs expose training information?',
 'How do observer permissions change the feasibility of disclosure?',
 'Which established defenses preserve useful analytics and model accuracy?'],size=23,step=1.04)

# 3
s=slide('What actually runs today','An attacker sees only what its documented observation role permits.')
card(s,'INSTITUTIONS','3','Separate local HTTP services and SQLite data',.8,1.78)
card(s,'POLICY QUERIES','1,080','90 populations × 4 controls × 3 queries',4.85,1.78)
card(s,'MODEL OUTPUT ATTACKS','30','10 trained models × 3 methods',8.9,1.78)
bullets(s,['Raw rows remain at institutions in the analytics demo',
           'Independent training workers produce real PyTorch FedAvg/FedProx results',
           'Evaluator-only ground truth is not sent to the attack implementations'],x=.95,y=4.18,size=18,step=.65)

# 4
s=slide('A real disclosure-control bypass','Caller-selected experiment IDs reset the old history check.')
card(s,'APPROVED TOTAL','110','Released count across all sites',.8,1.82)
card(s,'APPROVED NEGATIVE','107','Different experiment ID',4.86,1.82)
card(s,'INFERRED SENSITIVE','3','Exact complement inference',8.91,1.82)
bullets(s,['Original experiment-scoped history: both queries returned HTTP 200',
           'New principal-wide history: the second query returned HTTP 409',
           'Policy switching to weaker naive mode returned HTTP 403'],x=.9,y=4.1,size=19,step=.7)

# 5
s=slide('Matched query-history benchmark','Exact inference across 90 synthetic populations. Lower is better; not a DP claim.')
shape(s,.8,1.65,11.78,4.55,RGBColor(255,255,255))
image(s,ROOT/'reports/figures/v010_history_scope.png',.95,1.72,11.45,4.4)
text(s,'Residual inference occurs where the tested near-duplicate threshold does not reject the complement.',.9,6.43,11.6,.47,15,AMBER)

# 6
s=slide('Federated accuracy is not the same as client fairness','20 earlier PyTorch runs across five seeds and four equal-work partition conditions.')
shape(s,.8,1.6,11.8,4.58,RGBColor(255,255,255))
image(s,ROOT/'reports/figures/sprint5_balanced_heterogeneity.png',.95,1.76,11.45,4.28)
text(s,'Exploratory five-seed comparison: severe skew can preserve global accuracy while lowering worst-client accuracy.',.95,6.37,11.3,.45,14,AMBER)

# 7
s=slide('Gradient and model-update observer distinctions','The observer model matters more than a single aggregated “privacy score”.')
card(s,'EXACT GRADIENT','2.43e-9','Five-seed single-image mean MSE',.8,1.72)
card(s,'AGGREGATE-ONLY','0.1085','Five-seed mean MSE; simple target probe',4.85,1.72)
card(s,'COLLUSION','0.0166','Target isolation restores reconstruction fidelity',8.90,1.72)
bullets(s,['Exact gradients, local updates and aggregated updates are different observations',
           'Secure aggregation may hide an individual update, not prevent all inference',
           'All-but-one collusion is an explicit limitation of the idealized aggregate experiment'],x=.95,y=4.18,size=18,step=.70)

# 8
s=slide('Output-only membership inference: negative results','Five independently trained models per distribution, 120 identical sampled queries per model pair.')
shape(s,.8,1.55,11.8,4.7,RGBColor(255,255,255))
image(s,ROOT/'reports/figures/v010_membership_five_models.png',.95,1.65,11.5,4.48)
text(s,'All three heuristics are near chance. Underlying training sets are not identical; contrast is descriptive.',.91,6.38,11.6,.49,14,AMBER)

# 9
s=slide('What has NOT been verified','These remain hard release gates, not a matter of slide polish.')
bullets(s,['Native Docker Compose network, volume and process isolation',
           'Real Flower ServerApp/ClientApp training with tensor parity',
           'OpenDP count-release mechanism and verified accounting',
           'Opacus DP-SGD with native ε/δ accountant',
           'Flower SecAgg+ with participant threshold and dropout tests',
           'Full 28×28 MNIST and independent external replication'],size=19,step=.69)

# 10
s=slide('Research contribution candidates','These are hypotheses to investigate, not claims of novelty.')
bullets(s,['Governance-constrained federated analytics with cross-query budgets',
           'Disclosure under realistic client-update visibility and collusion',
           'Privacy–utility–fairness trade-offs under heterogeneous federation',
           'Benchmark design that separates attacker capabilities and ground-truth evaluation'],size=21,step=.97)

# 11
s=slide('What we need from faculty','Methodological feedback determines the thesis direction.')
bullets(s,['Which threat model and observer best match a credible deployment?',
           'Which negative result or mismatch is scientifically interesting?',
           'What additional datasets, baselines and power analysis are necessary?',
           'Can a narrow novel question be formulated and independently evaluated?'],size=21,step=.99)

# 12
s=slide('Next gates, in priority order','Release acceptance is evidence-based, not a progress-bar claim.')
bullets(s,['1. Native Flower, OpenDP, Opacus and Docker execution logs',
           '2. Matched defended versus undefended experiments with accounting',
           '3. Flower SecAgg+ and realistic attacker observation tests',
           '4. Full MNIST, five-seed replication and independent data source',
           '5. Revised manuscript, novelty review and faculty feedback'],size=20,step=.83)

OUTPUT.parent.mkdir(parents=True,exist_ok=True)
p.save(OUTPUT)
print(OUTPUT)
print('slide_count',len(p.slides))
