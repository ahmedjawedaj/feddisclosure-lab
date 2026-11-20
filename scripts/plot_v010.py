"""Generate reproducible v0.10 figures from previously verified scalar evidence."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'reports/figures'


def run():
    q=json.loads((ROOT/'reports/results/sprint1/history_matrix_summary.json').read_text())
    cases=q['condition_summary']
    order=['naive','history_same_id','history_reset_id','principal_history_reset_id']
    labels=['No history','History, same ID','History, reset ID','Principal history, reset ID']
    fig,ax=plt.subplots(figsize=(9,5))
    vals=[100*cases[k]['exact_inference_rate'] for k in order]
    ax.barh(labels,vals)
    for i,v in enumerate(vals):
        ax.text(min(v+1,100),i,f'{v:.1f}%',va='center',fontsize=10)
    ax.set(xlim=(0,112),xlabel='Exact complement-count inference (%)',
           title='Policy-scoped versus principal-wide query history\n90 paired synthetic populations, one two-query attack')
    ax.invert_yaxis()
    fig.tight_layout()
    FIG.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/'v010_history_scope.png',dpi=160)
    plt.close(fig)

    m=json.loads((ROOT/'reports/results/sprint3/membership_balanced_comparison/summary.json').read_text())
    names=list(m['aggregate'])
    fig,ax=plt.subplots(figsize=(9,5))
    x=list(range(len(names)))
    for delta,condition,label in [(-0.17,'iid','IID'),(0.17,'dirichlet_balanced_0.1','Severe non-IID')]:
        ys=[m['aggregate'][k]['by_partition'][condition]['mean_roc_auc'] for k in names]
        ax.bar([p+delta for p in x],ys,width=0.33,label=label)
    ax.axhline(.5,ls='--',linewidth=1,label='Chance AUC = 0.5')
    ax.set_ylim(0,0.65)
    ax.set_xticks(x,[v.replace('_',' ') for v in names])
    ax.set(ylabel='Mean output-only membership ROC-AUC',
           title='Five independently trained models per distribution\nSame sampled member and nonmember query records within each seed')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG/'v010_membership_five_models.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    run()
    print(FIG/'v010_history_scope.png')
    print(FIG/'v010_membership_five_models.png')
