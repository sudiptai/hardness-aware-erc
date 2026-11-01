# ===== POST-HOC CELL (paste as the LAST cell of hardness_aware_ERC_final_colab.ipynb and run after "Run all") =====
# Produces: (1) per-class F1 for IEMOCAP at k=0 vs k=8, (2) confusion matrices, (3) realised k-bar of the entropy-k policy,
#           (4) the trained random-k row at k-bar 6.55 (only if you deleted results/T2_random_kbar_iemocap.* before Run all).
import numpy as np, pandas as pd, os
from sklearn.metrics import f1_score, confusion_matrix

def pooled(name, corpus):
    p = f'{RESULTS}/{name}_{corpus}_pooled.npz'
    return np.load(p) if os.path.exists(p) else None

print('=' * 70); print('PER-CLASS F1 (IEMOCAP, Session 5, pooled over 5 seeds)')
L = LABELS['iemocap']; rows = []
for nm in ['T2_fixed0', 'T2_fixed8', 'T2_hardness']:
    z = pooled(nm, 'iemocap')
    if z is None: print('missing', nm); continue
    f = f1_score(z['y'], z['p'], average=None, labels=range(len(L))) * 100
    rows.append(dict(config=nm, **{L[i]: round(f[i], 1) for i in range(len(L))}, WF1=round(f1_score(z['y'], z['p'], average='weighted') * 100, 2)))
pc = pd.DataFrame(rows); print(pc.to_string(index=False)); pc.to_csv(f'{RESULTS}/table_perclass_iemocap.csv', index=False)

print('\nCONFUSION MATRIX k=0 (rows = true, cols = predicted, % of row)')
z0 = pooled('T2_fixed0', 'iemocap'); z8 = pooled('T2_fixed8', 'iemocap')
for tag, z in (('k=0', z0), ('k=8', z8)):
    cm = confusion_matrix(z['y'], z['p'], labels=range(len(L))); cm = cm / cm.sum(1, keepdims=True) * 100
    print(f'\n{tag}'); print(pd.DataFrame(cm.round(1), index=L, columns=L).to_string())
    pd.DataFrame(cm.round(2), index=L, columns=L).to_csv(f'{RESULTS}/confusion_{tag.replace("=", "")}_iemocap.csv')

print('\n' + '=' * 70); print('REALISED k-bar OF THE ENTROPY-k POLICY (test set, per corpus)')
for c in CFG['corpora']:
    z = pooled('T2_fixed0', c)
    if z is None: continue
    ent = -(z['prob'] * np.log(z['prob'] + 1e-9)).sum(1)
    kk = np.array([KS[min(4, int(np.searchsorted(ENT_Q, e)))] for e in ent])
    print(f'{c}: entropy-k k-bar = {kk.mean():.2f}  (ENT_Q = {np.round(ENT_Q, 3).tolist()})')

print('\n' + '=' * 70); print('RANDOM-k ROW (trained) ON IEMOCAP')
p = f'{RESULTS}/T2_random_kbar_iemocap.csv'
if os.path.exists(p):
    d = pd.read_csv(p); print(d[['seed', 'dev_WF1', 'test_WF1', 'test_MF1', 'test_UA']].round(2).to_string(index=False))
    print('mean WF1 %.2f +- %.2f   (k-bar used: %s)' % (d.test_WF1.mean(), d.test_WF1.std(), RUN_STATE.get('iemocap_fixed_Ses05_kbar', 'from hardness run')))
print('\nDONE - send this cell\'s output.')
