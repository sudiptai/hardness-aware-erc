import nbformat as nbf
nb = nbf.v4.new_notebook(); cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# Hardness-Aware Context Selection for Multimodal ERC — rev. 3 protocol
Companion notebook for the IEEE Transactions draft (rev. 3). Implements exactly the protocol in Sections III–V of the paper:

* **Encoders (frozen):** WavLM-Large with learned softmax layer weights; RoBERTa-base masked mean-pool; MiniLM-L6 compact branch; wav2vec2-base as ablation.
* **Hardness h(t):** normalised kNN label entropy over *training-fold* neighbours, excluding the utterance itself, its own dialogue and (IEMOCAP) its own speaker. Never reads y(t).
* **Representation:** learned PCA-initialised projection + FiLM gate on h. Ablations: scalar concat, random gate (matched marginal), MLP vs conv head.
* **Context:** attention head over z(t−k…t). Oracle k* by seed-averaged margin, monotone, with an *unresolvable* class. Policies: fixed k ∈ {0,1,2,4,8}, random-k at matched k̄, posterior-entropy-k, hardness-k (ours), oracle.
* **Loss:** CE × (1 + λh), λ ∈ {−1, −0.5, 0, 0.5, 1}.
* **Protocol:** IEMOCAP 6-class fixed split (Ses01–04 train / Ses05 test) primary + 5-fold LOSO secondary; MELD 7-class official. Inner dev split. 5 seeds, paired bootstrap CIs vs reproduced baselines, pooled ECE/AURC, Spearman + partial ρ vs annotator disagreement.

Runtime: Colab **A100 or L4** recommended (WavLM-Large on ~21k utterances ≈ 1.5–2.5 h on a T4; cached once). Everything after embedding extraction runs in minutes per configuration.

Order: `1 Setup → 2 Data → 3 Embeddings → 4 Hardness → 5 Model → 6 Oracle k* → 7 Policies & experiments → 8 Calibration & annotator analysis → 9 Compact branch → 10 Export tables`
""")

md("## 1. Setup")
code("""!pip -q install transformers==4.44.2 torchaudio soundfile scikit-learn pandas scipy tqdm
!apt-get -qq install -y ffmpeg > /dev/null
import os, re, json, time, glob, math, random, subprocess, itertools, collections, shutil
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
from tqdm.auto import tqdm
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'; print(DEVICE, torch.__version__)
def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(0)""")

code("""from google.colab import drive
drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/hardness_erc'
for d in ['cache', 'results', 'data']: os.makedirs(f'{ROOT}/{d}', exist_ok=True)
CACHE, RESULTS, DATA = f'{ROOT}/cache', f'{ROOT}/results', f'{ROOT}/data'""")

code("""CFG = dict(
    corpora      = ['meld', 'emorynlp', 'iemocap'],   # all three; a corpus whose files are missing is skipped with a message
    meld_labels_zip = f'{DATA}/meld_labels.zip',   # label CSVs from the MELD GitHub repo (uploaded)
    text_only_ok = True,                            # run the text branch alone for corpora whose audio is absent
    audio_model  = 'microsoft/wavlm-large',        # ablation: 'facebook/wav2vec2-base'
    text_model   = 'roberta-base',
    slm_model    = 'sentence-transformers/all-MiniLM-L6-v2',
    K_grid       = [5, 10, 20],                    # hardness neighbours (selected on inner dev)
    H            = 128,                            # projection size
    windows      = [0, 1, 2, 4, 8],
    tau_grid     = [0.5, 0.7, 0.9],                 # hardness-k decision threshold, selected on inner dev
    lam_grid     = [-1.0, -0.5, 0.0, 0.5, 1.0],
    seeds        = [11, 22, 33, 44, 55],
    oracle_seeds = [1, 2, 3],                      # seeds used to define k* (inside training fold)
    margin_tau   = 0.10,
    epochs       = 30, batch = 64, lr = 5e-4, wd = 1e-2,
    # ---- explicit dataset locations (your Drive layout); auto-search under hardness_erc/ is the fallback ----
    iemocap_archive = '/content/drive/MyDrive/IEMOCAP/archive.zip',             # complete IEMOCAP_full_release zip (12.4 GB), uploaded 28 Oct
    iemocap_path = f'{ROOT}/results/IEMOCAP/IEMOCAP_full_release',             # (incomplete older copy; used only if the archive is absent)
    iemocap_csv  = f'{ROOT}/results/IEMOCAP/iemocap_utterance_labels.csv',
    meld_archive = '/content/drive/MyDrive/MELD_final/MELD.Raw.tar.gz',          # official tarball (10.9 GB), uploaded 28 Oct
    meld_archive_alt = f'{ROOT}/results/MELD/archive.zip',
)
LABELS = {'meld':     ['anger','disgust','fear','joy','neutral','sadness','surprise'],
          'emorynlp': ['Joyful','Mad','Peaceful','Neutral','Sad','Powerful','Scared'],
          'iemocap':  ['angry','happy','neutral','sad','excited','frustrated']}
COMMON4 = {'angry':'angry','anger':'angry','happy':'happy','joy':'happy','excited':'happy','neutral':'neutral','sad':'sad','sadness':'sad'}
print(json.dumps(CFG, indent=1))""")

code("""# ---- Dataset path check (runs before anything heavy) ----
def _sz(p): return f'{os.path.getsize(p)/1e9:.1f} GB' if os.path.exists(p) else ''
checks = {
  'IEMOCAP archive (zip)'  : (CFG['iemocap_archive'], os.path.exists(CFG['iemocap_archive'])),
  'IEMOCAP release folder' : (CFG['iemocap_path'], os.path.isdir(f"{CFG['iemocap_path']}/Session1/dialog/transcriptions")),
  'IEMOCAP label CSV'      : (CFG['iemocap_csv'],  os.path.exists(CFG['iemocap_csv'])),
  'MELD archive (tar.gz)'  : (CFG['meld_archive'], os.path.exists(CFG['meld_archive'])),
  'MELD archive alt (zip)' : (CFG['meld_archive_alt'], os.path.exists(CFG['meld_archive_alt'])),
}
for name, (path, ok) in checks.items(): print(('OK      ' if ok else 'MISSING ') + f'{name:24s} {path} {_sz(path)}')
if not (checks['IEMOCAP archive (zip)'][1] or checks['IEMOCAP release folder'][1]): print('\\n!! no IEMOCAP source in Drive -> the Kaggle fallback will be tried')
if not (checks['MELD archive (tar.gz)'][1] or checks['MELD archive alt (zip)'][1]): print('!! no MELD archive in Drive -> converted wavs in data/meld_wav or the Kaggle fallback will be used')
print('\\nlocal disk free:', f'{shutil.disk_usage(\"/content\").free/1e9:.0f} GB (IEMOCAP + MELD extraction needs ~30 GB)')""")

md("""### 2-pre. IEMOCAP source: Drive archive → (fallback) Kaggle mirror
Order of preference: (1) a complete extracted `IEMOCAP_full_release` folder in Drive; (2) `MyDrive/IEMOCAP/archive.zip` (12.4 GB), unzipped once onto Colab local disk; (3) the Kaggle mirror ([dejolilandry/iemocapfullrelease](https://www.kaggle.com/datasets/dejolilandry/iemocapfullrelease)), used purely as a transfer path. You hold the USC licence; cite Busso et al. (2008), not Kaggle, and do not redistribute. MELD audio likewise comes from `MyDrive/MELD_final/MELD.Raw.tar.gz` (official tarball) or the converted wavs already in `hardness_erc/data/meld_wav`.

**One-time setup:** Kaggle → your profile → *Settings* → *API* → *Create New Token* (downloads `kaggle.json`). In Colab, open the key icon (Secrets) in the left bar and add two secrets: `KAGGLE_USERNAME` and `KAGGLE_KEY` with the values from that file, toggled on for this notebook.

The download lands on Colab's local disk (fast, ~5 min) and is re-fetched only if the cached embeddings are missing; the raw corpus never needs to live in Drive.""")
code("""KAGGLE_DIR = '/content/iemocap_kaggle'
def drive_iemocap_complete():
    r = find_iemocap_root() if 'find_iemocap_root' in globals() else None
    if not r: return False
    return all(len(glob.glob(f'{r}/Session{i}/dialog/transcriptions/*.txt')) >= 25 and len(glob.glob(f'{r}/Session{i}/sentences/wav/*/*.wav')) > 1500 for i in range(1, 6))

KAGGLE_MELD_DIR = '/content/meld_kaggle'
def kaggle_auth():
    \"\"\"Load KAGGLE_USERNAME / KAGGLE_KEY from Colab Secrets into the environment; False if they are not set.\"\"\"
    if os.environ.get('KAGGLE_KEY'): return True
    try:
        from google.colab import userdata
        key = userdata.get('KAGGLE_KEY'); os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
        os.environ['KAGGLE_KEY'] = key; os.environ['KAGGLE_API_TOKEN'] = key          # new-format KGAT_ tokens are accepted by both names
    except Exception as e:
        print('Kaggle secrets not set (KAGGLE_USERNAME / KAGGLE_KEY in Colab Secrets, notebook access ON) ->', e); return False
    subprocess.run(['pip', '-q', 'install', '-U', 'kaggle'], check=True); return True

def kaggle_download(slug, dest, label):
    os.makedirs(dest, exist_ok=True); print(f'downloading {label} from Kaggle ({slug}) ...')
    r = subprocess.run(['kaggle', 'datasets', 'download', '-d', slug, '-p', dest, '--unzip'], capture_output=True, text=True)
    print((r.stdout or '')[-600:], (r.stderr or '')[-600:])
    for z in glob.glob(f'{dest}/**/*.zip', recursive=True): subprocess.run(['unzip', '-qo', z, '-d', os.path.dirname(z)])
    for t in glob.glob(f'{dest}/**/*.tar.gz', recursive=True): subprocess.run(['tar', '-xzf', t, '-C', os.path.dirname(t)])
    return r.returncode == 0

def extract_iemocap_archive():
    \"\"\"Unzip the Drive copy of IEMOCAP_full_release (archive.zip) onto Colab local disk; ~10 min for 12 GB.\"\"\"
    arc = CFG.get('iemocap_archive', '')
    if not os.path.exists(arc): return False
    if glob.glob(f'{KAGGLE_DIR}/**/Session5/dialog/EmoEvaluation', recursive=True): print('IEMOCAP already extracted on local disk'); return True
    os.makedirs(KAGGLE_DIR, exist_ok=True); print(f'extracting {arc} ({os.path.getsize(arc)/1e9:.1f} GB) to local disk ...')
    r = subprocess.run(['unzip', '-qo', arc, '-d', KAGGLE_DIR, '-x', '__MACOSX/*', '*/._*'], capture_output=True, text=True)
    if r.returncode not in (0, 1, 2): print('unzip said:', (r.stderr or '')[-400:])
    for z in glob.glob(f'{KAGGLE_DIR}/**/*.zip', recursive=True): subprocess.run(['unzip', '-qo', z, '-d', os.path.dirname(z)])   # nested zips, if any
    for t in glob.glob(f'{KAGGLE_DIR}/**/*.tar.gz', recursive=True): subprocess.run(['tar', '-xzf', t, '-C', os.path.dirname(t)])
    ok = bool(glob.glob(f'{KAGGLE_DIR}/**/Session5/dialog/EmoEvaluation', recursive=True))
    print('IEMOCAP archive extracted, all five sessions present:', ok, '| wavs:', len(glob.glob(f'{KAGGLE_DIR}/**/Session*/sentences/wav/*/*.wav', recursive=True))); return ok

def fetch_iemocap_from_kaggle():
    if glob.glob(f'{KAGGLE_DIR}/**/Session5/dialog/EmoEvaluation', recursive=True): print('IEMOCAP copy already on local disk'); return True
    if extract_iemocap_archive(): return True                                     # Drive archive first; Kaggle only if that is absent
    if not kaggle_auth(): return False
    kaggle_download('dejolilandry/iemocapfullrelease', KAGGLE_DIR, 'IEMOCAP (several GB)')
    ok = bool(glob.glob(f'{KAGGLE_DIR}/**/Session5/dialog/EmoEvaluation', recursive=True)); print('Kaggle IEMOCAP copy complete:', ok); return ok

def fetch_meld_from_kaggle():
    \"\"\"Fallback when no MELD archive / clips are found in Drive: zaber666/meld-dataset (clips as dia*_utt*.mp4 or .wav + *_sent_emo.csv).\"\"\"
    if glob.glob(f'{KAGGLE_MELD_DIR}/**/dia*_utt*.*', recursive=True): print('Kaggle MELD copy already on local disk'); return True
    if not kaggle_auth(): return False
    kaggle_download('zaber666/meld-dataset', KAGGLE_MELD_DIR, 'MELD')
    n = len(glob.glob(f'{KAGGLE_MELD_DIR}/**/dia*_utt*.*', recursive=True)); print('Kaggle MELD clips found:', n); return n > 1000

def iemocap_meta_complete(p=f'{CACHE}/meta_iemocap.csv'):
    \"\"\"True only if the cached IEMOCAP metadata covers all five sessions (a Session1-only file is left over from an incomplete upload).\"\"\"
    if not os.path.exists(p): return False
    try: s = set(pd.read_csv(p, usecols=['session'], keep_default_na=False)['session'].astype(str))
    except Exception: return False
    return all(f'Ses0{i}' in s for i in range(1, 6))

def purge_iemocap_cache(reason):
    \"\"\"Delete stale per-corpus cache files so metadata and embeddings are rebuilt consistently.\"\"\"
    victims = [f'{CACHE}/meta_iemocap.csv'] + glob.glob(f'{CACHE}/*_iemocap.npy') + glob.glob(f'{CACHE}/kstar_iemocap_*.npy')
    for v in victims:
        if os.path.exists(v): os.remove(v)
    print(f'!! {reason} -> removed {len(victims)} stale IEMOCAP cache file(s); they will be rebuilt')

NEED_RAW_IEMOCAP = not os.path.exists(f'{CACHE}/audio_{CFG[\"audio_model\"].split(\"/\")[-1]}_iemocap.npy') or not iemocap_meta_complete()
if 'iemocap' in CFG['corpora'] and NEED_RAW_IEMOCAP:
    if os.path.exists(f'{CACHE}/meta_iemocap.csv') and not iemocap_meta_complete(): purge_iemocap_cache('cached IEMOCAP metadata does not cover all five sessions')
    print('raw IEMOCAP needed (no complete cached embeddings/metadata yet)')
else: print('IEMOCAP embeddings already cached for all five sessions -> raw corpus not needed this run')""")

md("""## 2. Data
Per-corpus tables with `corpus, split, dialogue_id, utterance_id, speaker, text, label, wav` and, where available, `annotators` (per-annotator labels, `|`-joined) for the disagreement analysis (Table VI).

**IEMOCAP** uses the fixed ERC split (Ses01–04 train, Ses05 test; 6 classes incl. *excited* and *frustrated*, matching MM-DFN/CORECT/SDT/UniMSE/MultiEMO/InstructERC). LOSO folds are derived from the session column.
**MELD** uses the official split, 7 classes. Labels/transcripts come from `meld_labels.zip` (the MELD GitHub repo you uploaded); audio comes from `MELD.Raw.tar.gz` **if present in Drive** — otherwise MELD runs text-only until you add it.
**EmoryNLP** (text-only, 7 classes, 659 scenes as dialogues, official splits) is included from the same zip as a third corpus for the text branch and the transfer check.""")

code("""MELD_URL = 'https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz'
MELD_CSV_RAW = 'https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/'
MELD_DIR, MELD_WAV, MELD_TAR = '/content/MELD.Raw', f'{DATA}/meld_wav', f'{DATA}/MELD.Raw.tar.gz'   # tarball kept in Drive

GH = 'https://raw.githubusercontent.com/declare-lab/MELD/master/data/'
LABEL_FILES = ['MELD/train_sent_emo.csv', 'MELD/dev_sent_emo.csv', 'MELD/test_sent_emo.csv',
               'emorynlp/emorynlp_train_final.csv', 'emorynlp/emorynlp_dev_final.csv', 'emorynlp/emorynlp_test_final.csv']
def unzip_labels():
    \"\"\"Label/transcript CSVs: from meld_labels.zip in Drive if present, else straight from the MELD GitHub repo.\"\"\"
    out = '/content/meld_labels'; os.makedirs(f'{out}/emorynlp', exist_ok=True)
    if os.path.exists(f'{out}/train_sent_emo.csv') and os.path.exists(f'{out}/emorynlp/emorynlp_train_final.csv'): return out
    if os.path.exists(CFG['meld_labels_zip']):
        subprocess.run(['unzip', '-qo', CFG['meld_labels_zip'], '-d', out], check=True)
    else:
        print('meld_labels.zip not in Drive -> downloading label CSVs from github.com/declare-lab/MELD')
        for f in LABEL_FILES:
            dst = f'{out}/{f.split("/",1)[1]}' if f.startswith('MELD/') else f'{out}/{f}'
            subprocess.run(['wget', '-q', '-O', dst, GH + f], check=True)
    assert os.path.exists(f'{out}/train_sent_emo.csv'), 'label CSVs could not be obtained'
    return out

MELD_EXTRACT = '/content/MELD_extract'
def _archives():
    for k in ('meld_archive', 'meld_archive_alt'):
        if os.path.exists(CFG.get(k, '')) and os.path.getsize(CFG[k]) > 5e8: return [CFG[k]]
    arcs = glob.glob(f'{ROOT}/**/*.zip', recursive=True) + glob.glob(f'{ROOT}/**/*.tar.gz', recursive=True) + glob.glob(f'{ROOT}/**/*.tgz', recursive=True)
    return [a for a in arcs if 'meld' in a.lower() and os.path.getsize(a) > 5e8]     # MELD archives: any name containing 'meld', > 500 MB

def ensure_meld_audio():
    \"\"\"True when MELD clips (diaX_uttY.mp4) are available; handles the official tarball, Kaggle zips, or an extracted folder anywhere under hardness_erc.\"\"\"
    if glob.glob(f'{MELD_EXTRACT}/**/dia*_utt*.mp4', recursive=True) or glob.glob(f'{MELD_EXTRACT}/**/dia*_utt*.wav', recursive=True): return True
    if glob.glob(f'{ROOT}/**/dia*_utt*.mp4', recursive=True):                           # already-extracted copy in Drive
        os.makedirs(MELD_EXTRACT, exist_ok=True)
        if not os.path.exists(f'{MELD_EXTRACT}/drive'): os.symlink(ROOT, f'{MELD_EXTRACT}/drive')
        return True
    arcs = _archives()
    if not arcs:
        print('no MELD archive (*.zip / *.tar.gz) or extracted dia*_utt*.mp4 found under', ROOT, '-> trying Kaggle (zaber666/meld-dataset)')
        if fetch_meld_from_kaggle():
            os.makedirs(MELD_EXTRACT, exist_ok=True)
            if not os.path.exists(f'{MELD_EXTRACT}/kaggle'): os.symlink(KAGGLE_MELD_DIR, f'{MELD_EXTRACT}/kaggle')
            return True
        print('MELD audio not available -> MELD runs TEXT-ONLY'); return False
    os.makedirs(MELD_EXTRACT, exist_ok=True)
    for arc in arcs:
        print('extracting', arc, f'({os.path.getsize(arc)/1e9:.1f} GB) ...')
        if arc.endswith('.zip'): subprocess.run(['unzip', '-qo', arc, '-d', MELD_EXTRACT], check=True)
        else: subprocess.run(['tar', '-xzf', arc, '-C', MELD_EXTRACT], check=True)
    for t in glob.glob(f'{MELD_EXTRACT}/**/*.tar.gz', recursive=True):                 # the official tarball nests three split tarballs
        subprocess.run(['tar', '-xzf', t, '-C', os.path.dirname(t)], check=True)
    for z in glob.glob(f'{MELD_EXTRACT}/**/*.zip', recursive=True):
        subprocess.run(['unzip', '-qo', z, '-d', os.path.dirname(z)], check=True)
    ok = bool(glob.glob(f'{MELD_EXTRACT}/**/dia*_utt*.mp4', recursive=True))
    print('MELD clips found:', len(glob.glob(f'{MELD_EXTRACT}/**/dia*_utt*.mp4', recursive=True))); return ok

def meld_clip_index():
    \"\"\"{(split, 'diaX_uttY'): path} by inspecting each clip's path for train/dev/test; works for official and Kaggle layouts.\"\"\"
    idx = {}
    for p in glob.glob(f'{MELD_EXTRACT}/**/dia*_utt*.mp4', recursive=True) + glob.glob(f'{MELD_EXTRACT}/**/dia*_utt*.wav', recursive=True):
        low = p.lower().replace(MELD_EXTRACT.lower(), '')
        split = 'train' if 'train' in low else ('dev' if ('dev' in low or 'val' in low) else ('test' if 'test' in low else None))
        key = (split, os.path.splitext(os.path.basename(p))[0])
        if split and (key not in idx or p.endswith('.wav')): idx[key] = p     # prefer a ready wav over an mp4
    return idx

def to_wav(src, dst):
    if os.path.exists(dst): return True
    return subprocess.run(['ffmpeg','-y','-loglevel','error','-i',src,'-ac','1','-ar','16000',dst]).returncode == 0

def load_meld():
    lab = unzip_labels(); files = {'train': 'train_sent_emo.csv', 'dev': 'dev_sent_emo.csv', 'test': 'test_sent_emo.csv'}
    os.makedirs(MELD_WAV, exist_ok=True); have_wav = set(os.path.basename(w) for w in glob.glob(f'{MELD_WAV}/*.wav'))
    # converted 16-kHz wavs from an earlier run are reused; the archive is only extracted if more than 1% of clips are missing
    n_total = sum(len(pd.read_csv(f'{lab}/{f}')) for f in files.values()); print(f'MELD: {len(have_wav)} / {n_total} converted wavs already in {MELD_WAV}')
    has_audio = len(have_wav) >= 0.99 * n_total or ensure_meld_audio(); clips = meld_clip_index() if has_audio else {}
    rows = []; missing = 0
    for split, csvn in files.items():
        df = pd.read_csv(f'{lab}/{csvn}')
        for _, r in tqdm(df.iterrows(), total=len(df), desc=f'MELD {split}'):
            name = f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}"; wav = ''
            if has_audio:
                w = f'{MELD_WAV}/{split}_{name}.wav'
                if os.path.basename(w) in have_wav: wav = w
                else:
                    mp4 = clips.get((split, name)) if clips else None             # (clips are empty when we rely on the converted wavs alone)
                    if mp4 and to_wav(mp4, w): wav = w
                    else: missing += 1
            rows.append(dict(corpus='meld', split=split, session='', dialogue_id=f"meld_{split}_{r['Dialogue_ID']}", utterance_id=int(r['Utterance_ID']),
                             speaker=str(r['Speaker']), text=str(r['Utterance']), label=r['Emotion'], wav=wav, annotators=''))
    if has_audio: print(f'MELD: {len(rows)} utterances, {missing} without a clip (a few corrupt/absent clips are normal)')
    return pd.DataFrame(rows)

def load_emorynlp():
    lab = unzip_labels(); rows = []
    for split in ('train', 'dev', 'test'):
        df = pd.read_csv(f'{lab}/emorynlp/emorynlp_{split}_final.csv')
        for _, r in df.iterrows():
            rows.append(dict(corpus='emorynlp', split=split, session='', dialogue_id=f"emory_s{r['Season']}e{r['Episode']}sc{r['Scene_ID']}", utterance_id=int(r['Utterance_ID']),
                             speaker=str(r['Speaker']), text=str(r['Utterance']), label=r['Emotion'], wav='', annotators=''))
    return pd.DataFrame(rows)

IEMO = {'ang':'angry','hap':'happy','neu':'neutral','sad':'sad','exc':'excited','fru':'frustrated'}
def find_iemocap_root():
    cands = [CFG['iemocap_path']] + [os.path.dirname(os.path.dirname(os.path.dirname(p))) for p in glob.glob(f'{ROOT}/**/Session1/dialog/EmoEvaluation', recursive=True) + glob.glob('/content/iemocap_kaggle/**/Session1/dialog/EmoEvaluation', recursive=True)]
    def complete(c): return all(os.path.isdir(f'{c}/Session{i}/dialog/transcriptions') and os.path.isdir(f'{c}/Session{i}/sentences/wav') for i in range(1, 6))
    good = [os.path.normpath(c) for c in cands if os.path.isdir(f'{c}/Session1/dialog/EmoEvaluation')]
    full = [c for c in good if complete(c)]
    if good and not full: print('!! IEMOCAP copies found but none has all five sessions complete:', good)
    return (full or good or [None])[0]

def find_iemocap_csv():
    if os.path.exists(CFG['iemocap_csv']): return CFG['iemocap_csv']
    hits = glob.glob(f'{ROOT}/**/iemocap_utterance_labels.csv', recursive=True); return hits[0] if hits else None

def iemocap_transcripts(root):
    \"\"\"{utterance_id: text} from SessionN/dialog/transcriptions/*.txt\"\"\"
    trans = {}
    for tpath in glob.glob(f'{root}/Session*/dialog/transcriptions/*.txt'):
        if os.path.basename(tpath).startswith('._'): continue                      # macOS resource-fork stubs
        with open(tpath, errors='ignore') as f:
            for line in f:
                m = re.match(r'(\\S+) \\[.*?\\]: ?(.*)', line.strip())
                if m: trans[m.group(1)] = m.group(2)
    return trans

def load_iemocap():
    if not drive_iemocap_complete(): fetch_iemocap_from_kaggle()
    root = find_iemocap_root(); assert root, f"IEMOCAP not found under {ROOT} or Kaggle dir: expected <root>/Session1/dialog/EmoEvaluation"
    csvp = find_iemocap_csv(); print('IEMOCAP root:', root, '| label CSV:', csvp)
    trans = iemocap_transcripts(root); rows = []
    if csvp:                                             # preferred: the verified utterance-level summary (10,039 rows, per-annotator votes, agreement, VAD)
        d = pd.read_csv(csvp); drop_lab = collections.Counter(); drop_tr = collections.Counter()
        for s_ in range(1, 6):
            n_t = len(glob.glob(f'{root}/Session{s_}/dialog/transcriptions/*.txt')); n_w = len(glob.glob(f'{root}/Session{s_}/sentences/wav/*/*.wav'))
            print(f'  Session{s_}: {n_t} transcription files, {n_w} wav files')
        for _, r in d.iterrows():
            if r['label'] not in IEMO: drop_lab[int(r['session'])] += 1; continue
            if r['utterance'] not in trans: drop_tr[int(r['session'])] += 1; continue
            wav = f"{root}/Session{r['session']}/sentences/wav/{r['dialog']}/{r['utterance']}.wav"
            rows.append(dict(corpus='iemocap', split='train' if int(r['session']) < 5 else 'test', session=f"Ses0{r['session']}", dialogue_id=f"iemocap_{r['dialog']}",
                             utterance_id=float(r['start']), speaker=f"Ses0{r['session']}{r['speaker']}", text=trans[r['utterance']], label=IEMO[r['label']],
                             wav=wav if os.path.exists(wav) else '', annotators=str(r['cat_votes']), agreement=float(r['agreement']),
                             valence=float(r['valence']), activation=float(r['activation']), dominance=float(r['dominance'])))
    else:                                                # fallback: parse SessionN/dialog/EmoEvaluation/*.txt directly
        for s_ in range(1, 6):
            ev_dir, wav_root = f'{root}/Session{s_}/dialog/EmoEvaluation', f'{root}/Session{s_}/sentences/wav'
            for ev in sorted(glob.glob(f'{ev_dir}/*.txt')):
                if os.path.basename(ev).startswith('._'): continue
                dia = os.path.basename(ev)[:-4]; labelled, annots, cur = [], collections.defaultdict(list), None
                with open(ev, errors='ignore') as f:
                    for line in f:
                        if line.startswith('['):
                            p = line.strip().split('\\t')
                            if len(p) >= 3: cur = p[1]; labelled.append((cur, p[2], float(re.findall(r'[\\d.]+', p[0])[0])))
                            else: cur = None
                        elif line.startswith('C-E') and cur: annots[cur].append(line.split(':', 1)[1].split(';')[0].strip())   # external evaluators only; C-F1/C-M1 are the actor's self-labels
                for utt, emo, t0 in labelled:
                    if emo not in IEMO or utt not in trans: continue
                    wav = f'{wav_root}/{dia}/{utt}.wav'; votes = annots[utt]
                    rows.append(dict(corpus='iemocap', split='train' if s_ < 5 else 'test', session=f'Ses0{s_}', dialogue_id=f'iemocap_{dia}', utterance_id=t0,
                                     speaker=f'Ses0{s_}{utt.split("_")[-1][0]}', text=trans[utt], label=IEMO[emo], wav=wav if os.path.exists(wav) else '',
                                     annotators='|'.join(votes), agreement=(collections.Counter(votes).most_common(1)[0][1] / len(votes)) if votes else np.nan))
    df = pd.DataFrame(rows)
    if csvp: print('  dropped per session -> not in 6-class set:', dict(sorted(drop_lab.items())), '| no transcript found:', dict(sorted(drop_tr.items())))
    assert len(df), 'IEMOCAP: no rows loaded - check the transcription folders listed above'
    df['utterance_id'] = df.groupby('dialogue_id')['utterance_id'].rank(method='first').astype(int) - 1   # order by start time within dialogue
    print(f"IEMOCAP: {len(df)} utterances in the 6-class set, {(df['wav']=='').sum()} without wav, {df['text'].eq('').sum()} empty transcripts")
    print('  per session:', df.groupby('session').size().to_dict(), '| per split:', df.groupby('split').size().to_dict())
    for s_ in range(1, 6):
        if (df.session == f'Ses0{s_}').sum() == 0: print(f'  !! Session{s_} contributed 0 utterances - its transcriptions or labels are missing')
    print(df['label'].value_counts().to_dict())
    return df

frames = []
for c in list(CFG['corpora']):
    p = f'{CACHE}/meta_{c}.csv'
    if c == 'iemocap' and os.path.exists(p) and not iemocap_meta_complete(p): purge_iemocap_cache('cached IEMOCAP metadata is incomplete')
    if c == 'iemocap' and not os.path.exists(p) and not drive_iemocap_complete() and not fetch_iemocap_from_kaggle():
        print('IEMOCAP not available (no complete copy in Drive, archive/Kaggle fetch failed) -> skipped'); CFG['corpora'].remove(c); continue
    loaders = {'meld': load_meld, 'emorynlp': load_emorynlp, 'iemocap': load_iemocap}
    if os.path.exists(p):
        df = pd.read_csv(p, keep_default_na=False)
        stale = c != 'emorynlp' and (df['wav'] == '').all() and (c != 'meld' or ensure_meld_audio())
        if stale:                                    # cached during a text-only run; audio is available now -> rebuild
            print(f'{c}: cached metadata has no audio paths but audio is now present -> rebuilding'); df = loaders[c](); df.to_csv(p, index=False)
    else:
        df = loaders[c]()
        if c == 'iemocap' and not all((df.session == f'Ses0{s_}').any() for s_ in range(1, 6)):
            print('!! IEMOCAP loaded with fewer than five sessions -> NOT cached and skipped (fix the raw copy / Kaggle secrets and re-run)'); CFG['corpora'].remove(c); continue
        df.to_csv(p, index=False)
    frames.append(df)
meta = pd.concat(frames, ignore_index=True).sort_values(['corpus','dialogue_id','utterance_id']).reset_index(drop=True)
for c in CFG['corpora']: print(c, meta[meta.corpus==c].groupby(['split','label']).size().unstack(fill_value=0), sep='\\n')""")

md("""## 3. Frozen embeddings (cached once per corpus)
WavLM-Large: all 25 hidden states are kept per utterance (mean-pooled over time) so the softmax layer weights can be *learned* in the model (§III-A) without re-running the encoder. Text: masked mean-pool of RoBERTa-base; MiniLM for the compact branch.""")

code("""from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor

@torch.no_grad()
def embed_audio_layers(df, model_name, bs=4, max_sec=12):
    fe = AutoFeatureExtractor.from_pretrained(model_name); m = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    L = m.config.num_hidden_layers + 1; out = np.zeros((len(df), L, m.config.hidden_size), np.float32)
    wavs = df['wav'].tolist()
    for i in tqdm(range(0, len(wavs), bs), desc=f'audio {model_name.split(\"/\")[-1]}'):
        batch = []
        for w in wavs[i:i+bs]:
            try:
                x, sr = sf.read(w, dtype='float32'); x = x.mean(1) if x.ndim > 1 else x
                if sr != 16000:
                    import torchaudio; x = torchaudio.functional.resample(torch.tensor(x), sr, 16000).numpy()
                x = x[:16000*max_sec]; x = np.pad(x, (0, max(0, 1600-len(x))))
            except Exception: x = np.zeros(16000, np.float32)
            batch.append(x)
        inp = fe(batch, sampling_rate=16000, return_tensors='pt', padding=True)
        hs = m(inp.input_values.to(DEVICE), output_hidden_states=True).hidden_states     # L x (B,T,D)
        out[i:i+len(batch)] = torch.stack([h.mean(1) for h in hs], 1).cpu().numpy()
    del m; torch.cuda.empty_cache(); return out

@torch.no_grad()
def embed_text(df, model_name, bs=64):
    tok = AutoTokenizer.from_pretrained(model_name); m = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    texts = df['text'].fillna('').astype(str).tolist(); out = np.zeros((len(df), m.config.hidden_size), np.float32)
    for i in tqdm(range(0, len(texts), bs), desc=f'text {model_name.split(\"/\")[-1]}'):
        inp = tok(texts[i:i+bs], return_tensors='pt', padding=True, truncation=True, max_length=96).to(DEVICE)
        h = m(**inp).last_hidden_state; mk = inp.attention_mask.unsqueeze(-1)
        out[i:i+bs] = ((h*mk).sum(1)/mk.sum(1)).cpu().numpy()
    del m; torch.cuda.empty_cache(); return out

def cached(name, fn):
    p = f'{CACHE}/{name}.npy'
    if os.path.exists(p): return np.load(p)
    a = fn(); np.save(p, a); return a

tag = lambda m: m.split('/')[-1]
def assemble(parts, shape_tail):
    \"\"\"Place each corpus's array on ITS OWN ROWS of meta. Never concatenate: meta is sorted by corpus NAME, not config order.\"\"\"
    out = np.zeros((len(meta), *shape_tail), np.float32)
    for c, arr in parts.items():
        if arr is not None: out[(meta.corpus == c).values] = arr
    return out

def audio_for(c, model_name):
    sub = meta[meta.corpus == c]; have = (sub['wav'] != '').mean()
    if have >= 0.8:                       # a handful of MELD clips are corrupt in the official release; they get a zero acoustic vector
        miss = int((sub['wav'] == '').sum())
        if miss: print(f'{c}: {miss} utterance(s) without audio ({100*(1-have):.2f}%) -> zero acoustic vector for those rows')
        return cached(f'audio_{tag(model_name)}_{c}', lambda: embed_audio_layers(sub, model_name))
    print(f'{c}: audio present for only {100*have:.1f}% of utterances -> zero acoustic branch (text-only)'); return None

_parts = {c: audio_for(c, CFG['audio_model']) for c in CFG['corpora']}
_ref = next((v for v in _parts.values() if v is not None), None)
A_L = assemble(_parts, _ref.shape[1:] if _ref is not None else (1, 8))
HAS_AUDIO = {c: _parts[c] is not None for c in CFG['corpora']}
print('=' * 70); print('AUDIO AVAILABLE PER CORPUS:', HAS_AUDIO)
for c, ok in HAS_AUDIO.items():
    if not ok and c != 'emorynlp': print(f'  !! {c}: running TEXT-ONLY. Expected files not found under {DATA} (MELD: MELD.Raw.tar.gz or a folder containing train_splits/*.mp4; IEMOCAP: .../SessionN/sentences/wav)')
print('=' * 70)
_tp = {c: cached(f'text_{tag(CFG[\"text_model\"])}_{c}', lambda c=c: embed_text(meta[meta.corpus==c], CFG['text_model'])) for c in CFG['corpora']}
T = assemble(_tp, (next(iter(_tp.values())).shape[1],))
for c in CFG['corpora']: assert (meta.corpus == c).sum() == len(_tp[c]), f'{c}: cache size mismatch -> delete cache/text_*_{c}.npy and re-run'
print('audio layers', A_L.shape, '| text', T.shape)""")

md("""## 4. Hardness h(t) — target-label-free neighbourhood entropy (§III-B)
Computed **per fold**: neighbours come only from that fold's training set. For a training utterance, its own dialogue (and on IEMOCAP its own speaker) is excluded so train-h and test-h are the same quantity. Returns h ∈ [0,1].""")

code("""def fold_splits(corpus):
    sub = meta[meta.corpus == corpus]
    if corpus == 'iemocap':
        tr_all = sub[sub.split=='train']; te = sub[sub.split=='test'].index.values
        dv = tr_all[tr_all.dialogue_id.isin(tr_all.dialogue_id.drop_duplicates().sample(frac=0.1, random_state=0))].index.values
        yield from _nonempty('fixed_Ses05', np.setdiff1d(tr_all.index.values, dv), dv, te)   # primary (Table I)
        for s in range(1,5):                                                          # LOSO secondary
            te = sub[sub.session==f'Ses0{s}'].index.values; rest = sub[sub.session!=f'Ses0{s}']
            if len(rest) == 0 or len(te) == 0: print(f'  !! loso_Ses0{s}: train={len(rest)} test={len(te)} -> fold skipped (IEMOCAP metadata incomplete)'); continue
            dv = rest[rest.dialogue_id.isin(rest.dialogue_id.drop_duplicates().sample(frac=0.1, random_state=s))].index.values
            yield from _nonempty(f'loso_Ses0{s}', np.setdiff1d(rest.index.values, dv), dv, te)
    else:
        yield from _nonempty('official', sub[sub.split=='train'].index.values, sub[sub.split=='dev'].index.values, sub[sub.split=='test'].index.values)

def _nonempty(name, tr, dv, te):
    if min(len(tr), len(dv), len(te)) == 0: print(f'  !! fold {name}: train={len(tr)} dev={len(dv)} test={len(te)} -> fold skipped'); return
    yield name, tr, dv, te

def hardness(F_, y, tr, K, corpus):
    \"\"\"h for every row in meta; neighbours restricted to tr; own dialogue/speaker excluded for rows in tr.\"\"\"
    Xn = F_ / (np.linalg.norm(F_, axis=1, keepdims=True) + 1e-8)
    dia, spk = meta['dialogue_id'].values, meta['speaker'].values
    assert len(tr) > 0, f'hardness({corpus}): empty training fold'
    C = len(LABELS[corpus]); nn_ = NearestNeighbors(n_neighbors=max(1, min(len(tr), K + 200)), metric='cosine').fit(Xn[tr])
    _, idx = nn_.kneighbors(Xn); h = np.full(len(F_), np.nan, np.float32); tr_set = set(tr)
    for i in range(len(F_)):
        if meta.corpus.values[i] != corpus: continue
        cand = tr[idx[i]]
        if i in tr_set:
            mask = (cand != i) & (dia[cand] != dia[i])
            if corpus == 'iemocap': mask &= (spk[cand] != spk[i])
            cand = cand[mask]
        cand = cand[:K]
        p = np.bincount(y[cand], minlength=C) / max(len(cand), 1); p = p[p > 0]
        h[i] = float(-(p * np.log(p)).sum() / np.log(C))
    return h

y_all = np.full(len(meta), -1);
for c in CFG['corpora']:
    m_ = meta.corpus == c; y_all[m_] = meta.loc[m_, 'label'].map({l:i for i,l in enumerate(LABELS[c])}).values
F_mean = np.concatenate([A_L.mean(1), T], 1)           # fixed fused embedding for kNN (layer-averaged audio + text)""")

md("""## 5. Model (§III-C, III-E, III-F)
Learned softmax layer weights over WavLM layers → fuse with text → z-score (training stats) → learned projection (PCA-init) → **FiLM gate** by h → attention over the selected window with positional embedding → residual to z(t) → MLP. Gate modes: `film` (ours), `concat`, `random` (matched marginal), `none`.""")

code("""class HardnessERC(nn.Module):
    def __init__(self, n_layers, d_audio, d_text, H, n_cls, kmax=8, gate='film', head='mlp', pca_init=None, ctx='attn'):
        super().__init__()
        self.layer_w = nn.Parameter(torch.zeros(n_layers)); self.gate, self.head_kind, self.ctx = gate, head, ctx
        self.spk = nn.Embedding(2, H)                                       # same-speaker / other-speaker flag per slot
        self.gru = nn.GRU(H, H // 2, batch_first=True, bidirectional=True) if ctx == 'gru' else None
        d = d_audio + d_text + (1 if gate == 'concat' else 0)
        self.proj = nn.Linear(d_audio + d_text, H)
        if pca_init is not None:
            with torch.no_grad(): self.proj.weight.copy_(torch.tensor(pca_init.T)); self.proj.bias.zero_()
        self.gamma = nn.Parameter(torch.zeros(H)); self.beta = nn.Parameter(torch.zeros(H))
        self.concat = nn.Linear(H + 1, H) if gate == 'concat' else None
        self.pos = nn.Embedding(kmax + 1, H); self.attn = nn.MultiheadAttention(H, 4, batch_first=True); self.q = nn.Parameter(torch.randn(1, 1, H) * 0.02)
        self.conv = nn.Sequential(nn.Conv1d(1, 16, 5, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(H // 4), nn.Flatten(), nn.Linear(16 * (H // 4), H)) if head == 'conv' else None
        self.cls = nn.Sequential(nn.LayerNorm(H), nn.Dropout(0.3), nn.Linear(H, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, n_cls))
        self.kpred = nn.Sequential(nn.Linear(H + 1, 64), nn.GELU(), nn.Linear(64, 6))   # 5 lengths + unresolvable
    def encode(self, A, Tx, h):                          # A: (N,L,Da) Tx: (N,Dt) h: (N,)
        a = (A * self.layer_w.softmax(0)[None, :, None]).sum(1)
        z = self.proj(torch.cat([a, Tx], 1))
        if self.gate in ('film', 'random'): z = (1 + self.gamma * h[:, None]) * z + self.beta * h[:, None]
        elif self.gate == 'concat': z = self.concat(torch.cat([z, h[:, None]], 1))
        if self.conv is not None: z = self.conv(z[:, None, :])
        return z
    def forward(self, Aw, Tw, hw, M, S=None):             # windows: (B,L,...) ; M (B,L) valid mask, last = current; S (B,L) same-speaker flag
        B, L = M.shape
        z = self.encode(Aw.flatten(0, 1), Tw.flatten(0, 1), hw.flatten(0, 1)).view(B, L, -1) + self.pos(torch.arange(L, device=M.device))[None]
        if S is not None: z = z + self.spk(S.long())
        if self.ctx == 'gru':                                 # speaker-aware bidirectional GRU over the (masked) window; context = state at the current slot
            zm = z * M[:, :, None]; g_out, _ = self.gru(zm); ctx = g_out[:, -1:, :]
        else:
            ctx, _ = self.attn(self.q.expand(B, -1, -1), z, z, key_padding_mask=(M == 0))
        cur = z[:, -1]; out = self.cls(cur + ctx[:, 0])
        klog = self.kpred(torch.cat([cur, hw[:, -1, None]], 1))
        return out, klog""")

code("""POS_IN_DIA = meta.groupby('dialogue_id').cumcount().values; DIA = meta['dialogue_id'].values; SPK = meta['speaker'].astype(str).values
def windows_for(rows, k):
    \"\"\"indices (len(rows), k+1) of context utterances within the same dialogue; -1 = pad.\"\"\"
    W = np.full((len(rows), k + 1), -1, np.int64); pos_in_dia, dia = POS_IN_DIA, DIA
    for n, i in enumerate(rows):
        for j in range(k + 1):
            src = i - (k - j)
            if src >= 0 and dia[src] == dia[i] and pos_in_dia[i] - pos_in_dia[src] == (k - j): W[n, j] = src
    return W

KMAX = 8   # every window is KMAX+1 slots long; the current utterance is always the last slot, shorter k is masked
class DS(torch.utils.data.Dataset):
    def __init__(self, rows, kvec, A_t, T_t, h_t, y, kstar=None):
        self.rows, self.kvec, self.A, self.T, self.h, self.y, self.ks = rows, kvec, A_t, T_t, h_t, y, kstar
        self.kmax = KMAX; self.rng = np.random.RandomState(len(rows))      # kvec=None -> 'mixed' training: k drawn uniformly from KS at every access
    def __len__(self): return len(self.rows)
    def __getitem__(self, n):
        i = self.rows[n]; k = int(self.rng.choice(KS)) if self.kvec is None else int(self.kvec[n]); idx = windows_for(np.array([i]), self.kmax)[0]
        M = (idx >= 0).astype(np.float32); M[:self.kmax - k] = 0            # mask beyond selected k
        idx = np.where(idx < 0, i, idx)
        S = (SPK[idx] == SPK[i]).astype(np.float32)
        ks = -1 if self.ks is None else int(self.ks[i])
        return self.A[idx], self.T[idx], self.h[idx], torch.tensor(M), int(self.y[i]), float(self.h[i]), ks, torch.tensor(S)

def metrics(y, p):
    return dict(WF1=f1_score(y, p, average='weighted')*100, MF1=f1_score(y, p, average='macro')*100, UA=balanced_accuracy_score(y, p)*100)

def train_eval(corpus, tr, dv, te, kvec_fn, A_t, T_t, h_t, y, seed, gate='film', head='mlp', lam=0.5, kstar=None, epochs=None, return_probs=False, ctx='attn'):
    \"\"\"kvec_fn(rows) -> per-utterance context length. Returns dev/test metrics (+ probs, k-logits, oracle-class preds).\"\"\"
    seed_all(seed); epochs = epochs or CFG['epochs']; n_cls = len(LABELS[corpus])
    # z-score every feature dimension with TRAINING-FOLD statistics (paper §III-A); frozen encoders have outlier dims that otherwise swamp the projection
    A_t = ((A_t - A_t[tr].mean(0, keepdims=True)) / (A_t[tr].std(0, keepdims=True) + 1e-6)).astype(np.float32)
    T_t = ((T_t - T_t[tr].mean(0, keepdims=True)) / (T_t[tr].std(0, keepdims=True) + 1e-6)).astype(np.float32)
    Ztr = np.concatenate([A_t[tr].mean(1), T_t[tr]], 1); pca = PCA(CFG['H'], random_state=0).fit(Ztr)
    model = HardnessERC(A_t.shape[1], A_t.shape[2], T_t.shape[1], CFG['H'], n_cls, gate=gate, head=head, pca_init=pca.components_.T, ctx=ctx).to(DEVICE)
    if gate == 'random':                                   # matched-marginal random gate
        rng = np.random.RandomState(seed); h_t = h_t.copy(); h_t[tr] = rng.permutation(h_t[tr]); h_t[dv] = rng.permutation(h_t[dv]); h_t[te] = rng.permutation(h_t[te])
    kv_of = lambda rows: None if kvec_fn == 'mixed' else kvec_fn(rows)          # 'mixed' = window-agnostic training (uniform random k per access)
    mk = lambda rows, sh, kv=None: torch.utils.data.DataLoader(DS(rows, kv, torch.tensor(A_t), torch.tensor(T_t), torch.tensor(h_t), y, kstar), batch_size=CFG['batch'], shuffle=sh)
    dl = {'tr': mk(tr, True, kv_of(tr)), 'dv': mk(dv, False, kv_of(dv)), 'te': mk(te, False, kv_of(te))}
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
    def run(split, train=False, loader=None):
        model.train(train); ys, ps, probs, klogs = [], [], [], []
        for Aw, Tw, hw, M, yb, hb, ks, S in (loader if loader is not None else dl[split]):
            Aw, Tw, hw, M, yb, hb, ks, S = [t.to(DEVICE) for t in (Aw, Tw, hw, M, yb, hb, ks, S)]
            with torch.set_grad_enabled(train):
                out, klog = model(Aw, Tw, hw, M, S)
                loss = (F.cross_entropy(out, yb, reduction='none') * (1 + lam * hb)).mean()
                if kstar is not None: loss = loss + 0.3 * F.cross_entropy(klog[ks >= 0], ks[ks >= 0]) if (ks >= 0).any() else loss
                if train: opt.zero_grad(); loss.backward(); opt.step()
            ys += yb.tolist(); ps += out.argmax(1).tolist(); probs.append(out.softmax(1).detach().cpu().numpy()); klogs.append(klog.detach().cpu().numpy())
        return np.array(ys), np.array(ps), np.concatenate(probs), np.concatenate(klogs)
    best, state = -1, None
    for ep in range(epochs):
        run('tr', True); yd, pd_, _, _ = run('dv'); m_ = metrics(yd, pd_)['WF1']
        if m_ > best: best, state = m_, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state); yt, pt, prt, klt = run('te')
    res = dict(dev_WF1=best, **{f'test_{a}': b for a, b in metrics(yt, pt).items()})
    if return_probs:                                   # posterior entropy + k-logits for EVERY row (entropy-k baseline, k-predictor) - unshuffled loaders
        ent, klog_all = {}, {}
        for rows in (tr, dv, te):
            kv = None if kvec_fn == 'mixed' else np.full(len(rows), 8) if kvec_fn is None else kvec_fn(rows)
            _, _, pr, kl = run('x', loader=mk(rows, False, kv)); ent.update(dict(zip(rows, -(pr * np.log(pr + 1e-9)).sum(1)))); klog_all.update(dict(zip(rows, kl)))
        train_eval.last_entropy = ent; train_eval.last_klog = klog_all
        train_eval.last_model = model; train_eval.last_arrays = (A_t, T_t, h_t, y, kstar)
        return res, yt, pt, prt, klt, model
    return res

def eval_with_k(rows, kvec):
    \"\"\"Inference only: last trained model, given per-row context lengths. Returns metrics dict.\"\"\"
    model = train_eval.last_model; A_t, T_t, h_t, y, kstar = train_eval.last_arrays
    dl_ = torch.utils.data.DataLoader(DS(rows, kvec, torch.tensor(A_t), torch.tensor(T_t), torch.tensor(h_t), y, kstar), batch_size=CFG['batch'], shuffle=False)
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for Aw, Tw, hw, M, yb, hb, ks, S in dl_:
            out, _ = model(*[t.to(DEVICE) for t in (Aw, Tw, hw, M, S)]); ys += yb.tolist(); ps += out.argmax(1).tolist()
    return metrics(np.array(ys), np.array(ps))

def k_from_klog(klog, tau):
    \"\"\"Decision rule: predicted-unresolvable -> k=8 (max context); else the smallest k whose predicted P(k* <= k | resolvable) >= tau.\"\"\"
    p = np.exp(klog - klog.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    unres = p[:, 5] >= p[:, :5].max(1); pr = p[:, :5] / (p[:, :5].sum(1, keepdims=True) + 1e-9); cum = np.cumsum(pr, 1)
    j = (cum >= tau).argmax(1); k = np.array(KS)[j]; k[unres] = 8; return k""")

md("""## 6. Oracle k* (§III-D) — margin-based, monotone, with an *unresolvable* class
Inside each training fold: inner 5-fold CV; for each k, fixed-window models with `oracle_seeds` seeds; margin = p(true) − max p(other), averaged over seeds. k*(t) = smallest k such that margin > τ for **all** k′ ≥ k. Never above τ → class 5 (*unresolvable*).""")

code("""KS = CFG['windows']
def oracle_kstar(corpus, tr, A_t, T_t, h_t, y):
    p = f'{CACHE}/kstar_{corpus}_{len(tr)}_{int(np.sum(tr[:50]))}.npy'     # deterministic cache key (hash() changes per session)
    if os.path.exists(p):
        ks = np.load(p); oracle_kstar.last = (ks, h_t, tr); return ks
    margin = np.full((len(meta), len(KS)), np.nan, np.float32)
    dias = meta.loc[tr, 'dialogue_id'].unique(); rng = np.random.RandomState(0); rng.shuffle(dias); folds = np.array_split(dias, 5)
    for fi, fd in enumerate(folds):
        inner_te = tr[meta.loc[tr, 'dialogue_id'].isin(fd).values]; inner_tr = np.setdiff1d(tr, inner_te)
        inner_dv = inner_tr[:max(1, len(inner_tr)//10)]; inner_tr = inner_tr[len(inner_tr)//10:]
        for ki, k in enumerate(KS):
            ms = []
            for s in CFG['oracle_seeds']:
                _, yt, _, prt, _, _ = train_eval(corpus, inner_tr, inner_dv, inner_te, lambda rows, k=k: np.full(len(rows), k), A_t, T_t, h_t, y, s, lam=0.0, epochs=15, return_probs=True)
                pt_true = prt[np.arange(len(yt)), yt]; oth = prt.copy(); oth[np.arange(len(yt)), yt] = -1
                ms.append(pt_true - oth.max(1))
            margin[inner_te, ki] = np.mean(ms, 0)
        print(f'  oracle fold {fi+1}/5 done')
    ks = np.full(len(meta), -1, np.int64)
    for i in tr:
        ok = margin[i] > CFG['margin_tau']
        good = [j for j in range(len(KS)) if ok[j:].all()]
        ks[i] = good[0] if good else 5
    np.save(p, ks); oracle_kstar.last = (ks, h_t, tr); return ks

def kstar_by_quartile(corpus):
    \"\"\"Diagnostic: does oracle context need rise with hardness, even before any predictor is trained?\"\"\"
    if not hasattr(oracle_kstar, 'last'): print('oracle k* not computed yet in this session -> run the hardness policy first'); return None
    ks, h_t, tr = oracle_kstar.last
    q = pd.qcut(stats.rankdata(h_t[tr], method='ordinal'), 4, labels=['Q1','Q2','Q3','Q4'])
    rows = []
    for Q in ['Q1','Q2','Q3','Q4']:
        m_ = (np.asarray(q) == Q); kk = ks[tr][m_]
        rows.append(dict(quartile=Q, mean_h=h_t[tr][m_].mean(), unresolvable=(kk==5).mean()*100, **{f'k*={KS[j]}': (kk==j).mean()*100 for j in range(5)}, mean_kstar_resolved=np.mean([KS[j] for j in kk if j < 5]) if (kk<5).any() else np.nan))
    df = pd.DataFrame(rows).round(2); df.to_csv(f'{RESULTS}/kstar_by_quartile_{corpus}.csv', index=False); print(f'oracle k* by hardness quartile [{corpus}]'); print(df); return df""")

md("""### Sanity check: linear probe on the frozen text features
Before any experiment, a logistic-regression probe on z-scored RoBERTa features must clear the majority-class baseline by a wide margin (MELD: majority ≈ 31 W-F1; a text probe should reach ≈ 50–58). If it does not, the embeddings are wrong — stop and fix that first.""")
code("""from sklearn.linear_model import LogisticRegression
for corpus in CFG['corpora']:
    _, tr, dv, te = next(fold_splits(corpus))
    if len(tr) == 0 or len(te) == 0:
        print(f'!! {corpus}: train={len(tr)} test={len(te)} rows - a split is EMPTY; fix the data before running experiments'); continue
    mu, sd = T[tr].mean(0), T[tr].std(0) + 1e-6; Xtr, Xte = (T[tr]-mu)/sd, (T[te]-mu)/sd
    clf = LogisticRegression(max_iter=2000, C=0.5, class_weight=None).fit(Xtr, y_all[tr]); p = clf.predict(Xte)
    maj = np.full(len(te), np.bincount(y_all[tr]).argmax())
    print(f"{corpus}: text linear probe W-F1 = {f1_score(y_all[te], p, average='weighted')*100:.1f} | majority-class W-F1 = {f1_score(y_all[te], maj, average='weighted')*100:.1f}")""")

md("""## 7. Policies and experiments → Tables I–IV
Policies, all with identical head and parameters: fixed k; `random` at matched k̄; `entropy` (posterior entropy of the k=0 model → k by quantile mapping); `hardness` (predicted by g, ours); `oracle` (k*, test-time cheat, upper bound).

**`hardness` policy (rev. 4 — fixes the train/test window mismatch of the 28 Oct run):** pass 1 trains a *window-agnostic* classifier (k drawn uniformly from {0,1,2,4,8} at every access) jointly with the k-predictor g on oracle k*. Decision rule: predicted *unresolvable* → k = 8; otherwise the smallest k with predicted P(k* ≤ k | resolvable) ≥ τ, τ ∈ `tau_grid` chosen on the inner dev split. The pass-1 model also yields, by inference alone, a W-F1-vs-k̄ curve for hardness-k and for random-k at the *same* k̄ (columns `curve_*`). Pass 2 retrains the classifier under its own policy (k assigned to every utterance, train and test alike) and reports that as the policy's test score — the same protocol the `entropy` and `random` policies use.""")

code("""def run_config(corpus, name, policy, gate='film', head='mlp', lam=0.5, audio=None, text=None, folds=None, ctx='attn'):
    done = f'{RESULTS}/{name}_{corpus}.csv'
    if os.path.exists(done) and os.path.exists(done.replace('.csv', '_pooled.npz')):   # resume after a disconnect: reuse finished experiments
        df = pd.read_csv(done); stale = None
        a_cache = f'{CACHE}/audio_{tag(CFG[\"audio_model\"])}_{corpus}.npy'
        if policy == 'hardness' and 'tau' not in df.columns: stale = 'old_policy_28oct'          # produced by the old (mismatched) policy
        elif HAS_AUDIO.get(corpus) and os.path.exists(a_cache) and os.path.getmtime(done) < os.path.getmtime(a_cache): stale = 'text_only_pilot'   # computed before audio existed
        elif policy == 'random' and os.path.exists(f'{RESULTS}/T2_hardness_{corpus}.csv'):
            hk = pd.read_csv(f'{RESULTS}/T2_hardness_{corpus}.csv').get('kbar'); rk = df.get('kbar')
            if hk is not None and (rk is None or abs(float(rk.mean()) - float(hk.mean())) > 0.05): stale = 'old_policy_28oct'   # random-k must use the CURRENT policy's k-bar
        if not stale and policy == 'hardness' and 'kbar' in df.columns:
            for fold_, g_ in df.groupby('fold'): RUN_STATE[f'{corpus}_{fold_}_kbar'] = float(g_.kbar.mean())      # so a later random-k run is matched to this policy
        if stale:
            os.makedirs(f'{RESULTS}/{stale}', exist_ok=True)
            for f_ in (done, done.replace('.csv', '_pooled.npz')): shutil.move(f_, f'{RESULTS}/{stale}/{os.path.basename(f_)}')
            print(f'== {name} [{corpus}]: stale checkpoint moved to results/{stale} -> re-running')
        else: print(f'== {name} [{corpus}] loaded from checkpoint (WF1 {df.test_WF1.mean():.2f})'); return df
    A_t = audio if audio is not None else A_L; T_t = text if text is not None else T
    recs, pooled = [], collections.defaultdict(list)
    for fold, tr, dv, te in fold_splits(corpus):
        if folds and fold not in folds: continue
        # --- hardness with K selected on inner dev (by dev WF1 of a quick k=0 model) ---
        bestK, bestF, h_t = None, -1, None
        for K in CFG['K_grid']:
            h_K = hardness(F_mean, y_all, tr, K, corpus); h_K = np.nan_to_num(h_K)
            r = train_eval(corpus, tr, dv, te, lambda rows: np.zeros(len(rows)), A_t, T_t, h_K, y_all, 0, gate=gate, lam=lam, epochs=8)
            if r['dev_WF1'] > bestF: bestK, bestF, h_t = K, r['dev_WF1'], h_K
        ks = oracle_kstar(corpus, tr, A_t, T_t, h_t, y_all) if policy in ('hardness', 'oracle') else None
        for seed in CFG['seeds']:
            if policy.startswith('fixed'):
                k = int(policy.split('_')[1]); kfn = lambda rows, k=k: np.full(len(rows), k); kstar = None
            elif policy == 'oracle':
                kfn = lambda rows: np.array([KS[min(ks[i], 4)] if ks[i] >= 0 else 8 for i in rows]); kstar = None   # test rows have no k*; this policy is reported only where ks exists
            elif policy == 'hardness':
                kstar = ks; kfn = 'mixed'        # pass 1: window-agnostic classifier (uniform random k) + k-predictor g trained on k*; pass 2 below
            elif policy == 'random':
                rng = np.random.RandomState(seed); kbar = RUN_STATE.get(f'{corpus}_{fold}_kbar', 4.0)
                kfn = lambda rows, rng=rng, kbar=kbar: rng.choice(KS, size=len(rows), p=kbar_dist(kbar)); kstar = None; RUN_STATE['_last_random_kbar'] = kbar
            elif policy == 'entropy':
                ent = RUN_STATE.get(f'{corpus}_{fold}_{seed}_ent')
                kfn = lambda rows, ent=ent: np.array([KS[min(4, int(np.searchsorted(ENT_Q, ent.get(i, np.median(list(ent.values()))))))] for i in rows]) if ent is not None else np.full(len(rows), 4); kstar = None
            res, yt, pt, prt, klt, model = train_eval(corpus, tr, dv, te, kfn, A_t, T_t, h_t, y_all, seed, gate=gate, head=head, lam=lam, kstar=kstar, return_probs=True, ctx=ctx)
            if policy == 'hardness':
                # --- tau selected on DEV (inference only, pass-1 model); curve of WF1 vs realised kbar for hardness-k and random-k at the same kbar ---
                KL = train_eval.last_klog; best_tau, best_dv = None, -1; rng_ = np.random.RandomState(seed)
                for tau in CFG.get('tau_grid', [0.5, 0.7, 0.9]):
                    k_dv = k_from_klog(np.stack([KL[i] for i in dv]), tau); k_te = k_from_klog(np.stack([KL[i] for i in te]), tau)
                    m_dv = eval_with_k(dv, k_dv)['WF1']; m_te = eval_with_k(te, k_te); kb = float(k_te.mean())
                    m_rd = eval_with_k(te, rng_.choice(KS, size=len(te), p=kbar_dist(kb)))
                    res.update({f'curve_tau{tau}_kbar': kb, f'curve_tau{tau}_WF1': m_te['WF1'], f'curve_tau{tau}_randWF1': m_rd['WF1'], f'curve_tau{tau}_devWF1': m_dv})
                    if m_dv > best_dv: best_dv, best_tau = m_dv, tau
                res['curve_k8_WF1'] = eval_with_k(te, np.full(len(te), 8))['WF1']; res['curve_k0_WF1'] = eval_with_k(te, np.zeros(len(te), int))['WF1']
                # --- pass 2: the policy assigns k to EVERY utterance (train, dev, test alike); classifier retrained under its own policy ---
                kv_all = {i: int(k) for i, k in zip(KL.keys(), k_from_klog(np.stack(list(KL.values())), best_tau))}
                kvec_te = np.array([kv_all[i] for i in te]); RUN_STATE[f'{corpus}_{fold}_kbar'] = float(kvec_te.mean())
                res2, yt, pt, prt, _, _ = train_eval(corpus, tr, dv, te, lambda rows, kv=kv_all: np.array([kv.get(i, 8) for i in rows]), A_t, T_t, h_t, y_all, seed, gate=gate, head=head, lam=lam, kstar=None, return_probs=True, ctx=ctx)
                res.update({k_: v for k_, v in res2.items() if k_.startswith('test')}); res['dev_WF1_pass1_mixed'] = res['dev_WF1']; res['dev_WF1'] = res2['dev_WF1']
                res['kbar'] = float(kvec_te.mean()); res['tau'] = best_tau
                res['unresolvable_frac'] = float((np.stack([KL[i] for i in te]).argmax(1) == 5).mean())
            if policy == 'fixed_0':                                       # store posterior entropy (all rows) for the entropy policy
                RUN_STATE[f'{corpus}_{fold}_{seed}_ent'] = train_eval.last_entropy
            if policy == 'random': res['kbar'] = RUN_STATE.get('_last_random_kbar')
            recs.append(dict(exp=name, corpus=corpus, fold=fold, seed=seed, policy=policy, gate=gate, head=head, ctx=ctx, lam=lam, K=bestK, **res))
            pooled['y'] += yt.tolist(); pooled['p'] += pt.tolist(); pooled['prob'].append(prt); pooled['h'] += h_t[te].tolist(); pooled['te'] += list(te)
    if not recs: print(f'== {name} [{corpus}]: no usable fold -> skipped'); return None
    df = pd.DataFrame(recs); df.to_csv(f'{RESULTS}/{name}_{corpus}.csv', index=False)
    np.savez(f'{RESULTS}/{name}_{corpus}_pooled.npz', y=np.array(pooled['y']), p=np.array(pooled['p']), prob=np.concatenate(pooled['prob']), h=np.array(pooled['h']), te=np.array(pooled['te']))
    print(f'== {name} [{corpus}] policy={policy} ctx={ctx} gate={gate} lam={lam}'); print(df[['dev_WF1','test_WF1','test_MF1','test_UA']].agg(['mean','std']).round(2)); return df

RUN_STATE = {}
ENT_Q = np.array([0.2, 0.4, 0.6, 0.8])     # placeholder quantiles; replaced after fixed_0 runs
def kbar_dist(kbar):
    # distribution over KS with mean kbar (linear interpolation between adjacent lengths)
    ks = np.array(KS, float); j = np.searchsorted(ks, kbar); j = min(max(j, 1), len(ks)-1)
    w = (kbar - ks[j-1]) / (ks[j] - ks[j-1]); p = np.zeros(len(ks)); p[j-1], p[j] = 1-w, w; return p""")

code("""ALL = []
for corpus in CFG['corpora']:
    primary = ['fixed_Ses05'] if corpus == 'iemocap' else None
    if corpus == 'iemocap':                                   # secondary, speaker-independent LOSO for the three headline configs
        for nm, pol in (('LOSO_fixed0', 'fixed_0'), ('LOSO_fixed8', 'fixed_8')): ALL.append(run_config(corpus, nm, pol, folds=[f'loso_Ses0{s_}' for s_ in range(1,5)]))
    # Table II: fixed windows
    for k in KS: ALL.append(run_config(corpus, f'T2_fixed{k}', f'fixed_{k}', folds=primary))
    _ents = [np.array(list(v.values())) for k_, v in RUN_STATE.items() if k_.endswith('_ent')]
    if not _ents and os.path.exists(f'{RESULTS}/T2_fixed0_{corpus}_pooled.npz'):   # fixed_0 came from checkpoint: rebuild entropies from its pooled test probabilities
        z0 = np.load(f'{RESULTS}/T2_fixed0_{corpus}_pooled.npz'); e0 = -(z0['prob'] * np.log(z0['prob'] + 1e-9)).sum(1)
        for seed in CFG['seeds']: RUN_STATE[f'{corpus}_{next(fold_splits(corpus))[0]}_{seed}_ent'] = dict(zip(z0['te'], e0))
        _ents = [e0]
    ENT_Q = np.quantile(np.concatenate(_ents), [0.2, 0.4, 0.6, 0.8])
    # ours, then matched baselines
    ALL.append(run_config(corpus, 'T2_hardness', 'hardness', folds=primary)); kstar_by_quartile(corpus)
    ALL.append(run_config(corpus, 'T2_random_kbar', 'random', folds=primary))
    ALL.append(run_config(corpus, 'T2_entropy', 'entropy', folds=primary))
    # stronger, speaker-aware context encoder (bidirectional GRU): does ANY encoder extract context from these features?
    for k in [0, 8]: ALL.append(run_config(corpus, f'T2g_fixed{k}', f'fixed_{k}', ctx='gru', folds=primary))
    ALL.append(run_config(corpus, 'T2g_hardness', 'hardness', ctx='gru', folds=primary))
    # Table IV ablations (hardness policy)
    ALL.append(run_config(corpus, 'T4_concat', 'hardness', gate='concat', folds=primary))
    ALL.append(run_config(corpus, 'T4_randomgate', 'hardness', gate='random', folds=primary))
    ALL.append(run_config(corpus, 'T4_nogate', 'hardness', gate='none', folds=primary))
    ALL.append(run_config(corpus, 'T4_convhead', 'hardness', head='conv', folds=primary))
    for lam in CFG['lam_grid']: ALL.append(run_config(corpus, f'T4_lam{lam}', 'hardness', lam=lam, folds=primary))
res = pd.concat([a for a in ALL if a is not None]); summ = res.groupby(['exp','corpus']).agg(dev_WF1=('dev_WF1','mean'), WF1=('test_WF1','mean'), WF1_std=('test_WF1','std'), MF1=('test_MF1','mean'), UA=('test_UA','mean'), kbar=('kbar','mean')).round(2)
summ.to_csv(f'{RESULTS}/summary_tables_II_IV.csv'); summ""")

code("""# Paired bootstrap 95% CI (seed-paired) of ours vs each reproduced baseline  -> Table I/II CI column
def paired_boot(a, b, n=5000, seed=0):
    rng = np.random.RandomState(seed); d = np.array(a) - np.array(b); bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
ci = []
for corpus in CFG['corpora']:
    ours = res[(res.exp=='T2_hardness') & (res.corpus==corpus)].sort_values(['fold','seed'])['test_WF1'].values
    for e in sorted(res.exp.unique()):
        if e == 'T2_hardness': continue
        b = res[(res.exp==e) & (res.corpus==corpus)].sort_values(['fold','seed'])['test_WF1'].values
        if len(b) == len(ours): m, lo, hi = paired_boot(ours, b); ci.append(dict(corpus=corpus, vs=e, delta=round(m,2), lo=round(lo,2), hi=round(hi,2)))
pd.DataFrame(ci).to_csv(f'{RESULTS}/paired_bootstrap_CIs.csv', index=False); pd.DataFrame(ci)""")

code("""# Table III: context gain and k assignment by hardness quartile (unresolvable excluded)
for corpus in CFG['corpora']:
    z0 = np.load(f'{RESULTS}/T2_fixed0_{corpus}_pooled.npz'); z8 = np.load(f'{RESULTS}/T2_fixed8_{corpus}_pooled.npz'); zh = np.load(f'{RESULTS}/T2_hardness_{corpus}_pooled.npz')
    h = z0['h']; q = pd.qcut(stats.rankdata(h, method='ordinal'), 4, labels=['Q1','Q2','Q3','Q4']); rows = []   # rank-based: robust to tied h
    for Q in ['Q1','Q2','Q3','Q4']:
        m_ = (q == Q)
        rows.append(dict(quartile=Q, mean_h=h[m_].mean().round(3), WF1_k0=f1_score(z0['y'][m_], z0['p'][m_], average='weighted')*100, WF1_k8=f1_score(z8['y'][m_], z8['p'][m_], average='weighted')*100, WF1_hardness=f1_score(zh['y'][m_], zh['p'][m_], average='weighted')*100))
    t3 = pd.DataFrame(rows); t3['gain_k0_to_k8'] = t3.WF1_k8 - t3.WF1_k0
    t3.round(2).to_csv(f'{RESULTS}/table_III_{corpus}.csv', index=False); print(corpus); print(t3.round(2))""")

md("""## 8. Hardness as an ambiguity signal → Table VI
Three questions, each answered on pooled test predictions of the k = 0 model (so the signal is judged on its own, not through a context policy):
1. Does h predict errors? ρ(h, error), error rate by h-decile.
2. Does h carry information **beyond the model's own confidence**? partial ρ(h, error | model entropy), and AURC of selective prediction using model confidence, −h, and both.
3. Does h predict **human annotator disagreement**, after controlling for model entropy? (needs `annotators`; IEMOCAP provides it, MELD/EmoryNLP do not.)
A paper-worthy hardness signal needs (2) or (3) clearly positive; (1) alone is expected and is not a contribution.""")
code("""def aurc(score, err):
    o = np.argsort(-score); return np.mean(np.cumsum(err[o]) / np.arange(1, len(err)+1)) * 100
def ece(prob, y, bins=15):
    conf, pred = prob.max(1), prob.argmax(1); acc = (pred == y).astype(float); e = 0
    for lo, hi in zip(np.linspace(0,1,bins+1)[:-1], np.linspace(0,1,bins+1)[1:]):
        m_ = (conf > lo) & (conf <= hi)
        if m_.any(): e += m_.mean() * abs(acc[m_].mean() - conf[m_].mean())
    return e * 100
def partial_spearman(x, y, z):
    rx, ry, rz = [stats.rankdata(v) for v in (x, y, z)]
    rxy, rxz, ryz = [stats.pearsonr(a, b)[0] for a, b in ((rx, ry), (rx, rz), (ry, rz))]
    return (rxy - rxz*ryz) / np.sqrt((1-rxz**2)*(1-ryz**2))
zs = lambda v: (v - v.mean()) / (v.std() + 1e-9)

t6 = []
for corpus in CFG['corpora']:
    z = np.load(f'{RESULTS}/T2_fixed0_{corpus}_pooled.npz'); y, p, pr, h = z['y'], z['p'], z['prob'], z['h']
    err = (p != y).astype(float); conf = pr.max(1); ent = -(pr * np.log(pr + 1e-9)).sum(1)
    row = dict(corpus=corpus, error_rate=err.mean()*100, ECE=ece(pr, y),
               rho_h_err=stats.spearmanr(h, err)[0], rho_ent_err=stats.spearmanr(ent, err)[0], rho_h_ent=stats.spearmanr(h, ent)[0],
               partial_rho_h_err_given_ent=partial_spearman(h, err, ent),
               AURC_model_conf=aurc(conf, err), AURC_hardness=aurc(-h, err), AURC_conf_plus_h=aurc(zs(conf) + zs(-h), err))
    dec = pd.qcut(stats.rankdata(h, method='ordinal'), 10, labels=False)
    row['error_by_h_decile'] = ' '.join(f'{err[dec==d].mean()*100:.0f}' for d in range(10))
    ann = meta.loc[z['te'], 'annotators'].astype(str).values
    if 'agreement' in meta.columns and meta.loc[z['te'], 'agreement'].notna().any():
        dis = 1 - pd.to_numeric(meta.loc[z['te'], 'agreement'], errors='coerce').values; ok = ~np.isnan(dis)
    elif (ann != '').any():
        dis = np.array([1 - collections.Counter(a.split('|')).most_common(1)[0][1] / len(a.split('|')) if a else np.nan for a in ann]); ok = ~np.isnan(dis)
    else: ok = None
    if ok is not None:
        row.update(rho_h_disagree=stats.spearmanr(h[ok], dis[ok])[0], rho_ent_disagree=stats.spearmanr(ent[ok], dis[ok])[0],
                   partial_rho_h_disagree_given_ent=partial_spearman(h[ok], dis[ok], ent[ok]), n_with_annotators=int(ok.sum()))
    t6.append(row)
t6 = pd.DataFrame(t6).round(3); t6.to_csv(f'{RESULTS}/table_VI.csv', index=False); t6.T""")

md("""## 8b. Cross-corpus transfer check → Table V (4-class common set)
Train on one corpus, test on the other, labels mapped to {angry, happy, neutral, sad}; k = 0 and hardness-k. A transfer *check*, not a generalisation claim.""")
code("""C4 = ['angry','happy','neutral','sad']
y4 = np.array([C4.index(COMMON4[l]) if l in COMMON4 else -1 for l in meta['label']])
pairs = [(a, b) for a in CFG['corpora'] for b in CFG['corpora'] if a != b and a != 'emorynlp' and b != 'emorynlp']
t5 = []
LAB_BACKUP, Y_BACKUP = dict(LABELS), y_all.copy()
for src, dst in pairs:
    tr = meta[(meta.corpus==src) & (meta.split=='train') & (y4>=0)].index.values
    dv = meta[(meta.corpus==src) & (meta.split.isin(['dev','test'])) & (y4>=0)].index.values[:800]
    te = meta[(meta.corpus==dst) & (meta.split=='test') & (y4>=0)].index.values
    LABELS[src] = C4; y_all[:] = y4
    h_t = np.nan_to_num(hardness(F_mean, y_all, tr, 10, src))
    r0 = np.mean([train_eval(src, tr, dv, te, lambda rows: np.zeros(len(rows)), A_L, T, h_t, y_all, sd_)['test_WF1'] for sd_ in CFG['seeds'][:3]])
    r8 = np.mean([train_eval(src, tr, dv, te, lambda rows: np.full(len(rows), 8), A_L, T, h_t, y_all, sd_)['test_WF1'] for sd_ in CFG['seeds'][:3]])
    t5.append(dict(train=src, test=dst, WF1_k0=r0, WF1_k8=r8)); print(t5[-1])
LABELS.update(LAB_BACKUP); y_all[:] = Y_BACKUP
if t5: pd.DataFrame(t5).round(2).to_csv(f'{RESULTS}/table_V_cross_corpus.csv', index=False)
else: print('cross-corpus needs both MELD and IEMOCAP in this run -> Table V skipped')
pd.DataFrame(t5).round(2)""")

md("## 9. Compact text branch (MiniLM) and acoustic ablation → Tables IV (row) and VII")
code("""_sp = {c: cached(f'text_{tag(CFG[\"slm_model\"])}_{c}', lambda c=c: embed_text(meta[meta.corpus==c], CFG['slm_model'])) for c in CFG['corpora']}
T_slm = assemble(_sp, (next(iter(_sp.values())).shape[1],))
for corpus in CFG['corpora']:
    ALL.append(run_config(corpus, 'T7_minilm', 'hardness', text=T_slm, folds=['fixed_Ses05'] if corpus=='iemocap' else None))
if any(HAS_AUDIO.values()):
    _p2 = {c: audio_for(c, 'facebook/wav2vec2-base') for c in CFG['corpora']}; _r2 = next(v for v in _p2.values() if v is not None)
    A_w2v = assemble(_p2, _r2.shape[1:])
    for corpus in [c for c in CFG['corpora'] if HAS_AUDIO[c]]:
        ALL.append(run_config(corpus, 'T4_wav2vec2', 'hardness', audio=A_w2v, folds=['fixed_Ses05'] if corpus=='iemocap' else None))
else: print('no audio in any corpus yet -> acoustic ablation skipped')

@torch.no_grad()
def cpu_latency(model_name, n=30):
    tok = AutoTokenizer.from_pretrained(model_name); m = AutoModel.from_pretrained(model_name).eval()
    inp = tok(meta['text'].astype(str).head(5).tolist(), return_tensors='pt', padding=True, truncation=True, max_length=96); m(**inp); t0 = time.time()
    for _ in range(n): m(**inp)
    return (time.time()-t0)/n*1000, sum(p.numel() for p in m.parameters())/1e6
for nm, mdl in [('RoBERTa-base', CFG['text_model']), ('MiniLM-L6', CFG['slm_model'])]:
    ms, P = cpu_latency(mdl); print(f'{nm}: {P:.1f} M params, {ms:.1f} ms / 5-utterance window (CPU)')""")

md("## 10. Paper tables (markdown) — paste into the manuscript")
code("""def md_table(df):
    cols = list(df.columns); out = ['| ' + ' | '.join(map(str, cols)) + ' |', '| ' + ' | '.join(['---']*len(cols)) + ' |']
    for _, r in df.iterrows(): out.append('| ' + ' | '.join(str(v) for v in r.values) + ' |')
    return '\\n'.join(out)
S = pd.read_csv(f'{RESULTS}/summary_tables_II_IV.csv')
print('### Table II — W-F1 vs context policy'); print(md_table(S[S.exp.str.startswith('T2')].pivot(index='exp', columns='corpus', values='WF1').round(2).reset_index()))
print('\\n### Table IV — ablations (hardness policy)'); print(md_table(S[S.exp.str.startswith('T4')].pivot(index='exp', columns='corpus', values='WF1').round(2).reset_index()))
for c in CFG['corpora']:
    p = f'{RESULTS}/table_III_{c}.csv'
    if os.path.exists(p): print(f'\\n### Table III — {c}'); print(md_table(pd.read_csv(p).round(2)))
    p = f'{RESULTS}/kstar_by_quartile_{c}.csv'
    if os.path.exists(p): print(f'\\n### oracle k* by quartile — {c}'); print(md_table(pd.read_csv(p).round(2)))
for c in CFG['corpora']:                                   # Table II-b: W-F1 vs realised kbar, hardness-k vs random-k at the same kbar (pass-1 model, inference only)
    p = f'{RESULTS}/T2_hardness_{c}.csv'
    if os.path.exists(p):
        d_ = pd.read_csv(p); cc = [x for x in d_.columns if x.startswith('curve_tau')]
        if cc:
            rows_ = [dict(tau=t, kbar=d_[f'curve_tau{t}_kbar'].mean(), WF1_hardness_k=d_[f'curve_tau{t}_WF1'].mean(), WF1_random_k_same_kbar=d_[f'curve_tau{t}_randWF1'].mean(), dev_WF1=d_[f'curve_tau{t}_devWF1'].mean()) for t in CFG['tau_grid']]
            rows_.append(dict(tau='k=0', kbar=0, WF1_hardness_k=d_['curve_k0_WF1'].mean())); rows_.append(dict(tau='k=8', kbar=8, WF1_hardness_k=d_['curve_k8_WF1'].mean()))
            print(f'\\n### Table II-b — W-F1 vs kbar ({c}; chosen tau per seed: {d_.tau.tolist()})'); print(md_table(pd.DataFrame(rows_).round(2)))
for name in ['table_V_cross_corpus', 'table_VI', 'paired_bootstrap_CIs']:
    p = f'{RESULTS}/{name}.csv'
    if not os.path.exists(p) or os.path.getsize(p) < 5: print(f'\\n### {name}: not produced (needs IEMOCAP in this run)'); continue
    print(f'\\n### {name}'); print(md_table(pd.read_csv(p).round(3)))
print('\\nCORPORA IN THIS RUN:', CFG['corpora'], '| AUDIO:', {c: bool(len(meta[(meta.corpus==c) & (meta.wav!='')])) for c in CFG['corpora']})""")

md("""## 11. Export
All CSVs in `MyDrive/hardness_erc/results/`: `summary_tables_II_IV.csv`, `paired_bootstrap_CIs.csv`, `table_III_<corpus>.csv`, `table_VI.csv`, plus per-experiment CSVs (per seed/fold) and `*_pooled.npz` (predictions, probabilities, hardness, test indices) for confusion matrices and supplementary plots.

Published baseline rows (†) for Table I are copied from the cited papers; only ‡ rows carry CIs.

**Decision rule written into the paper (§V-B):** if `T2_hardness` ≈ `T2_random_kbar` the hardness signal is irrelevant to context; if it ≈ `T2_fixed8` at equal k̄ the contribution is compute-only. Check Table III for the crossing interaction before writing the results section.""")

md("""## 11. Post-hoc analysis for the manuscript (rev. 4 review items)
Per-class F1 and confusion matrices for IEMOCAP at k = 0 vs k = 8, the realised k̄ of the entropy-k policy, and the trained random-k row. Reads only checkpoints; no GPU time.""")
code("""# ===== POST-HOC CELL (paste as the LAST cell of hardness_aware_ERC_final_colab.ipynb and run after "Run all") =====
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

print('\\nCONFUSION MATRIX k=0 (rows = true, cols = predicted, % of row)')
z0 = pooled('T2_fixed0', 'iemocap'); z8 = pooled('T2_fixed8', 'iemocap')
for tag, z in (('k=0', z0), ('k=8', z8)):
    cm = confusion_matrix(z['y'], z['p'], labels=range(len(L))); cm = cm / cm.sum(1, keepdims=True) * 100
    print(f'\\n{tag}'); print(pd.DataFrame(cm.round(1), index=L, columns=L).to_string())
    pd.DataFrame(cm.round(2), index=L, columns=L).to_csv(f'{RESULTS}/confusion_{tag.replace("=", "")}_iemocap.csv')

print('\\n' + '=' * 70); print('REALISED k-bar OF THE ENTROPY-k POLICY (test set, per corpus)')
for c in CFG['corpora']:
    z = pooled('T2_fixed0', c)
    if z is None: continue
    ent = -(z['prob'] * np.log(z['prob'] + 1e-9)).sum(1)
    kk = np.array([KS[min(4, int(np.searchsorted(ENT_Q, e)))] for e in ent])
    print(f'{c}: entropy-k k-bar = {kk.mean():.2f}  (ENT_Q = {np.round(ENT_Q, 3).tolist()})')

print('\\n' + '=' * 70); print('RANDOM-k ROW (trained) ON IEMOCAP')
p = f'{RESULTS}/T2_random_kbar_iemocap.csv'
if os.path.exists(p):
    d = pd.read_csv(p); print(d[['seed', 'dev_WF1', 'test_WF1', 'test_MF1', 'test_UA']].round(2).to_string(index=False))
    print('mean WF1 %.2f +- %.2f   (k-bar used: %s)' % (d.test_WF1.mean(), d.test_WF1.std(), RUN_STATE.get('iemocap_fixed_Ses05_kbar', 'from hardness run')))
print('\\nDONE - send this cell\\'s output.')
""")

nb['cells'] = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}}
nbf.write(nb, 'hardness_aware_ERC_rev3_colab.ipynb'); print('ok', len(cells))
