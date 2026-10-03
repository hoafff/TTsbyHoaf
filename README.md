# TTsbyHoaf

Pipeline riêng để xây dựng bộ dữ liệu TTS MayMay sạch từ 16 nguồn gốc, kiểm tra chất lượng, huấn luyện Piper/VITS và quản lý checkpoint.

## Vai trò của từng nơi lưu trữ

- **GitHub / repo này**: code, config, pipeline state, docs, Kaggle job definitions.
- **Hugging Face private `hoa748/TTSbyMayMay`**: checkpoints, manifests/QC lớn, transcript packs, stage backups.
- **Kaggle Dataset/Input**: 16 raw compressed audio sources và dataset train V2 khi hoàn tất.
- **Kaggle /working**: file tạm của từng job.
- **Máy local / VS Code**: control center + nghe sample review.

Không commit audio/model/token/cookie vào Git.

## Source lock

Giữ đúng 16 video:

```text
RiAsGTSORZ4
HdbsrkbYzHM
-XC3WibmUKQ
YaxBsFrzwZE
I29SlhGqXgg
0_gFaWnMiCY
Pf9QTip2hqI
S9YN9llAFv4
arAbGLeCUsU
E_AgFYj41Nk
vimmfFJdrYM
hDNesxEPAws
HAe5kdfMP9U
Itad_gcdHHM
0Wjq6gqd1Sc
9XJ78IeFBxY
```

Loại duy nhất:

```text
0nJXkX0L3kQ
```

## Main pipeline

```text
16 original sources
→ canonical transcript/SRT/ASR with timestamps
→ text review in sentence-safe ~40k–60k character parts
→ clean_text
→ semantic sentence planner
→ align text ↔ raw source audio
→ fresh recut WAV
→ boundary/VAD QC
→ repetition + phonetic-crowding QC
→ audio-quality QC
→ ASR verification
→ KEEP only
→ MAYMAY_TRAIN_SAFE_V2
→ continue Piper from epoch 9
```

Old V1 8,974 clips are regression/failure references only. They are not an active repair target.

## Windows / VS Code quick start

Use Python 3.11 for this repo.

```powershell
cd E:\AIProjects\TTsbyHoaf
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements-control.txt
```

Then authenticate Kaggle:

```powershell
kaggle auth login
```

Check the local environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\check_env.ps1
```

## Security

Never commit:
- Kaggle credentials/tokens
- Hugging Face tokens
- YouTube cookies
- checkpoints/models
- raw audio/video
- generated training WAVs

See `.gitignore`.
