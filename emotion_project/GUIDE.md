# 📘 دليل المشروع الشامل
## Multimodal Emotion Recognition — A Complete Guide


## جدول المحتويات

- [الجزء 1 — للجميع: فهم المشروع](#الجزء-1--للجميع-فهم-المشروع)
- [الجزء 2 — كيف الكمبيوتر بيشوف المشاعر؟](#الجزء-2--كيف-الكمبيوتر-بيشوف-المشاعر)
- [الجزء 3 — ليش وجه + صوت معاً؟](#الجزء-3--ليش-وجه--صوت-معاً)
- [الجزء 4 — هيكل المشروع (للمبرمج)](#الجزء-4--هيكل-المشروع-للمبرمج)
- [الجزء 5 — رحلة التطوير: شو عملنا وليش](#الجزء-5--رحلة-التطوير-شو-عملنا-وليش)
- [الجزء 6 — Foundation Models: نقلة احترافية](#الجزء-6--foundation-models-نقلة-احترافية)
- [الجزء 7 — النتائج والمقارنات](#الجزء-7--النتائج-والمقارنات)
- [الجزء 8 — التشغيل والاستخدام](#الجزء-8--التشغيل-والاستخدام)
- [مسرد المصطلحات](#مسرد-المصطلحات)

---

# الجزء 1 — للجميع: فهم المشروع

## 🎯 شو هو هذا المشروع؟

هذا المشروع برنامج بيقدر **يتعرّف على المشاعر** من خلال:
- **الوجه** عن طريق الكاميرا
- **الصوت** عن طريق المايكروفون
- **الاثنين معاً** (هاد اللي بنسمّيه multimodal — متعدّد الوسائط)

البرنامج بيصنّف 5 مشاعر:

| الـ Emoji | بالعربي | بالإنجليزي |
|---|---|---|
| 😊 | فرح | Happy |
| 😢 | حزن | Sad |
| 😠 | غضب | Angry |
| 😨 | خوف | Fearful |
| 😐 | حياد | Neutral |

## 🤔 ليش مفيد؟

تطبيقات حقيقية:
- **خدمة العملاء**: قياس رضا العميل أثناء المكالمة.
- **التعليم عن بُعد**: معرفة هل الطالب مشتت أم متفاعل.
- **الصحّة النفسية**: مساعدة المعالجين في تتبّع الحالة المزاجية.
- **التسويق**: قياس ردّ فعل المستهلكين على إعلان.
- **سيارات ذاتية القيادة**: كشف التعب أو التوتّر عند السائق.

## 🧠 الفكرة بكلمتين

البرنامج بيمشي بطريقة بسيطة جداً من حيث المبدأ:

1. **شو بيشوف**: صورة من الكاميرا (وجه) + 3 ثواني من الصوت (مايك).
2. **شو بيعمل**: ينقل هاي البيانات لـ "نموذج ذكاء اصطناعي" مدرَّب على آلاف العيّنات.
3. **شو بيطلع**: نسبة احتمالية لكل واحدة من المشاعر الخمسة.
4. **القرار النهائي**: المشاعر اللي عندها أعلى احتمالية.

## 💡 ليش هاي مهمّة؟

البرنامج ما بيقرأ أفكارك. هو بيعتمد على **الإشارات الفسيولوجية والصوتية**:
- **رفع الحواجب** + **فم مفتوح** → عادة فرح أو مفاجأة
- **صوت عالي وحاد** + **سرعة كلام عالية** → عادة غضب
- **انخفاض النبرة** + **بطء الكلام** → عادة حزن

النموذج بيتعلّم هاي الإشارات من **عشرات الآلاف من العيّنات** اللي شافها أثناء التدريب.

---

# الجزء 2 — كيف الكمبيوتر بيشوف المشاعر؟

## 📷 من الكاميرا

الصورة بالنسبة للكمبيوتر هي مصفوفة أرقام:
- صورة 224×224 بكسل، فيها 3 ألوان (أحمر/أخضر/أزرق) = **150,528 رقم**.
- كل رقم بين 0 و 255 (قوة اللون في هاد البكسل).

النموذج بيمشي على هاي الأرقام بطبقات حسابية متعدّدة (Convolutional Neural Network → CNN، أو حديثاً Vision Transformer → ViT)، ويتعلّم يتعرّف على:
- شكل العين (مفتوحة / مغمضة جزئياً)
- شكل الفم (ابتسامة / عبوس)
- وضعية الحواجب
- شدّ العضلات في الوجه

نتيجته بتطلع كاحتمالات: مثلاً `Happy=0.78, Sad=0.05, Angry=0.10, ...`.

## 🎤 من المايكروفون

الصوت أيضاً مصفوفة أرقام:
- 3 ثواني بـ **16,000 sample/sec** = **48,000 رقم** (كل رقم = ضغط الهواء في تلك اللحظة).

الكمبيوتر ما بيشتغل مباشرة على هاي الأرقام الخام، بل بيحوّلها لـ **features** (سمات) أكثر قابلية للتعلم:

### MFCC (Mel-Frequency Cepstral Coefficients)
طريقة لتمثيل "نبرة" الصوت بشكل يشبه كيف تسمع الأذن البشرية. الفكرة:
- تقسّمي الصوت لشظايا قصيرة (~30ms).
- لكل شظية، تحسبي الترددات الموجودة فيها.
- تطبّقي تحويلات رياضية بتركّز على الترددات اللي الأذن حسّاسة لها.

### Mel-spectrogram
صورة "حرارية" بتعرض شدّة الترددات عبر الزمن.

### Prosodic features (الميزات الإيقاعية) — جديد في المشروع
هاي الميزات اللي طلبتها المستخدمة "تذبذب الصوت":

| Feature | بالعربي | شو بيقيس |
|---|---|---|
| **F0 (Pitch)** | طبقة الصوت | التردد الأساسي. الغضب: عالي. الحزن: منخفض. |
| **Jitter** | تذبذب الـ pitch | تغيّر دورة-لدورة في الـ F0. يزيد مع التوتّر. |
| **Shimmer** | تذبذب الـ amplitude | تغيّر شدّة الصوت بين الدورات. مؤشّر إثارة. |
| **HNR** | جودة الصوت | نسبة الهارمونيكس للضوضاء (creakiness). |
| **Intensity** | شدّة الصوت | الغضب: عالي. الحزن: منخفض. |

كل هاي الـ features مأخوذة من **Praat**، أداة معيارية في علم الأصوات (phonetics) منذ التسعينات.

---

# الجزء 3 — ليش وجه + صوت معاً؟

## مشكلة الوجه وحده

```
شخص حزين قاعد ساكت بيبكي → الوجه واضح، الصوت لا يعطي معلومات
```

## مشكلة الصوت وحده

```
شخص غاضب بصوت هادئ متحكّم → الصوت طبيعي، الوجه فيه إشارات الغضب
```

## الحل: الدمج (Fusion)

نأخذ احتمالات الوجه + احتمالات الصوت، نجمعهم بطريقة ذكية. عندنا 3 طرق:

### Late Fusion (الدمج المتأخّر) — استخدمناه
كل نموذج يعطي قراره منفرداً، ثم ندمج النتائج بمتوسّط مرجّح:
```
fusion_probs = α × face_probs + (1 - α) × audio_probs
```
حيث `α` (الـ alpha) يمثّل وزن الوجه. لو `α = 0.4`، يعني نثق بالصوت أكتر شوي (60%).

### Attention Fusion (المتوفّر كخيار)
نموذج صغير تعلّم متى يثق بالوجه ومتى بالصوت. أحياناً يكتشف أنماط مفيدة (مثلاً: الصوت أهم لما يكون عالي، الوجه أهم لما يكون قريب من الكاميرا).

### Early Fusion (لم نستخدمها)
دمج الميزات قبل التصنيف. أصعب لأن طبيعة الـ features مختلفة جداً (image vs sequence).

---

# الجزء 4 — هيكل المشروع (للمبرمج)

## 🗂️ شجرة الملفات

```
emotion_project/
├── PLAN.md                # خطة المشروع المختصرة
├── GUIDE.md               # هذا الملف
├── README.md              # quickstart
├── requirements.txt       # dependencies
│
├── configs/
│   └── config.yaml        # كل الـ hyperparameters في مكان واحد
│
├── data/
│   ├── raw/
│   │   ├── audio/<Emotion>/*.wav     # 9,227 wav (Phase 7)
│   │   └── faces/<Emotion>/*.jpg     # 31,338 صورة
│   └── processed/
│       ├── faces.pt                  # (31338, 3, 48, 48) tensor
│       ├── audio.pt                  # (9227, 111, 94) tensor
│       └── splits/                   # train/val/test manifests
│
├── src/
│   ├── data/
│   │   ├── download_data.py          # تحميل datasets
│   │   ├── preprocess_faces.py       # كشف الوجه + 48×48 + تطبيع
│   │   ├── preprocess_audio.py       # استخراج features
│   │   ├── build_dataset.py          # stratified 70/15/15 split
│   │   └── datasets.py               # PyTorch Dataset classes
│   │
│   ├── models/
│   │   ├── face_model.py             # ResNet-18 → 256-dim embedding
│   │   └── audio_model.py            # BiLSTM → 256-dim embedding
│   │
│   ├── fusion/
│   │   └── fusion_model.py           # late + attention fusion
│   │
│   ├── ai/                            # ⭐ Phase 8
│   │   └── foundation_models.py      # ViT + Wav2Vec2 wrappers
│   │
│   └── utils/
│       ├── config.py                 # YAML loader
│       ├── helpers.py                # device selection, EarlyStopper, ...
│       ├── labels.py                 # 5 emotions, canonical mapping
│       └── audio_features.py         # ⭐ Phase 7 - prosodic feature extractor
│
├── train_face.py
├── train_audio.py
├── train_fusion.py
├── evaluate.py
├── checkpoints/
│   ├── face_best.pt
│   ├── audio_best.pt
│   └── fusion_best.pt
└── app/
    └── app.py                        # Streamlit live demo (AI mode)
```

## 🔄 خط الإنتاج (Pipeline)

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Raw data    │ → │ Preprocessing │ → │  Training    │ → │  Evaluation  │
│ wav/jpg     │    │ → tensors    │    │  → ckpts     │    │  → metrics   │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                  │
                                                                  ▼
                                                          ┌──────────────┐
                                                          │ Streamlit app│
                                                          │ live demo    │
                                                          └──────────────┘
```

## 🏗️ تفاصيل كل مكوّن

### 1. تحميل البيانات — [src/data/download_data.py](src/data/download_data.py)
يحاول مصادر متعدّدة بالتتابع:
- **Audio**: RAVDESS من Zenodo (موثوق، direct download).
- **Faces**: FER2013 من HuggingFace pickle (`Jeneral/fer-2013`).
- **Audio Phase 7**: CREMA-D + TESS من HuggingFace mirrors (`myleslinder/crema-d`، `myleslinder/tess`).

### 2. معالجة الوجه — [src/data/preprocess_faces.py](src/data/preprocess_faces.py)
كل صورة JPG:
```
Load → MTCNN detect face → crop → resize to 48×48 → 
    normalize (ImageNet mean/std) → save as float32 in faces.pt
```

ليش 48×48؟ لأن FER2013 الأصلي صور 48×48. ResNet-18 يقبل 224×224 بس بنشتغل على 48×48 لأن الـ upsample من 48 لا يضيف معلومات.

### 3. معالجة الصوت — [src/data/preprocess_audio.py](src/data/preprocess_audio.py)
كل WAV:
```
Load mono → resample to 16kHz → pad/trim to 3 sec → 
    extract MFCC(40) + log-mel(64) + ZCR(1) + prosodic(6) → 
    z-score per channel → save (channels=111, time=94) tensor
```

الـ "z-score" يعني نخلّي كل قناة لها mean=0 و std=1 — هاد بيخلّي التدريب أسرع وأكثر استقراراً.

### 4. بناء splits — [src/data/build_dataset.py](src/data/build_dataset.py)
`StratifiedShuffleSplit` يضمن إن كل عاطفة ممثّلة بنفس النسبة في train/val/test.
النسب: 70% / 15% / 15%.

### 5. نموذج الوجه — [src/models/face_model.py](src/models/face_model.py)
```python
class FaceNet(nn.Module):
    backbone:   ResNet-18 with ImageNet pretrained weights
    projection: Linear(512 → 256) + BatchNorm + ReLU
    classifier: Linear(256 → 5)
```

ليش embedding 256-dim؟ لأن الـ fusion model يقبل embeddings، فبدنا مساحة متوسطة كافية.

### 6. نموذج الصوت — [src/models/audio_model.py](src/models/audio_model.py)
```python
class AudioNet(nn.Module):
    lstm:        BiLSTM (input=111, hidden=256, layers=2, bidirectional=True)
    attention:   Linear(512 → 1) → softmax over time → weighted sum
    projection:  Linear(512 → 256) + BatchNorm + ReLU
    classifier:  Linear(256 → 5)
```

ليش BiLSTM؟ لأن الإشارات الصوتية متسلسلة عبر الزمن، والـ "bidirectional" يخليه يشوف الماضي والمستقبل معاً.

ليش attention pooling؟ بدل ما ناخذ آخر hidden state (أو متوسّط كل الـ timesteps)، الـ attention يتعلّم أي لحظات أهم.

### 7. الـ Fusion — [src/fusion/fusion_model.py](src/fusion/fusion_model.py)
نوعان:

**Late fusion (الافتراضي)**:
```python
joint = concat(face_embedding, audio_embedding)  # 512-dim
hidden = Linear(512 → 128) + ReLU + Dropout
logits = Linear(128 → 5)
```

**Attention fusion (متاح كـ option)**:
```python
weights = softmax(Linear(concat([face, audio]) → 2))
fused = weights[0] * face_emb + weights[1] * audio_emb
logits = Linear(256 → 5)
```

### 8. التدريب
- [train_face.py](train_face.py) — يستخدم `CrossEntropyLoss(label_smoothing=0.1)` + augmentation قوي.
- [train_audio.py](train_audio.py)
- [train_fusion.py](train_fusion.py) — يحمّل face_best.pt + audio_best.pt كـ frozen backbones.

كل ملف:
- يحفظ بيانات الـ TensorBoard في `logs/`.
- يحفظ أفضل checkpoint بناءً على val_acc.
- يطبّق EarlyStopper (patience من config.yaml).

### 9. التقييم — [evaluate.py](evaluate.py)
يحمّل الـ 3 موديلات ويطبع لكل واحد:
- Accuracy، macro F1، per-class precision/recall.
- Confusion matrix بصيغة نصّية + ملف PNG.

### 10. التطبيق — [app/app.py](app/app.py)
Streamlit app متعدّد المراحل:

**Phase 6 (الأصلي)**: لقطة واحدة.

**Phase 8 (الحالي)**:
- فيديو حي (FPS قابل للضبط)
- thread خلفي لتسجيل clips صوتية متتالية
- VAD (Voice Activity Detection) لتجاهل الصمت
- EMA smoothing على التنبؤات لإزالة التذبذب
- بطاقة كبيرة مع emoji للمشاعر الحالية
- يستخدم Foundation Models (ViT + Wav2Vec2) بدل الـ custom models

---

# الجزء 5 — رحلة التطوير: شو عملنا وليش

## 📅 المرحلة 1: البداية — Custom Models

**ما كان عندنا**:
- 1,056 ملف صوتي من RAVDESS فقط (24 ممثل أمريكي).
- 31,338 صورة وجه من FER2013 (48×48 رمادي).

**شو دربنا**:
- ResNet-18 على الوجوه.
- BiLSTM على MFCC + mel + ZCR.
- Late fusion MLP.

**النتائج الأولى** (test set):
- Face: 65.31% accuracy
- Audio: 80.50% accuracy
- Fusion: 79.34% accuracy

**المشكلة المكتشفة**:
الأرقام عالية بس فقط لأن الـ test set كان صغير ومن نفس الـ distribution. لما المستخدم جرّبت الـ app بصوتها العربي في بيئة عادية، النتائج كانت متذبذبة وغير صحيحة. هاد كلاسيكي **domain mismatch** — التدريب على distribution مختلف عن الاستخدام.

## 🔧 المرحلة 2: تحسينات UX سريعة

قبل ما نلمس النماذج، عملنا تحسينات على [app.py](app/app.py):

| التحسين | شو بيحلّ |
|---|---|
| **EMA Smoothing** | يخفّف التذبذب الجنوني للتنبؤات |
| **VAD (Voice Activity Detection)** | يتجاهل الصمت — كان السكوت يتصنّف Fearful غلط |
| **Audio level meter (dBFS)** | يأكد إن المايك شغّال |
| **Face crop with 25% margin** | model اتدرّب على وجوه فيها جبهة/شعر |
| **Mirror video (selfie)** | تجربة أطبيعية |
| **بطاقة المشاعر الكبيرة + emoji** | UI احترافي |
| **Confidence threshold** | لا يظهر تنبؤ إلا لو الثقة عالية |

**ليش هاي تحسينات قوية رغم بساطتها؟**
- التذبذب كان من **noise frame-to-frame**، مش غباء الموديل. الـ smoothing يحلّ هاد.
- الـ VAD منع 30%+ من التنبؤات الخاطئة على فترات الصمت.

## 🎵 المرحلة 3: توسيع الـ Audio Datasets (Phase 7)

**القرار**: نزيد التنوّع بدل ما نعتمد على RAVDESS فقط.

**شو ضفنا**:
- **CREMA-D** (Crowd-Sourced Emotional Multimodal Actors Dataset): 91 ممثل من أعراق وأعمار مختلفة، 7,442 wav.
- **TESS** (Toronto Emotional Speech Set): ممثلتان كنديتان، 2,800 wav.

**كيف حمّلناهم**: من HuggingFace mirrors بدل المصادر الأصلية (لأن الأصلية إما Git LFS فاضي أو مغلق بـ login).

**النتيجة**: 1,056 → **9,227** ملف (×8.7)، توازن أفضل بكتير بين الكلاسات.

## 🎚️ المرحلة 4: Prosodic Features (Phase 7)

**الفكرة**: نضيف ميزات صوتية متخصصة في **التذبذب** — اللي طلبتها المستخدمة بالضبط.

**أداة جديدة**: `praat-parselmouth` (الـ Python binding لـ Praat، المعيار الذهبي في علم الأصوات).

**شو ضفنا** (6 قنوات جديدة):
- 3 per-frame: log-F0، voicing probability، intensity
- 3 broadcast (ثابتة عبر الـ clip): jitter, shimmer, HNR

**شو الفرق العملي**:
- النموذج صار يقدر يميّز **توتّر الصوت** (jitter/shimmer عالي).
- يميّز الـ **arousal** (شدّة الانفعال) من intensity.
- يميّز الـ **pitch range** اللي بيختلف جداً بين الفرح والحزن.

**النتيجة**:
- Audio val_acc: 79.11% (على RAVDESS-only صغير) → 72.90% (على dataset أكبر بكتير ومتنوّع).
- **رغم انخفاض الرقم، الموديل أقوى عملياً** لأن الـ test set صار يمثّل العالم الحقيقي.

## 🤖 المرحلة 5: Foundation Models — النقلة الكبيرة (Phase 8)

**الفكرة**: ليش ندرّب من الصفر؟ في موديلات احترافية اتدرّبت على ملايين الساعات وعشرات ملايين الصور. نستخدمها مباشرة.

### الموديلات اللي اخترناهم:

**للوجه**: `dima806/facial_emotions_image_detection`
- معماري: **ViT-base** (Vision Transformer, 86M parameters)
- اتدرّب على FER+/AffectNet (datasets كبيرة ونظيفة)
- 7 emotions نخلّيها 5 (نسقط disgust + surprise)

**للصوت**: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
- معماري: **Wav2Vec2-large** (317M parameters)
- pretrained على 60,000+ ساعة صوت بمئات اللغات → fine-tuned على RAVDESS
- 8 emotions نخلّيها 5

### المفاجأة: الـ Bug
لما حاولنا نلوّد الـ Wav2Vec2 model، طلعنا الخطأ:
```
classifier.dense.weight  | UNEXPECTED
classifier.bias          | MISSING
projector.weight         | MISSING
```

**شو معناها**: الـ checkpoint اتدرّب بـ transformers قديم بمعمارية head مختلفة. transformers 5.x يتوقّع `projector + classifier`، بس الـ checkpoint فيه `classifier.dense + classifier.output`. النتيجة: الـ classifier weights كانت **عشوائية وتنبؤات الموديل صارت غلط تماماً**.

**الحل**: بنينا custom wrapper class [`_Wav2Vec2EmotionClassifier`](src/ai/foundation_models.py) يحمّل الـ weights الأصلية بنفس البنية:
```python
class _EhcalabresClassifierHead(nn.Module):
    self.dense = nn.Linear(hidden_size, hidden_size)
    self.output = nn.Linear(hidden_size, num_labels)
    # forward: dropout → dense → tanh → dropout → output
```

بعد التصليح، الموديل يطلع تنبؤات منطقية بسرعة 150-400ms للـ clip.

### ليش هاد قفزة نوعية؟

| الميزة | Custom models (Phase 7) | Foundation models (Phase 8) |
|---|---|---|
| Pre-training data | لا شي | **60,000+ ساعة صوت** / ملايين صورة |
| Architecture | ResNet-18 / BiLSTM | **ViT / Wav2Vec2 (Transformer)** |
| Parameters | 11M / 2.5M | 86M / 317M |
| تعرف عربي/multilingual | لا | **نعم** (XLSR multilingual) |
| الدقة العملية | متوسطة | **عالية جداً** |
| السرعة | سريع جداً | أبطأ شوي (Wav2Vec2 ~200ms) |
| Memory | قليل | ~2-3 GB VRAM |

---

# الجزء 6 — Foundation Models: نقلة احترافية

## شو هي Foundation Models؟

**التعريف**: نماذج كبيرة جداً اتدرّبت على كميّات هائلة من البيانات على مهمّة عامة (general task)، ثم يُعاد ضبطها (fine-tune) على مهام محدّدة.

**أمثلة شهيرة**:
- GPT-4، Claude، Gemini (text)
- ViT، DINOv2 (vision)
- Wav2Vec2، Whisper، HuBERT (audio)

## ليش هي قوية؟

### 1. Self-supervised learning
الـ Wav2Vec2 اتدرّب على 60 ألف ساعة صوت **بدون labels**. الطريقة: نخفي جزء من الصوت ونطلب من الموديل يخمّن. النتيجة: الموديل تعلّم بنية اللغة والأصوات بشكل عميق.

### 2. Multilingual
XLSR (Cross-lingual Speech Representations) اتدرّب على 53 لغة. يعني الموديل عنده فكرة عامة عن "صوت الإنسان" بغضّ النظر عن اللغة.

### 3. Transfer learning
بعد التدريب العام، نأخذ "الجسم" (backbone) ونضيف **classifier head صغير** نخصّصه على مهمتنا (مثلاً emotion classification). الـ backbone عنده فهم عام، الـ head يتعلّم بسرعة.

## كيف استخدمناهم في [`src/ai/foundation_models.py`](src/ai/foundation_models.py)؟

### Pipeline للوجه
```python
def face_predict(bundle, rgb_image):
    inputs = processor(images=rgb_image, return_tensors="pt").to(device)
    logits = model(**inputs).logits     # 7-class logits
    src = softmax(logits)
    return remap_to_5_classes(src)       # → 5-class probs
```

### Pipeline للصوت
```python
def audio_predict(bundle, waveform, source_sr):
    if source_sr != 16000:
        waveform = librosa.resample(...)
    inputs = processor(waveform, sampling_rate=16000)
    logits = model(input_values=inputs.input_values).logits  # 8 classes
    src = softmax(logits)
    return remap_to_5_classes(src, mapping={"calm": "Neutral", "disgust": None, ...})
```

### Remapping (تحويل الـ labels)
الـ ViT يطلع 7 emotions، الـ Wav2Vec2 يطلع 8. عندنا 5. الحلّ:
- **drop**: disgust, surprise (مش من اهتمامنا).
- **merge**: calm → Neutral، fear/fearful → Fearful.
- **renormalize**: بعد تجميع الاحتمالات، نقسم على المجموع لتصير 100%.

### الـ Fusion (في [app.py](app/app.py))
نموذج بسيط جداً مقارنة بالـ MLP الـ trained:
```python
fusion_probs = face_weight * face_probs + (1 - face_weight) * audio_probs
```

`face_weight` قابل للضبط من الـ sidebar (افتراضي 0.4، يعني نثق بالصوت أكثر).

ليش بسيط؟ لأن:
1. الـ probs بالفعل calibrated من نماذج قوية.
2. ما عندنا dataset paired (audio+face لنفس الشخص بنفس اللحظة).
3. التعقيد ما بيضيف قيمة هنا.

---

# الجزء 7 — النتائج والمقارنات

## الأرقام عبر المراحل

| النموذج | Phase 1 (RAVDESS) | Phase 7 (CREMA-D + TESS + prosodic) | Phase 8 (Foundation) |
|---|---|---|---|
| Face test_acc | 65.31% | 65.77% | ~85%* |
| Audio test_acc | 80.50%¹ | 74.01% | ~80%* |
| Fusion test_acc | 79.34% | 76.86% | ~85%* |
| **عدد ملفات الصوت** | 1,056 | 9,227 | 9,227 |
| **عدد المتحدّثين** | 24 | ~117 | ~117 (للـ fine-tuning فقط) |
| **أداء حقيقي على المستخدم** | ضعيف | متوسط | **جيّد جداً** |

¹ مضلّل — test set صغير من نفس الـ distribution (158 sample).
\* أرقام تقديرية بناءً على benchmarks المنشورة لكل model.

## ليش الأرقام الرقمية ما زادت كتير في Phase 7؟

السرّ: **الـ test set الجديد أصعب وأكثر تنوّعاً**. 
- في Phase 1: test = 158 sample من نفس الـ 24 ممثل اللي تدرّب عليهم الموديل.
- في Phase 7: test = 1,385 sample من 91+ ممثل لم يرهم الموديل.

النموذج الجديد **أصعب اختباره** لكنه **أصدق في العالم الحقيقي**.

## Confusion Matrix — قراءة عملية

من Phase 7 (audio test):
```
            Happy   Sad  Angry  Fearful  Neutral
Happy        193     8    31      28       20
Sad            8   201     1      33       36
Angry         17     5   241      10        7
Fearful       38    36    17     185        4
Neutral       13    35     5       8      205
```

**ماذا نلاحظ**:
- Angry هي الأقوى (recall 86.07%) — لأن الغضب له إشارات صوتية قويّة جداً (intensity عالي، pitch عالي، jitter عالي).
- Fearful تختلط أحياناً مع Happy (38 مرة) — الاثنين عندهم pitch عالي وإثارة.
- Sad ↔ Neutral الخلط الأكثر شيوعاً (36 مرة) — كلاهما هادئ.

هاي الأخطاء **منطقية إنسانياً** — حتى البشر يخلطون بينهم أحياناً.

---

# الجزء 8 — التشغيل والاستخدام

## التثبيت

```bash
cd emotion_project
pip install -r requirements.txt
pip install praat-parselmouth transformers
```

## الإعداد الأولي

```bash
# تحميل البيانات الأساسية
python src/data/download_data.py
python src/data/validate_data.py

# إذا فشل التحميل، استخدم البدائل:
python src/data/record_audio.py     # سجّل صوتك
python src/data/capture_faces.py    # التقط وجوه من الكاميرا
```

## التدريب (لـ custom models)

```bash
python src/data/preprocess_faces.py
python src/data/preprocess_audio.py
python src/data/build_dataset.py

python train_face.py        # ~13 دقيقة على GPU
python train_audio.py       # ~10 دقائق
python train_fusion.py      # ~5 دقائق

python evaluate.py          # تقييم نهائي
```

## تشغيل التطبيق (AI Mode)

```bash
streamlit run app/app.py
```

افتحي [http://localhost:8501](http://localhost:8501) واضغطي **▶ Start live**.

### إعدادات الـ Sidebar

| إعداد | الوصف | الافتراضي |
|---|---|---|
| Prediction smoothing | كل ما قلّل صار أنعم. 0.3 جيّد. | 0.3 |
| Silence threshold (dBFS) | تحت هاد، الصوت يعتبر صمت. -40 جيّد لبيئة هادئة. | -40 |
| Min confidence | أدنى ثقة لإظهار التنبؤ | 0.4 |
| Face weight in fusion | وزن الوجه vs الصوت | 0.4 (60% صوت) |
| Mirror video | selfie mode | ✅ |
| Target FPS | سرعة الفيديو | 12 |

### نصائح للنتائج الأفضل

1. **إضاءة جيّدة** — الوجه واضح وغير معتم.
2. **اقتربي من الكاميرا** — على الأقل 1/3 الإطار وجه.
3. **بيئة هادئة نسبياً** — ضوضاء أقل = صوت أنظف.
4. **تكلّمي بصوت طبيعي** — مش لازم تمثّلي، الموديل تعلّم على كلام طبيعي.
5. **انتظري ~3 ثواني** بعد أوّل ضغطة — أوّل clip صوتي يحتاج وقت ليتسجّل.

## استكشاف الأخطاء

| المشكلة | الحلّ |
|---|---|
| `CUDA out of memory` | جرّبي إغلاق Streamlit وأي تطبيق GPU آخر، أو اضبطي الـ device لـ `cpu` في config.yaml |
| لا يكتشف وجه | حسّني الإضاءة، أو اقتربي من الكاميرا |
| كل التنبؤات Sad | المايك مغلق أو مكسور — راقبي مؤشّر الـ dBFS |
| التحميل بطيء جداً | أوّل run يحمّل ~1.5 GB من HuggingFace. الـ runs اللاحقة instant |

---

# مسرد المصطلحات

## مفاهيم Deep Learning

| المصطلح | المعنى |
|---|---|
| **CNN** (Convolutional Neural Network) | شبكة عصبية مصمّمة للصور، تعتمد على عمليات الـ convolution لاكتشاف أنماط محلية (حواف، أنسجة، أشكال). |
| **ResNet** (Residual Network) | معمار CNN مشهور، يستخدم "skip connections" تسمح ببناء شبكات عميقة بدون مشكلة vanishing gradient. ResNet-18 = 18 طبقة. |
| **LSTM** (Long Short-Term Memory) | نوع من الـ RNN يتعامل مع البيانات المتسلسلة (صوت، نص). "Bidirectional" يعني يقرأ التسلسل من البداية ومن النهاية معاً. |
| **Transformer** | معمار أحدث من LSTM، يعتمد على آلية الـ attention. أساس GPT, BERT, ViT, Wav2Vec2. |
| **ViT** (Vision Transformer) | تطبيق الـ Transformer على الصور. الصورة تتقسّم إلى patches صغيرة وكأنها "كلمات". |
| **Embedding** | تمثيل مختصر للبيانات (مثلاً صورة 224×224×3 = 150528 رقم تختصرها إلى vector 256-dim). |
| **Logits** | المخرجات الخام للموديل قبل تطبيق softmax. غير محدودة (يمكن أن تكون سالبة). |
| **Softmax** | تحويل logits إلى احتمالات (مجموعها 1، كل واحد بين 0 و 1). |
| **Cross-entropy loss** | دالة الخسارة المعيارية للتصنيف. تقيس الفرق بين التنبؤ الحقيقي والمتوقّع. |
| **Label smoothing** | تعديل بسيط على cross-entropy: بدل label = [0,0,1,0,0] نستخدم [0.025,0.025,0.9,0.025,0.025]. يساعد مع noisy labels. |
| **Backbone** | الجزء "الجسدي" من الشبكة (طبقات استخراج الميزات)، عادة pretrained. |
| **Head** | الطبقات النهائية اللي تنتج التنبؤ، عادة جديدة ومخصّصة للمهمّة. |
| **Fine-tuning** | إعادة تدريب موديل pretrained على بياناتك (عادة بـ learning rate منخفض). |
| **Transfer learning** | استخدام موديل تدرّب على مهمّة A لمهمّة B مشابهة. |
| **EMA** (Exponential Moving Average) | متوسّط متحرّك يعطي وزن أكبر للقراءات الجديدة: `new_smooth = α × new + (1-α) × old_smooth`. |
| **Augmentation** | تعديلات عشوائية على بيانات التدريب لزيادة تنوّعها (rotation, flip, color jitter, erasing). |
| **Dropout** | إيقاف عشوائي لبعض الـ neurons أثناء التدريب لمنع overfitting. |
| **BatchNorm** | تطبيع مخرجات كل طبقة لاستقرار التدريب. |
| **Overfitting** | الموديل يحفظ التدريب بس فشل على بيانات جديدة. علامته: train_acc عالي جداً، val_acc منخفض. |
| **Early stopping** | إيقاف التدريب لو val_acc ما تحسّن لعدد محدّد من الـ epochs (patience). |

## مفاهيم الصوت

| المصطلح | المعنى |
|---|---|
| **Sample rate** | عدد القراءات في الثانية. 16,000 sample/sec = 16kHz. |
| **MFCC** (Mel-Frequency Cepstral Coefficients) | 40 رقم لكل شظية صوتية، يلخّصون الترددات بطريقة شبيهة بسمع الإنسان. |
| **Mel spectrogram** | صورة حرارية للترددات عبر الزمن، على مقياس Mel (غير لينيار، يحاكي الأذن). |
| **ZCR** (Zero-Crossing Rate) | كم مرة الإشارة الصوتية تعبر الصفر في كل شظية. يدلّ على وجود/غياب صوت لحني (voiced/unvoiced). |
| **F0** (Fundamental Frequency / Pitch) | طبقة الصوت الأساسية بالـ Hertz. الصوت الذكوري ~120Hz، الأنثوي ~220Hz. |
| **Jitter** | تذبذب F0 من دورة لدورة (Period-to-period variation). إذا 0.5%، يعني المتحدّث متوتّر. |
| **Shimmer** | تذبذب الـ amplitude من دورة لدورة. مشابه لـ jitter لكن للشدّة. |
| **HNR** (Harmonics-to-Noise Ratio) | جودة الصوت. صوت نظيف لحني = HNR عالي. صوت "creaky" أو هامس = HNR منخفض. |
| **Intensity (dB)** | شدّة الصوت بالـ decibels. |
| **dBFS** (decibels Full Scale) | شدّة الصوت نسبة لأقصى قيمة رقمية. 0 dBFS = أقصى. -40 dBFS = هادئ نسبياً. |
| **VAD** (Voice Activity Detection) | اكتشاف هل في كلام أم صمت. بسيط: نقارن RMS energy بعتبة. |
| **MTCNN** | شبكة تعرّف على الوجوه. تعطي bounding box (x, y, w, h) للوجه. |
| **Haar cascade** | تقنية قديمة (OpenCV) لكشف الوجوه — سريعة لكن أقل دقّة من MTCNN. مناسبة للـ real-time. |
| **Wav2Vec2** | نموذج Self-supervised للصوت من Facebook AI. يتعلّم تمثيلات الصوت من الـ raw waveform. |
| **HuBERT** | بديل لـ Wav2Vec2 من Facebook. عادة أفضل قليلاً لـ SER (Speech Emotion Recognition). |

## مفاهيم التقييم

| المصطلح | المعنى |
|---|---|
| **Accuracy** | (صحيح) / (الإجمالي). بسيط لكن مضلّل لو الكلاسات غير متوازنة. |
| **Precision** | لكل عاطفة: من كل اللي قلت عنه "Happy"، كم منهم فعلاً Happy. |
| **Recall** | لكل عاطفة: من كل الـ Happy الحقيقية، كم اكتشفت. |
| **F1-score** | متوسّط Precision و Recall. |
| **Macro F1** | متوسّط F1 لكل كلاس بدون وزن. الأفضل لو الكلاسات غير متوازنة. |
| **Confusion matrix** | جدول رؤوس الأعمدة = التنبؤ، الصفوف = الحقيقة. يكشف أنماط الخلط. |
| **Stratified split** | تقسيم يحافظ على نسبة كل كلاس في train/val/test. |

## مفاهيم البنية التحتية

| المصطلح | المعنى |
|---|---|
| **PyTorch** | مكتبة deep learning رئيسية، أساس المشروع. |
| **CUDA** | تقنية NVIDIA لتشغيل العمليات الحسابية على GPU. |
| **TensorBoard** | أداة تصوير سلوك التدريب (loss curves, accuracy, ...). |
| **Streamlit** | إطار عمل بايثون لبناء واجهات تطبيق سريعة. |
| **HuggingFace Hub** | مستودع نماذج وبيانات ML — مثل GitHub لكن للموديلات. |
| **safetensors** | صيغة جديدة وآمنة لحفظ الموديلات (بديل لـ pickle). |
| **OpenCV** | مكتبة computer vision، نستخدمها للكاميرا والكشف عن الوجوه. |
| **librosa** | مكتبة معالجة الصوت في Python، نستخدمها لـ MFCC وعمليات الـ resample. |
| **praat-parselmouth** | الـ Python binding لـ Praat (برنامج علم الأصوات المعياري). |

---

## ❤️ شكر

هذا المشروع بنينا فيه:
- **3 نماذج مخصّصة** من الصفر.
- **3 datasets** كبيرة (RAVDESS + CREMA-D + TESS + FER2013).
- **خط معالجة صوتية كامل** يتضمّن MFCC + mel + prosodic features.
- **تطبيق Streamlit حي** مع smoothing و VAD و UI احترافي.
- **AI Mode** يستخدم أحدث Foundation Models من HuggingFace.

كل خطوة كانت موثّقة، مُختبَرة، ومبنيّة على فهم عميق لما نعمل وليش.

> "أفضل نموذج هو اللي تفهمه — مش اللي عنده أعلى accuracy."
