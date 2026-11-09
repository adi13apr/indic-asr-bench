# Indic ASR Bench: Fine-tuning Whisper for Hindi Speech Recognition

Fine-tuning OpenAI's Whisper-small on Hindi speech with LoRA, then benchmarking it against the
zero-shot model on clean and noisy audio, with a word-level failure-mode analysis.

**Fine-tuned model:** [Adi-13-adi/whisper-small-hi-lora](https://huggingface.co/Adi-13-adi/whisper-small-hi-lora)

## Results (FLEURS Hindi test set, 418 utterances)

| Model | WER | CER |
|---|---|---|
| Whisper-small, zero-shot | TBD | TBD |
| Whisper-small + LoRA (this project) | **27.8%** | **10.4%** |

The LoRA adapter trains only **1.44% of the model's parameters** (3.5M of 245M).

## Setup

- **Data:** [FLEURS](https://huggingface.co/datasets/google/fleurs) Hindi (`hi_in`): 2,120 train,
  239 validation and 418 test utterances; clips over 30 seconds are filtered out.
- **Model:** `openai/whisper-small` with LoRA (r=32, alpha=64) on the attention `q_proj` and
  `v_proj` layers, using Hugging Face Transformers and PEFT.
- **Training:** 600 steps, batch size 8 with 2 gradient-accumulation steps, learning rate 1e-3
  with 50 warmup steps, fp16, best checkpoint chosen by validation loss. Trained on a single
  Kaggle T4 GPU.
- **Decoding:** language forced to Hindi, `max_new_tokens=440`.
- **Metrics:** WER and CER with `jiwer`.

### Text normalization for Hindi

Whisper's built-in text normalizer removes Unicode combining marks, which deletes Hindi vowel
signs (matras) and makes scores misleading. This project uses its own normalizer that removes
punctuation (including the danda "।") and extra spaces, but keeps all Devanagari characters.

## Noise robustness benchmark

Gaussian noise added to the test set at fixed signal-to-noise ratios (`audiomentations`, seed 0).

| Condition | Zero-shot WER | LoRA WER | Zero-shot CER | LoRA CER |
|---|---|---|---|---|
| 10 dB SNR | 85.1% | 48.4% | 50.9% | 22.3% |
| 5 dB SNR | 93.2% | 61.3% | 60.2% | 32.2% |
| 0 dB SNR | 103.2% | 75.1% | 74.9% | 43.9% |

- The fine-tuned model is better at every noise level. At 5 dB it still beats the zero-shot
  model on clean audio.
- WER above 100% at 0 dB comes from the zero-shot model inserting words that were never spoken.
- **Fine-tuning improved accuracy, not robustness:** both models lose roughly 20 WER points going
  from clean audio to 10 dB. The whole curve moved down, but its slope didn't change.

*Noise results were produced with `max_new_tokens=225` (see Evaluation notes).*

## Failure-mode analysis

### Error types (automatic, fine-tuned model)

Substitutions make up 81% of all errors (2,402), with deletions at 11% and insertions at 7%.
Each substituted word was tagged automatically:

| Error type | Share of errors |
|---|---|
| Wrong word | 38.7% |
| Near-miss spelling (character error rate ≤ 34%) | 37.9% |
| Deleted word | 11.3% |
| Inserted word | 7.2% |
| Orthographic variant (nukta / chandrabindu) | 2.7% |
| Numerals | 1.9% |
| Non-Devanagari output | 0.3% |

Ignoring nukta and chandrabindu differences ("lenient WER") lowers WER by under 1 point, so
spelling conventions are not the main problem. The errors are real recognition errors.

### Most common confusions

| Pattern | Examples | Likely cause |
|---|---|---|
| Grammatical agreement (61% of the top-20 confusions) | है→हैं, के→की, कि→की | Nearly identical in fast speech; the right form depends on grammar elsewhere in the sentence |
| Vowel length | सामान→समान, हानि→हानी | Short vs. long vowels (a/aa, i/ii) confused |
| Aspiration / place of articulation | पदार्थ→पदार्त, मछली→मचली, दक्षिण→दक्षिन | Aspirated and retroflex sounds collapsed |
| English loanwords | वेबसाइट→वेपसाइट, पार्क→पार | Code-switched vocabulary underrepresented in training data |

### Review of the 50 worst utterances (LLM-assisted, manually verified)

| Main cause | Count |
|---|---|
| General mishearing (spread of small sound confusions) | 22 |
| Names and places | 8 |
| English words | 6 |
| Words split or merged | 6 |
| Output truncated | 4 |
| Hallucination (repetition loop) | 2 |
| Numerals | 2 |

Key observations:
- **Names, places and English words cause 28% of the worst errors**, which points to
  code-switched and named-entity speech as the main weakness.
- **Repetition loops:** in two utterances the model repeats a syllable ("पापापा…") until it hits
  the token limit. Because the loop contains no spaces, WER counts it as a single wrong word and
  understates the damage.
- **Some numeral errors are reference conventions, not model errors.** References sometimes spell
  numbers as words (उन्नीस, छप्पन) where the model writes digits (19, 56).
- **Speaker variation:** six sentences appear twice among the 50 worst, spoken by different
  people, with noticeably different WER for the same text.

## Evaluation notes

The first evaluations used `max_new_tokens=225`. Hindi uses many tokens in Whisper's tokenizer,
so long sentences were cut off before they finished. Raising the limit to 440 (Whisper's maximum
is 448) improved the fine-tuned model from 28.6% to 27.8% WER. The clean-test results above use
440; the noise benchmark and the error breakdown come from the earlier 225-token run.

## Limitations and next steps

- **Agreement errors:** rescoring with a Hindi language model could fix है/हैं and के/की choices.
- **Noise:** add noise augmentation during training to flatten the noise curve.
- **More data:** larger Indian-language datasets (e.g. AI4Bharat IndicVoices) and code-switched
  Hindi-English speech.
- **Number normalization:** convert numbers on both sides to one format before scoring.
- **Comparison:** benchmark an Indic-specific model such as AI4Bharat's wav2vec2 models.
- **Engineering:** package evaluation in Docker, version data with DVC and track runs with MLflow.

## Files

| File | Contents |
|---|---|
| `intern-project.ipynb` | Full Kaggle notebook: baseline, training, evaluation, noise benchmark, error analysis |
| `finetuned_preds.csv` | Reference and fine-tuned prediction for every test utterance |
| `noise_benchmark.csv` | Noise benchmark results |
| `worst50_tagged.csv` | 50 worst utterances with error tags and notes |

## Using the model

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(base, "Adi-13-adi/whisper-small-hi-lora").merge_and_unload()

# audio: 16 kHz mono numpy array
features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
ids = model.generate(features, language="hindi", task="transcribe", max_new_tokens=440)
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```
