# Flan-T5 Text-to-SQL: WikiSQL → Spider

Fine-tunes [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) for text-to-SQL
generation in two stages:

1. **`Flan-T5-WikiSQL.ipynb`** — fine-tunes the base model on [WikiSQL](https://github.com/salesforce/WikiSQL).
2. **`Flan-T5-Spider.ipynb`** — loads that checkpoint and continues fine-tuning on the harder,
   multi-table [Spider](https://yale-lily.github.io/spider) dataset, then serves an interactive
   Gradio demo.

## Results

| Dataset | Split      | Exact match |
|---------|------------|-------------|
| WikiSQL | test       | see `wikisql_predictions.csv` after running |
| Spider  | validation | ~23% (development run, 10 epochs, single T4) |

Numbers depend on how many epochs / samples you train with — see the training config cells
in each notebook.

## Environment

Both notebooks were built and run on **Kaggle** with a **GPU T4 x2** accelerator, and write
their outputs to `/kaggle/working/`. They will raise a `RuntimeError` if no GPU is available.

To run them elsewhere (Colab, a local machine with a GPU, etc.):
- Replace the `/kaggle/working/...` and `/kaggle/input/...` paths with local directories.
- In `Flan-T5-Spider.ipynb`, replace the `/kaggle/input/` checkpoint-discovery logic with a
  direct path to your WikiSQL checkpoint.

## Running order

1. Run `Flan-T5-WikiSQL.ipynb` end to end. It saves a checkpoint to `WIKISQL_FINAL_DIR`
   (`/kaggle/working/wikisql-flan-t5-large` by default).
2. On Kaggle: add that notebook's output as an input to `Flan-T5-Spider.ipynb`
   (*Notebook → Add Input → Your Work*). Locally: point the checkpoint-loading cell at the
   directory from step 1.
3. `Flan-T5-Spider.ipynb` also needs Spider's `tables.json` schema file available under
   `/kaggle/input/` (the notebook prints manual upload instructions if it can't find one).
4. Run `Flan-T5-Spider.ipynb` end to end.

## Setup

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is required for training in a reasonable amount of time; both notebooks
also run beam-search generation for evaluation, which is slow on CPU.

## Repo structure

```
.
├── Flan-T5-WikiSQL.ipynb   # stage 1: base model -> WikiSQL fine-tune
├── Flan-T5-Spider.ipynb    # stage 2: WikiSQL checkpoint -> Spider fine-tune + Gradio demo
├── requirements.txt
└── README.md
```

## Notes

- Training uses mixed precision (`fp16`) and gradient checkpointing to fit on a single T4.
- The Gradio demo at the end of the Spider notebook launches with `share=True`, which creates
  a temporary public URL — remove that argument if you don't want one.
