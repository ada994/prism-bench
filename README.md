# PRISM Benchmark & FLUX-Reason-6M Dataset

🌟  This is the official repository for the paper "[FLUX-Reason-6M & PRISM-Bench: A Million-Scale Text-to-Image Reasoning Dataset and Comprehensive Benchmark](https://github.com/yuyouxixi/Exp)", which contains both evaluation code and data for the **PRISM Benchmark**.

[[🌐 Homepage](https://github.com/yuyouxixi/Exp)] [[🤗 Huggingface Dataset](https://github.com/yuyouxixi/Exp)] [[📊 Leaderboard ](https://github.com/yuyouxixi/Exp)] [[📊 Leaderboard-ZH ](https://github.com/yuyouxixi/Exp)] [[📖 Paper](https://github.com/yuyouxixi/Exp)]

## 💥 News
- **[2024-09-12]** Our paper is now accessible at [ArXiv Paper](https://github.com/yuyouxixi/Exp).
- **[2025-09-12]** Our FLUX-Reason-6M dataset is now accessible at [huggingface](https://github.com/yuyouxixi/Exp).

## 📈 Evaluation

### Data
Please organize the image data as follows.
```sh
└── images
│   ├── imagination
│   │   ├── 0.png
│   │   ├── 1.png
│   │   ├── ...
│   │   ├── 99.png
│   ├── entity
│   ├── text_rendering
│   ├── style
│   ├── affection
│   ├── composition
│   ├── long_text
```

### PRISM-Bench Evaluation

#### Eval with GPT4.1:

```sh
python evaluation/eval_gpt41.py --image_path <path to image data> --api_key <OpenAI API key> --base_url <OpenAI base URL for custom or proxy endpoints>
```
#### Eval with Qwen2.5-VL-72B:
```sh
python evaluation/eval_qwen25.py --image_path <path to image data> --model_path <path to qwen model> --output_dir <path to save results>
```

### PRISM-Bench-ZH Evaluation

#### Eval with GPT4.1:

```sh
python evaluation/eval_gpt41.py --image_path <path to image data> --api_key <OpenAI API key> --base_url <OpenAI base URL for custom or proxy endpoints> --zh
```
#### Eval with Qwen2.5-VL-72B:
```sh
python evaluation/eval_qwen25.py --image_path <path to image data> --model_path <path to qwen model> --output_dir <path to save results> --zh
```

## 📝 Citation

If you find this benchmark and dataset useful in your research, please consider citing this BibTex:

```

```
