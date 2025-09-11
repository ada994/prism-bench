<div align="center">

# PRISM Benchmark & FLUX-Reason-6M Dataset

[[🌐 Homepage](https://flux-reason-6m.github.io/)] [[🤗 Huggingface Dataset](https://huggingface.co/datasets/LucasFang/FLUX-Reason-6M)] [[📊 Leaderboard ](https://flux-reason-6m.github.io/)] [[📊 Leaderboard-ZH ](https://flux-reason-6m.github.io/)] [[📖 Paper](https://flux-reason-6m.github.io/)]

[Rongyao Fang](https://rongyaofang.github.io/)<sup>1*</sup>&ensp; [Aldrich Yu](https://aldrichyu.github.io/)<sup>1*</sup>&ensp; [Chengqi Duan](https://scholar.google.com/citations?user=r9qb4ZwAAAAJ&hl=en)<sup>2*</sup>&ensp; [Linjiang Huang](https://leonhlj.github.io/)<sup>3</sup>&ensp; [Shuai Bai](https://scholar.google.com/citations?user=ylhI1JsAAAAJ&hl=zh-CN)<sup>4</sup>&ensp; 

[Yuxuan Cai](https://scholar.google.com/citations?user=EzYiBeUAAAAJ&hl=en)<sup>4</sup>&ensp; [Kun Wang](https://openreview.net/profile?id=~Kun_Wang8)&ensp; [Si Liu](https://scholar.google.com/citations?user=-QtVtNEAAAAJ&hl=en)<sup>3</sup>&ensp; [Xihui Liu](https://xh-liu.github.io/)<sup>2†</sup>&ensp; [Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/)<sup>1†</sup>&ensp;

<sup>1</sup>CUHK&ensp;&ensp; <sup>2</sup>HKU&ensp;&ensp; <sup>3</sup>BUAA&ensp;&ensp; <sup>4</sup>Alibaba&ensp;&ensp; 

<sup>*</sup>Equal Contribution&ensp; <sup>†</sup>Corresponding Author

</div>

## 📖 Introduction

🌟  This is the official repository for the paper "[FLUX-Reason-6M & PRISM-Bench: A Million-Scale Text-to-Image Reasoning Dataset and Comprehensive Benchmark](https://flux-reason-6m.github.io/)", which contains both evaluation code and data for the **PRISM Benchmark**.

<p align="center">
  <img src="assets/teaser.png" alt="Teaser" width="1000"/>
</p>

We introduce **FLUX-Reason-6M** and **PRISM-Bench**. FLUX-Reason-6M is a **6-million-scale** synthesized dataset designed to incorporate reasoning capabilities into the architecture of T2I generation. PRISM-Bench serves as a comprehensive and discriminative benchmark with **7 independent tracks** that closely align with human judgment.

## 💥 News
- **[2024-09-12]** Our paper is now accessible at [ArXiv Paper](https://flux-reason-6m.github.io/).
- **[2025-09-12]** Our FLUX-Reason-6M dataset is now accessible at [huggingface](https://huggingface.co/datasets/LucasFang/FLUX-Reason-6M).

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
