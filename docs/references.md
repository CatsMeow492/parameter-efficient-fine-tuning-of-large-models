# References

## Core Papers Reviewed

### LoRA and Low-Rank Adaptation Methods

**LoRA: Low-Rank Adaptation of Large Language Models**  
*Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*  
**arXiv:** https://arxiv.org/abs/2106.09685  
**Year:** 2021  
**Summary:** Introduces LoRA method for parameter-efficient fine-tuning using low-rank matrices

**QLoRA: Efficient Finetuning of Quantized LLMs**  
*Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer*  
**arXiv:** https://arxiv.org/abs/2305.14314  
**Year:** 2023  
**Summary:** Combines LoRA with 4-bit quantization for extreme memory efficiency

**AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning**  
*Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, Tuo Zhao*  
**arXiv:** https://arxiv.org/abs/2303.10512  
**Year:** 2023  
**Summary:** Adaptive rank allocation for LoRA based on importance scoring

### Adapter Methods

**Parameter-Efficient Transfer Learning for NLP**  
*Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly*  
**arXiv:** https://arxiv.org/abs/1902.00751  
**Year:** 2019  
**Summary:** Seminal work introducing adapter modules for parameter-efficient transfer learning

### Prompt-Based Methods

**Prefix-Tuning: Optimizing Continuous Prompts for Generation**  
*Xiang Lisa Li, Percy Liang*  
**arXiv:** https://arxiv.org/abs/2101.00190  
**Year:** 2021  
**Summary:** Learn task-specific prefix vectors while keeping model weights frozen

## Additional Resources

### Foundational Transformer Papers
- **Attention Is All You Need** (Vaswani et al., 2017): https://arxiv.org/abs/1706.03762
- **BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018): https://arxiv.org/abs/1810.04805

### Large Language Models
- **LLaMA: Open and Efficient Foundation Language Models** (Touvron et al., 2023): https://arxiv.org/abs/2302.13971
- **LLaMA 2: Open Foundation and Fine-Tuned Chat Models** (Touvron et al., 2023): https://arxiv.org/abs/2307.09288

### Parameter-Efficient Fine-Tuning Surveys
- **Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning** (Lialin et al., 2023): https://arxiv.org/abs/2303.15647

## Citation Format

```bibtex
@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}

@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}

@article{houlsby2019parameter,
  title={Parameter-efficient transfer learning for NLP},
  author={Houlsby, Neil and Giurgiu, Andrei and Jastrzebski, Stanislaw and Morrone, Bruna and De Laroussilhe, Quentin and Gesmundo, Andrea and Attariyan, Mona and Gelly, Sylvain},
  journal={arXiv preprint arXiv:1902.00751},
  year={2019}
}

@article{zhang2023adalora,
  title={AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning},
  author={Zhang, Qingru and Chen, Minshuo and Bukharin, Alexander and He, Pengcheng and Cheng, Yu and Chen, Weizhu and Zhao, Tuo},
  journal={arXiv preprint arXiv:2303.10512},
  year={2023}
}

@article{li2021prefix,
  title={Prefix-tuning: Optimizing continuous prompts for generation},
  author={Li, Xiang Lisa and Liang, Percy},
  journal={arXiv preprint arXiv:2101.00190},
  year={2021}
}
```

---

**Note:** All arXiv links provide direct access to PDFs for reproducible research. BibTeX entries are formatted for LaTeX paper writing. 