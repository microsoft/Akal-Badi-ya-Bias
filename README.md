# Akal Badi ya Bias: An Exploratory Study of Gender Bias in Hindi Language Technology

**Official Repository for ["Akal Badi ya Bias: An Exploratory Study of Gender Bias in Hindi Language Technology"](https://dl.acm.org/doi/abs/10.1145/3630106.3659017) (Presented at FAccT 2024)**

## Abstract
Existing research in measuring and mitigating gender bias predominantly centers on English, overlooking the intricate challenges posed by non-English languages and the Global South. This paper presents the first comprehensive study delving into the nuanced landscape of gender bias in Hindi, the third most spoken language globally. Our study employs diverse mining techniques, computational models, field studies and sheds light on the limitations of current methodologies. Given the challenges faced with mining gender biased statements in Hindi using existing methods, we conducted field studies to bootstrap the collection of such sentences. Through field studies involving rural and low-income community women, we uncover diverse perceptions of gender bias, underscoring the necessity for context-specific approaches. This paper advocates for a community-centric research design, amplifying voices often marginalized in previous studies. Our findings not only contribute to the understanding of gender bias in Hindi but also establish a foundation for further exploration of Indic languages. By exploring the intricacies of this understudied context, we call for thoughtful engagement with gender bias, promoting inclusivity and equity in linguistic and cultural contexts beyond the Global North.

## Directory Structure
```bash
.
├── README.md
├── corgi_classifier
│   └── corgi.ipynb
├── corgi_pm
│   ├── AGSS-zh-CN-hi.xlsx
│   ├── corgipm.py
│   └── eng-hi_adj.csv
├── data
│   └── data_mining_annotations.xlsx
├── fsb_scorer
│   ├── eval.py
│   └── train.py
├── labse_mining
│   ├── create_embeddings_labse_gpu.py
│   ├── mine_similar_pairs.py
│   └── sample.sh
└── translation
    └── translate.ipynb
```

## Dataset
- The datasets are available in the `data/` directory. Please refer to our [paper](https://dl.acm.org/doi/abs/10.1145/3630106.3659017) for the dataset curation and annotation methodology. 
- <span style="color: red;">**As of now, the dataset is NOT available and is under review before open-sourcing.**</span>

## Citation
In you use this dataset, or code-base, please cite our works,
```bibtex
@inproceedings{10.1145/3630106.3659017,
    author = {Hada, Rishav and Husain, Safiya and Gumma, Varun and Diddee, Harshita and Yadavalli, Aditya and Seth, Agrima and Kulkarni, Nidhi and Gadiraju, Ujwal and Vashistha, Aditya and Seshadri, Vivek and Bali, Kalika},
    title = {Akal Badi ya Bias: An Exploratory Study of Gender Bias in Hindi Language Technology},
    year = {2024},
    isbn = {9798400704505},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3630106.3659017},
    doi = {10.1145/3630106.3659017},
    abstract = {Existing research in measuring and mitigating gender bias predominantly centers on English, overlooking the intricate challenges posed by non-English languages and the Global South. This paper presents the first comprehensive study delving into the nuanced landscape of gender bias in Hindi, the third most spoken language globally. Our study employs diverse mining techniques, computational models, field studies and sheds light on the limitations of current methodologies. Given the challenges faced with mining gender biased statements in Hindi using existing methods, we conducted field studies to bootstrap the collection of such sentences. Through field studies involving rural and low-income community women, we uncover diverse perceptions of gender bias, underscoring the necessity for context-specific approaches. This paper advocates for a community-centric research design, amplifying voices often marginalized in previous studies. Our findings not only contribute to the understanding of gender bias in Hindi but also establish a foundation for further exploration of Indic languages. By exploring the intricacies of this understudied context, we call for thoughtful engagement with gender bias, promoting inclusivity and equity in linguistic and cultural contexts beyond the Global North.},
    booktitle = {Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency},
    pages = {1926–1939},
    numpages = {14},
    keywords = {Community centric, Gender bias, Global South, Hindi, India, Indic languages},
    location = {Rio de Janeiro, Brazil},
    series = {FAccT '24}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.


## Privacy

You can read more about Microsoft's privacy statement [here](https://go.microsoft.com/fwlink/?LinkId=521839).
