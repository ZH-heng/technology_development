### **Reveal Technology Evolution of NLP from the Scientific Entities Perspective**

### 1. Technology-related entity recognition

- **Dataset**

    './data/mdmt.parquet': Parquet format. Our self-annotated dataset. As further research is required, we make available a portion of the data which includes 500 training samples, 100 validation samples, and 100 test samples. This dataset comprises three fields, namely 'word', 'label', and 'type'. The field 'type' is used to differentiate the trainset, validset, and testset. The entities we annotated consist of four types: method, dataset, metric, and tool.

    *'word': ['According', 'to', 'Chen', 'et', 'al', '.', '(', '2016', ')', ',', 'Bilinear', 'outperforms', 'multi-layer', 'forward', 'neural', 'networks', 'in', 'relevance', 'measurement', '.']*

    *'label': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'S-Method', 'O', 'B-Method', 'I-Method', 'I-Method', 'E-Method', 'O', 'O', 'O', 'O']*

    *'type': 'train'*

    

    './data/scierc.parquet': The open dataset SciERC, consists of six entity types, namely, Task, Method, Metric, Material, Other-ScientificTerm, and Generic.

    './data/tdm.parquet': The open dataset TDM, defining three types of entities: Task, Dataset, and Metric

- **Code**

    'ner-base.py': Code for baseline models.

    'ner-cascade.py': Code for the SciBERT+BiLSTM(cascade) model.

- **Model evaluation**

    Evaluation of models on our annotated dataset

|**Model**|**P**|**R**|**F1**|
|-|-|-|-|
|BERT|83.64|85.92|84.77|
|BERT+CRF|84.10|84.45|84.28|
|RoBERTa|85.84|81.51|83.62|
|RoBERTa+CRF|83.86|82.98|83.42|
|T5|84.41|85.29|84.85|
|T5+ CRF|85.08|85.08|85.08|
|SciBERT|85.95|86.13|86.04|
|SciBERT+CRF|83.53|87.39|85.42|
|SciBERT+BiLSTM (cascade)|86.22|86.76|86.49|
|SciBERT+BiLSTM (cascade)+data_aug|86.82|87.18|**87.00**|



Evaluation on SciERC and TDM dataset

|**Dataset**|**Authors**|**Model**|**P**|**R**|**F1**|
|-|-|-|-|-|-|
|SciERC|Luan et al. (2018)|SCIIE| 67.20 | 61.50 | 64.20 |
||Zhong and Chen (2021)|PURE||| 68.90 |
||Eberts and Ulges (2021)|SpERT| 70.87 | 69.79 | 70.33 |
||Zaratiana et al. (2022)|Hierarchical Transformer| 67.99 | 74.11 | **70.91** |
||Our|SciBERT+BiLSTM (cascade)+data_aug| 66.95 | 71.49 | 69.14 |
|TDM|Hou et al. (2021)|SCIIE| 67.17 | 58.27 | 62.40 |
||Zaratiana et al. (2022)|Hierarchical Transformer| 65.56 | 70.21 | 67.81 |
||Our|SciBERT+BiLSTM (cascade)+data_aug| 68.84 | 70.73 | **69.77** |



