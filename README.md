# Revealing the Technology Evolution of Natural Language Processing: A Scientific Entity-Centric Perspective

## Overview

**Dataset and source code for paper "Revealing the Technology Evolution of Natural Language Processing: A Scientific Entity-Centric Perspective".**

This study analyzes the technology evolution of the NLP field from a scientific entity perspective. Our work includes the followig aspects:
  - We utilize pre-trained language models to identify technology-related entities. Additionally, we have developed a semi-supervised data augmentation technique to increase the number of training samples and improve the model's robustness. This approach has been demonstrated to enhance entity recognition accuracy, achieving an F1 score of 87.00 for the final model.
  - We have developed a semi-automated approach to normalize entities extracted from papers, then randomly sample entity pairs for human judgment on whether they refer to the same entity, achieving a precision of 91.30.
  - Considering that researchers often combine different methods to address a research problem, we measure the impact of technology-related entities by constructing co-occurrence networks of them and computing their z-scores.
  - After completing the above preparation work, we analyze the annual changes in the number of entities, the situation of high-impact new entities, and the variations in the popularity degree and speed of top entities.

## Directory structure

<pre>
technology_evolution                               Root directory
├── Code                                           Source code folder
│   ├── entity-normalization.ipynb                 Source code for entity normalization
│   ├── ner-base.py                                Source code for baseline models
│   ├── ner-cascade.py                             Source code for the SciBERT+BiLSTM(cascade) model
│   └── z-score_calculation .ipynb                 Source code for calculating impact of entities
├── Dataset                                        Dataset folder
│   ├── mapping-list.txt                           "abbreviation-full name" mapping dictionary for entities
│   ├── mdmt.parquet                               Our annotated dataset
│   ├── paper-ents.parquet                         Extracted entities of papers
│   ├── pid2conf.txt                               Dictionary of paper_id to its conference
│   ├── remove-words.txt                           Words with little semantic contribution
│   ├── scierc.parquet                             Open dataset SciERC
│   ├── tdm.parquet                                Open dataset TDM
│   └── top-ents.csv                               Full list of high-impact entities
└── README.md
</pre>

## Dataset Discription

This study encompasses three datasets: our self-annotated dataset and two open datasets, namely SciERC and TDM.

  - <code>./Dataset/mdmt.parquet</code>  Parquet format. Our self-annotated dataset. As further research is required, we make available a portion of the data which includes 500 training samples, 100 validation samples, 100 test samples, and 2493 samples of data augmentation.This dataset comprises three fields, namely 'word', 'label', and 'type'. The field 'type' is used to differentiate the trainset, validset, and testset. The entities we annotated consist of four types: method, dataset, metric, and tool.
<br/><code>'word': ['According', 'to', 'Chen', 'et', 'al', '.', '(', '2016', ')', ',', 'Bilinear', 'outperforms', 'multi-layer', 'forward', 'neural', 'networks', 'in', 'relevance', 'measurement', '.']</code>
<br/><code>'label': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'S-Method', 'O', 'B-Method', 'I-Method', 'I-Method', 'E-Method', 'O', 'O', 'O', 'O']</code>
<br/><code>'type': 'train'</code>

  - <code>./Dataset/scierc.parquet</code>  The open dataset SciERC, consists of six entity types, namely, Task, Method, Metric, Material, Other-ScientificTerm, and Generic. It contains 2687 sentences.

  - <code>./Dataset/tdm.parquet</code>  The open dataset TDM, defining three types of entities: Task, Dataset, and Metric. It contains 2010 sentences.

## Quick Start

  - <b>Technology-related entity recognition</b>
    - <code>python ./Code/ner-cascade.py</code>  Execute this command to run our best model: SciBERT+BiLSTM(cascade).
    - code>python ./Code/ner-base.py</code>  Execute this command to run baseline models. Please utilize various pre-trained models by configuring the parameters in the <code>Config</code> class.

  - <b>Entity normalization</b>
    - <code>./Code/entity-normalization.ipynb</code>  Execute the program step by step in a Jupyter Notebook. Normalizing entities based on edit distance similarity and hierarchical clustering. The relevant resources can be found in the <code>Dataset</code> folder.
    - A total of 534,500 entities were extracted, and the number of entities after normalization was 268,392.  Subsequently, we filtered out entities with an annual frequency of less than 5, and ultimately obtained 37,624 valid technology-related entities. Each valid entity corresponds to a cluster in the clustering result. To verify the effectiveness of entity normalization, we randomly selected 1000 pairs of entities from these entity clusters and manually judged whether they belonged to the same entity. The precision metric, calculated based on the human reviews and the normalization results, was 91.30.

  - <b>z-score calculation</b>
<br/><code>./Code/z-score_calculation.ipynb</code>  Execute the program step by step in a Jupyter Notebook. After completing entity normalization, the co-occurrence networks are constructed based on papers for each year, and the z-scores of entities are calculated to measure their impact.The relevant resources can be found in the <code>Dataset</code> folder.

## Evaluation of entity recognition

Evaluation of models on our annotated dataset

|Model|P|R|F1|
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

<table>
    <tr>
        <td><b>Dataset</b></td>
        <td><b>Authors</b></td>
        <td><b>Model</b></td>
        <td><b>P</b></td>
        <td><b>R</b></td>
        <td><b>F1</b></td>
    </tr>
    <tr>
        <td rowspan="5">SciERC</td>
        <td>Luan et al. (2018)</td>
        <td>SCIIE</td>
        <td>67.2</td>
        <td>61.5</td>
        <td>64.2</td>
    </tr>
    <tr>
        <td>Zhong and Chen (2021)</td>
        <td>PURE</td>
        <td></td>
        <td></td>
        <td>68.9</td>
    </tr>
    <tr>
        <td>Eberts and Ulges (2021)</td>
        <td>SpERT</td>
        <td>70.87</td>
        <td>69.79</td>
        <td>70.33</td>
    </tr>
    <tr>
        <td>Zaratiana et al. (2022)</td>
        <td>Hierarchical Transformer</td>
        <td>67.99</td>
        <td>74.11</td>
        <td><b>70.91</b></td>
    </tr>
    <tr>
        <td>Our</td>
        <td>SciBERT+BiLSTM (cascade)+data_aug</td>
        <td>66.95</td>
        <td>71.49</td>
        <td>69.14</td>
    </tr>
    <tr>
        <td rowspan="3">TDM</td>
        <td>Hou et al. (2021)</td>
        <td>SCIIE</td>
        <td>67.17</td>
        <td>58.27</td>
        <td>62.4</td>
    </tr>
    <tr>
        <td>Zaratiana et al. (2022)</td>
        <td>Hierarchical Transformer</td>
        <td>65.56</td>
        <td>70.21</td>
        <td>67.81</td>
    </tr>
    <tr>
        <td>Our</td>
        <td>SciBERT+BiLSTM (cascade)+data_aug</td>
        <td>68.84</td>
        <td>70.73</td>
        <td><b>69.77</b></td>
    </tr>
</table>

## High-impact technology-related entities

Entities with z-scores exceeding 2.5 are defined as high-impact technology-related entities. Since 2001, 179 high-impac new entities have emerged in the NLP field, and their complete list can be found in the file "*./Dataset/top-ents.csv*".

The top 5 entities for each type are as follows:

<table>
    <tr>
        <td><b>Type</b></td>
        <td><b>Entity</b></td>
        <td><b>z-score</b></td>
        <td><b>Type</b></td>
        <td><b>Entity</b></td>
        <td><b>z-score</b></td>
    </tr>
    <tr>
        <td rowspan="5">Method</td>
        <td>BERT</td>
        <td>43.3138</td>
        <td rowspan="5">Metric</td>
        <td>BLEU</td>
        <td>15.9303</td>
    </tr>
    <tr>
        <td>Transformer</td>
        <td>34.6696</td>
        <td>Cross-Entropy</td>
        <td>13.1292</td>
    </tr>
    <tr>
        <td>LSTM</td>
        <td>28.8231</td>
        <td>ROUGE</td>
        <td>7.8905</td>
    </tr>
    <tr>
        <td>Attention Mechanism</td>
        <td>26.2604</td>
        <td>Fluency</td>
        <td>6.9009</td>
    </tr>
    <tr>
        <td>Adam</td>
        <td>20.3561</td>
        <td>Standard Deviation</td>
        <td>6.1762</td>
    </tr>
    <tr>
        <td rowspan="5">Dataset</td>
        <td>Wikipedia</td>
        <td>17.4187</td>
        <td rowspan="5">Tool</td>
        <td>PyTorch</td>
        <td>6.1565</td>
    </tr>
    <tr>
        <td>MNLI</td>
        <td>6.7163</td>
        <td>MOSES</td>
        <td>5.3327</td>
    </tr>
    <tr>
        <td>SQuAD</td>
        <td>5.783</td>
        <td>GIZA++</td>
        <td>5.2089</td>
    </tr>
    <tr>
        <td>Twitter</td>
        <td>5.3056</td>
        <td>TensorFlow</td>
        <td>3.563</td>
    </tr>
    <tr>
        <td>SST-2</td>
        <td>5.2605</td>
        <td>Stanford Parser</td>
        <td>3.2967</td>
    </tr>
</table>

## Dependency packages
System environment is set up according to the following configuration:
- pytorch 2.0.1
- transformers 4.28.1
- pandas 2.0.0
- pytorch-crf 0.7.2
- tqdm 4.65.0
- loguru 0.7.0
- fasttext 0.9.2
- flashtext 2.7
- nltk 3.8.1
- thefuzz 0.19.0
- numpy 1.24.1

## Citation
Please cite the following paper if you use this code and dataset in your work.
    
>Heng Zhang, Chengzhi Zhang, Yuzhuo Wang. Revealing the Technology Evolution of Natural Language Processing: A Scientific Entity-Centric Perspective. ***Information Processing and Management***, 2023 (Under Review).
