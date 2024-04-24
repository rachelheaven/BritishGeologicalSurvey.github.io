---
title:  "AI Natural Language Processing: Entity and Relation Extraction from Geological Text"
author: Rachel Heaven
categories:
  - Data Science
tags:
  - NLP
---


This blog post describes the methodology and lessons learned in the creation of [labelled data](https://github.com/BritishGeologicalSurvey/princeton-nlp-relation-extraction/geo_data) 
 and its use in training of a proof of concept Entity Relation Extraction model. 
The intention is that a model fine-tuned with this data can be used in inference mode to extract structured geological knowledge from where it is currently hidden in large corpus of archive reports.

# Acknowledgement

This work was funded by the UK Government Office for Technology Transfer, Knowledge Asset Grant Fund, 
project 10083604 "Using AI Natural Language Processing to unlock data from hidden geological records to aid Net Zero".

# What is Entity Relation Extraction and why are we doing this ?

Entity Relation Extraction is a AI Natural Language Processing (NLP) model that can read passages of text and identify entities of interest, and how those entities are related. 
We are targetting knowledge useful for subsurface characterisation such as rock formations, lithologies, ages, locations and physical properties like temperature and permeability.
 This information is necessary for a range of applications towards Net Zero and sustainable development including
geotechnical engineering, geothermal heat extraction, radioactive waste disposal,
groundwater research and critical mineral resources assessments required for green technologies such as batteries and solar panels.
The knowledge needed to understand these issues is embedded in
tens of thousands of text reports at BGS within the National Geoscience Data Centre (NGDC), but manually trawling through these to find and
extract the relevant details is too labour intensive and these valuable public sector knowledge assets are under-utilised.

Using AI to create a geoscience-aware automated method to extract information will support the
expanded use of geological text archives by making the pertinent information
readily available in machine readable form, hence available for data analytics, spatial modelling applications and decision support tools.


# How we go about this

There are various methods available to tackle this problem with NLP tools. We chose to follow a supervised method: fine tuning a pre-trained transformers Large Language Model using transfer learning, using the wrapper code from a project at Princeton University.
This method requires some labelled training data to be supplied to teach the model about our domain.
Training data was created from 3 of BGS's publicly available document corpora:  memoirs, DECC Land wells and Mineral Reconnaissance Programme (MRP) reports

# Document corpora for training data

## Memoirs
 
A sample of publications (memoirs and technical reports) of the British Geological Survey, previously used as training data in https://github.com/BritishGeologicalSurvey/geo-ner-model and already available in machine readable form as high quality machine readable text. The full reports can be viewed at https://webapps.bgs.ac.uk/Memoirs/index.html
The previous training data for geo-ner-model consists of 3000 sentences, with geological ages and rock formations annotated with custom entity types CHRONOSTRAT and LEXICON respectively (named after the BGS Lexicon of Named Rock Units)
We processed this data to re-purpose it, with the use of bash and python scripts in which:

 - utf8 encoding errors fixed
 - `CHRONOSTRAT` labels were renamed to CSTRAT to make shorter label name
 - `LEXICON` labels were renamed to LSTRAT to make shorter and more generic label name
 - The `CONLL PREFIX` format used in geo-ner-model was converted to `CONLL-IOB2`
 - `CONLL-IOB2` format was loaded into a local instance of [doccano annotation software](https://github.com/doccano/doccano)
 - data was exported from doccano as `JSONL`, which identifies each sentence as a separate text passage.
 
 
## DECC Land Wells reports

A sample of well reports and drilling reports from onshore hydrocarbon boreholes held by BGS in the National Geoscience Data Centre (NGDC) 
and [released to the public in 2023](https://www.bgs.ac.uk/news/a-new-open-dataset-to-benefit-onshore-geoscience-research/)
Reports were downloaded from https://webapps.bgs.ac.uk/services/ngdc/accessions/index.html?simpleText=DECC/OGA
An index database exists for the corpus for internal staff us, and a subset of the reports was identified and analysed to determine the presence of any text layer and the quality of scan images.
Python scripts were created to OCR and convert these pdf reports to per-page, per-sentence/paragraph plain text files and then to jsonlines (JSONL) format required by doccano, using the libraries:

```
pdf2image
pytesseract
PyMuPDF
textblob
nltk
```

![doccano jsonlines format for imported data](../../assets/images/2024-03-14-nlp-entity-relation-extraction/doccano_jsonl.png)

 The reports are of mixed quality, but mostly old typeset, lots of data tables and diagrams and some handwritten pages. The OCR results are therefore relatively poor quality.

## MRP

A sample of reports from the Mineral Reconnaissance Programme (MRP) which provided geological, geochemical, geophysical, mineralogical and metallogenic information
 on prospective areas in Britain from the early 1907s to 1997. Reports were downloaded from the BGS hosted [MineralsUK website](https://www2.bgs.ac.uk/mineralsuk/exploration/potential/mrp.html).
These were processed in the same way as the DECC Landwells, and the OCR's text is of a similar mixed quality.


# Text pre-processing

## Spelling correction

We experimented with improving the OCR text using `textblob` spelling correction, including adding custom items to the dictionary using terms fetched from geoscience dictionary labels. 
We found this was too agressive, changing correct words such as "source" to "course", and was omitted from the processing. 
More work is needed to tune this processing step, and to consider other OCR quality improvements e.g. to train a custom spelling correction algorithm appropriate for the errors typically generated from OCR of the typeset used.

## Auto labelling

### Dictionary based auto-labelling

To speed up the manual labelling, the JSONL text were first auto-labelled using terms from a range of dictionaries appropriate to each entity label type.
Dictionary labels were fetched from SPARQL endpoints

`https://cgi.vocabs.ga.gov.au` 

`https://data.bgs.ac.uk`

`https://www.qudt.org`

This processing ran very slowly on the poor quality OCR text as there were so many separate tokens due to numerical tables and split words, and also runs slowly for dictionaries containing several thousand terms such as the BGS Lexicon of Named Rock Units. 
We performed this processing on our local machines using scripts, though doccano does provide allow configuration to use auto-labelling through an API if one is available.
NB The dictionaries downloaded here can also potentially be used in the results of model inference/prediction runs to perform Entity Linking - matching the labelled terms to specific vocabulary items where possible, and in custom spelling correction as explained above.

### auto-labelling using third party NLP model

A recent third party NER model for geoscience data, [GeoBERT](https://huggingface.co/botryan96/GeoBERT), trained on ~1 million sentences, was applied to automatically apply 4 types of NER labels for geoscience: 
 - geological age (`GeoTime`, equivalent to `CSTRAT` in our label set below)
 - rock type (`GeoPetro`, roughly equivalent to `LITH`)
 - location (`GeoLoc` equivalent to `LOC`)
 - methods (`GeoMeth`, no equivalent)
 
Once inspected in the manual labelling process, it was apparent that this model did not perform well enough on our documents, and resulted in a lot of false positive labels. 
The `GeoMeth` label appeared to cover a wide range of entity types and some abbreviations - these labels were largely discarded, though some were converted to GEOTHERMAL or to QKIND.
 The processing was performed locally using scripts, not through the doccano auto-label interface.

## Manual labelling

The auto-labelled jsonl documents were loaded to a doccano `Sequence Labelling` project, configured to handle relation labels and overlapping spans.
Using the doccano interface: 
 - Auto-labels were edited as required to remove false positives.
 - Additional labels were added that the auto-labelling had not identified, or where no dictionary was available for auto-labelling
 - doccano failed to import the Memoirs `CONLL-IOB2` tags correctly, assigning a separate label to each token of a multi-word phrase rather than a single label. 
   These were corrected within doccano during manual labelling, but an outstanding issue is to resolve this programmatically as it causes extra work during this task.
 - Relation tags were added to link pairs of entities

Auto-labelling should be inspected before manual labelling commences, to check the auto-labelling is performing as required. If the auto-labelling is not precise (i.e. adds labels where it shouldn’t) it creates more manual effort to correct, rather than saving manual labelling effort!
Relation tags were determined through an iterative process, balancing the desire to capture as much information as possible whilst ensuring sufficient labelled examples of each type and a balanced training dataset.

Doccano allows you to search within the loaded sentences for terms and specific labels. Knowing that we likely wouldn't have time in this funded project to label the entire datasets, we searched for sentences that contained terms that were useful for our labelling, rather than stepping through all sentences in sequence order. This may produce bias in the training data and ideally the entire source datasets or a large but random selection of sentences should be labelled.  We targetted the sentences containing 'porosity','permeability','heat flow' (relevant for the geothermal potential use case), 'thickness','overlies','underlies' (relevant for 3D stratigraphic analysis), and 'copper' for a specific use case for critical minerals needed for green technologies.

![Example paragraph from the memoirs labelled in doccano ](../../assets/images/2024-03-14-nlp-entity-relation-extraction/memoirs_label.png)

![Example paragraph from an onshore hydrocarbons well report labelled in doccano ](../../assets/images/2024-03-14-nlp-entity-relation-extraction/DECCLandwells_label.png)

![Example paragraph from a MRP report labelled in doccano ](../../assets/images/2024-03-14-nlp-entity-relation-extraction/MRP_label.png)

‘Images of doccano software. Copyright © 2018 TIS inc.’

## Training data format

Sentences with completed labelling were exported from doccano in their JSONL(Relations) format. 

![Example paragraph from the memoirs labelled in doccano ](../../assets/images/2024-03-14-nlp-entity-relation-extraction/doccano_jsonl_relations.png)

A script was written to convert these to the jsonl training format required by PURE (https://github.com/princeton-nlp/PURE#input-data-format-for-the-entity-model)

The PURE jsonl training file was shuffled and split into a train, test and validation set (referred to in PURE as train/dev/test) in a ratio of 60/20/20 and saved in file names expected by the PURE training scripts.

Both these formats are provided in the data directory of [our fork of the PURE project](https://github.com/BritishGeologicalSurvey/princeton-nlp-relation-extraction)

See [geo_data/labelled_data](https://github.com/BritishGeologicalSurvey/princeton-nlp-relation-extraction/geo_data/labelled_data)



## Label sets 

### Entity label set

The final set of entity labels submitted for training was:

label | dictionary source | dictionary | consolidated label | 
 --- | --- | --- | ---
COMCAT  | https://cgi.vocabs.ga.gov.au | compositioncategory | COMCAT
EVTPCS  | https://cgi.vocabs.ga.gov.au | eventprocess | EVTPCS
COMCDE  | https://cgi.vocabs.ga.gov.au | commodity-code | COMCDE
EDUPNT  | https://cgi.vocabs.ga.gov.au | end-use-potential | EDUPNT
LITH  | https://cgi.vocabs.ga.gov.au | simplelithology | LITH
LITH  | https://data.bgs.ac.uk | EarthMaterialClass | LITH
CSTRAT  | https://data.bgs.ac.uk | Chronostratigraphy, geological age | CSTRAT
LSTRAT  | https://data.bgs.ac.uk | Lithostratigraphic formation | LSTRAT
QKIND  | https://www.qudt.org | QuantityKind - a property type | QKIND
QUNIT  | https://www.qudt.org | Unit - unit of a measured or observed property value | QUNIT
QVAL | n/a | quantity value - combination of numerical value and unit | QVAL
LOC | n/a though terms can be matched to various gazetteers | Location | LOC
FEAT | n/a | Subsurface or geographical feature | 
GEOTHERM | list supplied by BGS researchers  | selection of terms relating to geothermal research; may contain terms from other categories above | GEOTHERM

### Relation label set

relation label | description 
--- | ---
hasLithology | links a LSTRAT to a LITH
isPartOf | links any entity to another entity of the same type 
hasAge | links a LSTRAT or FEAT to CSTRAT
overlies | links a LSTRAT to another LSTRAT
hasProperty | links a LSTRAT or FEAT to a QVAL or GEOTHERMAL 
hasObservationContext | consolidates observedAtLocation (links any of the other entities to a LOC) and observedInFormation  (links any of the other entities to a LSTRAT) |
formedInEnvironment | links a LSTRAT to ENV
valueOf | links a QVAL to a QKIND
lateralEquivalent | links a LSTRAT to another LSTRAT

## Using the labelled data to train a model

We use the code provided by a Princeton University project
 [PURE: Entity and Relation Extraction from Text](https://github.com/princeton-nlp/PURE) - which is wrapper code for the huggingface transformers API - to perform transfer learning
 to fine tune a pre-trained huggingface Large Languagel Model (LLM).

PURE provides options for a few different LLM.
We have used the transformer model [allenai/scibert_scivocab_uncased](https://huggingface.co/allenai/scibert_scivocab_uncased) in the assumption that will be better for scientific data. 
Further work could experiment with other base models within PURE.

As an alternative to PURE - which is 3 years old and needs updating - fresh code could be written to use the Huggingface Transformer library to configure the model fine tuning. It's likely that the doccano jsonlines format will need converted, although other open source code 

### Environment and dependencies

We ran the PURE model training code on ubuntu 22.04 (WSL on windows 11), in a python 3.7 environment. PURE does not work on later versions of python. A small batch size was required to avoid out of memory errors and it took a few hours; training on GPU and high-RAM compute would be expected to be considerably faster. Apart from the batch size, the default hyper-parameters provided in PURE were used for this PoC.  

Minor changes to the PURE code were needed because of out of support libraries.
More extensive changes would be needed to update all the code to work with more recent python versions and the most recent transformers library.

### Training command

With the checked out repository https://github.com/BritishGeologicalSurvey/princeton-nlp-relation-extraction 

# Train and evaluate the entity model

```
geo_dataset='geo_data/labelled_data/memoirs/PURE_jsonl'
mkdir geo_models
mkdir geo_models/ent-scib-ctx0
geo_model_ent='geo_models/ent-scib-ctx0/'

python run_entity.py \
    --do_train --do_eval \
    --learning_rate=1e-5 --task_learning_rate=5e-4 \
    --train_batch_size=4 \
    --context_window  300  \
    --task  geo_data \
    --data_dir ${geo_dataset} \
    --model allenai/scibert_scivocab_uncased \
    --output_dir ${geo_model_ent}

# The resulting entities (ner) model is stored in 'geo_models/ent-scib-ctx0/'
# the resulting new predicted entities will be stored in 'geo_models/ent-scib-ctx0/ent_pred_dev.json'

# Train the full relation model

geo_dataset_rel_train='geo_data/labelled_data/memoirs/PURE_jsonl/train.json'
mkdir geo_models/rel-scib-ctx0
geo_model_rel=geo_models/rel-scib-ctx0

python run_relation.py \
  --task geo_data \
  --do_train --train_file ${geo_dataset_rel_train} \
  --do_eval --eval_test \
  --eval_with_gold \
  --model allenai/scibert_scivocab_uncased \
  --do_lower_case \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --context_window 100 \
  --max_seq_length 128  \
  --entity_output_dir ${geo_model_ent} \
  --output_dir ${geo_model_rel}
  
# Output end-to-end evaluation results

python run_eval.py --prediction_file ${geo_model_rel}/predictions.json

```

## Evaluation

Evaluation results are shown here for a model training run using 178 paragraphs from the memoirs corpus. The best model was obtained after 6 epochs, after which the model loss started increasing, indicating that the model started over-fitting. 

The evaluation below indicates that the model was able to identify approximately 20% of the expected relation labels (recall measure 0.20), and out of the relation labels it predicts, approximately 40% of those are correct (precision measure 0.40).

[F1 score](https://en.wikipedia.org/wiki/F-score) is the harmonic mean of precision and recall measures.

```
Epoch: 6, Step: 348 / 585, used_time = 3632.76s, loss = 0.214742
***** Eval results *****
  accuracy = 0.9
  eval_loss = 0.5276005428926847
  f1 = 0.2702702702702703
  n_correct = 20
  n_gold = 99
  n_pred = 49
  precision = 0.40816326530612246
  recall = 0.20202020202020202
  task_f1 = 0.26490066225165565
  task_ngold = 102
  task_recall = 0.19607843137254902
!!! Best dev f1 (lr=2e-05, epoch=6): 27.03
Saving model to geo_models/rel-scib-ctx0
``

# Using the model

Refer to [PURE](https://github.com/princeton-nlp/PURE) repository for how to use a trained model to infer/predict entities and relations in unseen text.

# Conclusion and next steps

This performance measure is not yet high enough for a usable model. Creating more training data from the higher quality text of the memoirs corpus would be the first thing to address to improve the model performance. 
but also tuning the model hyperparameters and running on GPU to speed up the process and allow larger batch sizes. There are a large number of labels, and some of these only have
a few sampling points which will perform badly in the training and bring the average performance score down, so we should interrogate the evaluation to get per-label metrics and consider training only on the labels that have a large number of examples.
To remove any bias from the choice of test/train split, the model could be trained multiple times on all possible splits and average performance used.
A small amount of additional labelled data from DECC Landwells (63 paragraphs) and Mineral Reconnaissance Programme reports (11 paragraphs) was created, but when these were included 
in the training data it degraded the model performance. The OCR text from these two resources is poorer quality due to the old typeset, and due to the presence of tables, figures, handwritten sections. In addition, the sample size
from both these corpus was too small for the model to learn well from.

To fully meet the aims of this project - beyond a proof of concept - the model needs to work well on these old archive reports, so the next steps should be: 
- create more labelled samples from the DECC landwells and MRP (or other minerals archives) reports
- identify and isolate the tables and figures from the continuous text
- if necessary, optimize the OCR output through image pre-processing, training a bespoke OCR model on the typeset used, or using licenced software (eg.g. Amazon Textract) which has been reported to perform better than the open source `tesseract` library we use.

The PURE code could be updated to use latest python version and the latest transformers library. Alternative frameworks can be tried, such as SpaCy [2], for example https://github.com/explosion/projects/tree/v3/tutorials/rel_component, or https://github.com/synlp/RE-AGCN [3]

When we are able to produce a more performant model, the built model could be hosted elsewhere on BGS data sharing resources or on third party model hubs such as huggingface. 

In the meantime, the labelled data is provided as a resource for others to experiment with in model training, or to extend with their own additional labelling.


# References

[1] PURE project (MIT license)

```
@inproceedings{zhong2021frustratingly,
   title={A Frustratingly Easy Approach for Entity and Relation Extraction},
   author={Zhong, Zexuan and Chen, Danqi},
   booktitle={North American Association for Computational Linguistics (NAACL)},
   year={2021}
}
```

[2] https://github.com/explosion/projects/tree/v3/tutorials/rel_component (MIT license)