

capgemini example app?

Nicola Korherr, Mutius Lab & Daniel Weiß, Klein Lab

CWAS study >2000 real patients + 500 control group -> there is a control group!
- all have a curated ICD10 code
87 kids validation set

each kid is a node, each kid is connected to the genes

Prof. Matthias Mann

only for 30%, the reason for illness is known

we do not get genetic data from kids

dummy dataset with identical graph attributes

push to Baumbach's featurecloud software

Daniel reads + sends output log
Tell Daniel which metrics to implement!

20% business model
- maybe AI diagnostics app to reduce appointment number?
- make hospitals pay for multi-omics analysis, 3-5K per patiens

scRNA, transcriptomics, blood+urine proteins, protein interactions

= clinical data
- blood + urine
- 0-30 terms

= unstructured data
- 900 questions questionnaire

= genomic data
- 10-20 genes

= proteomic data
- for 550 kids
- from urine
- quantitative for 1000-2000 proteins



= graph
- has_phenotype: big feet not bad
- has_damage: genetic?
- has_protein
- has_disease: 2/3 have a disease

= start hacking
- graph-guided RF fc_grandforest gitlab

- entry-level target: differentiate between sick and healthy
- then: before "S52.521A" 
- F1 for healthy/disease, accuracy for the "S"

No GNNs in this hackathon?

= areas

| field          | who         | notes            |
|----------------|-------------|------------------|
| Neo4j + cipher | Benn        |                  |
| featurecloud   | Erta        |                  |
| ML             | Rohan + Leo | output CSV files |


| infra       | who   |
|-------------|-------|
| github repo | Leo   |
| discord     | Rohan |















