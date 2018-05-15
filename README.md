# Convolutional Neural Networks for Relation Extraction

Tensorflow Implementation of Deep Learning Approach for Relation Extraction Challenge([**SemEval-2010 Task #8**: *Multi-Way Classification of Semantic Relations Between Pairs of Nominals*](https://docs.google.com/document/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/preview)) via Convolutional Neural Networks.

<p align="center">
	<img width="700" height="400" src="https://user-images.githubusercontent.com/15166794/32838125-475cbdba-ca53-11e7-929c-2e27f1aca180.png">
</p>


## Usage
### Train
* train data is located in "*<U>SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT*</U>".
* "[GoogleNews-vectors-negative300](https://code.google.com/archive/p/word2vec/)" is used as pre-trained word2vec model.
* Display help message:
	```bash
	$ python train.py --help
	```

	```bash
	optional arguments:
		-h, --help            show this help message and exit
		--train_dir TRAIN_DIR
								Path of train data
		--dev_sample_percentage DEV_SAMPLE_PERCENTAGE
								Percentage of the training data to use for validation
		--max_sentence_length MAX_SENTENCE_LENGTH
								Max sentence length in train(98)/test(70) data
								(Default: 100)
		--word2vec WORD2VEC   Word2vec file with pre-trained embeddings
		--text_embedding_dim TEXT_EMBEDDING_DIM
								Dimensionality of word embedding (Default: 300)
		--position_embedding_dim POSITION_EMBEDDING_DIM
								Dimensionality of position embedding (Default: 100)
		--filter_sizes FILTER_SIZES
								Comma-separated filter sizes (Default: 2,3,4,5)
		--num_filters NUM_FILTERS
								Number of filters per filter size (Default: 128)
		--dropout_keep_prob DROPOUT_KEEP_PROB
								Dropout keep probability (Default: 0.5)
		--l2_reg_lambda L2_REG_LAMBDA
								L2 regularization lambda (Default: 3.0)
		--batch_size BATCH_SIZE
								Batch Size (Default: 64)
		--num_epochs NUM_EPOCHS
								Number of training epochs (Default: 100)
		--display_every DISPLAY_EVERY
								Number of iterations to display training info.
		--evaluate_every EVALUATE_EVERY
								Evaluate model on dev set after this many steps
		--checkpoint_every CHECKPOINT_EVERY
								Save model after this many steps
		--num_checkpoints NUM_CHECKPOINTS
								Number of checkpoints to store
		--learning_rate LEARNING_RATE
								Which learning rate to start with. (Default: 1e-3)
		--allow_soft_placement [ALLOW_SOFT_PLACEMENT]
								Allow device soft device placement
		--noallow_soft_placement
		--log_device_placement [LOG_DEVICE_PLACEMENT]
								Log placement of ops on devices
		--nolog_device_placement
	```

* **Train Example:**
    ```bash
	$ python train.py --word2vec "GoogleNews-vectors-negative300.bin"
	```

### Evalutation
* test data is located in "<U>*SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT*</U>".
* You must give "**checkpoint_dir**" argument, path of checkpoint(trained neural model) file, like below example.

* **Evaluation Example:**
	```bash
	$ python eval.py --checkpoint_dir "runs/1523902663/checkpoints"
	```

* **Official Evaluation of SemEval 2010 Task #8**
	1. After evaluation like the example, you can get the "*prediction.txt*" and "*answer.txt*" in "*result*" directory.
	2. Install <U>[perl](https://www.perl.org/get.html)</U>.
	3. Move to <U>*SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2*</U>.
        ```bash
        $ cd SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2
		```
	4. Check your prediction file format.
		```bash
		$ perl semeval2010_task8_format_checker.pl ../../result/prediction.txt
		```
	5. Scoring your prediction.
		```bash
		$ perl semeval2010_task8_scorer-v1.2.pl ../../result/prediction.txt ../../result/answer.txt
		```
	6. The scorer shows the 3 evaluation reuslts for prediction. The official evaluation result, **(9+1)-WAY EVALUATION TAKING DIRECTIONALITY INTO ACCOUNT -- OFFICIAL**, is the last one. See the [README](SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/README.txt) for more details.



## SemEval-2010 Task #8
* Given: a pair of *nominals*
* Goal: recognize the semantic relation between these nominals.
* Example:
	* "There were apples, **<U>pears</U>** and oranges in the **<U>bowl</U>**." 
		<br> → *CONTENT-CONTAINER(pears, bowl)*
	* “The cup contained **<U>tea</U>** from dried **<U>ginseng</U>**.” 
		<br> → *ENTITY-ORIGIN(tea, ginseng)*


### The Inventory of Semantic Relations
1. *Cause-Effect(CE)*: An event or object leads to an effect(those cancers were caused by radiation exposures)
2. *Instrument-Agency(IA)*: An agent uses an instrument(phone operator)
3. *Product-Producer(PP)*: A producer causes a product to exist (a factory manufactures suits)
4. *Content-Container(CC)*: An object is physically stored in a delineated area of space (a bottle full of honey was weighed) Hendrickx, Kim, Kozareva, Nakov, O S´ eaghdha, Pad ´ o,´ Pennacchiotti, Romano, Szpakowicz Task Overview Data Creation Competition Results and Discussion The Inventory of Semantic Relations (III)
5. *Entity-Origin(EO)*: An entity is coming or is derived from an origin, e.g., position or material (letters from foreign countries)
6. *Entity-Destination(ED)*: An entity is moving towards a destination (the boy went to bed) 
7. *Component-Whole(CW)*: An object is a component of a larger whole (my apartment has a large kitchen)
8. *Member-Collection(MC)*: A member forms a nonfunctional part of a collection (there are many trees in the forest)
9. *Message-Topic(CT)*: An act of communication, written or spoken, is about a topic (the lecture was about semantics)
10. *OTHER*: If none of the above nine relations appears to be suitable.


### Distribution for Dataset
* **SemEval-2010 Task #8 Dataset [[Download](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?layout=list&ddrp=1&sort=name&num=50#)]**

	| Relation           | Train Data          | Test Data           | Total Data           |
	|--------------------|:-------------------:|:-------------------:|:--------------------:|
	| Cause-Effect       | 1,003 (12.54%)      | 328 (12.07%)        | 1331 (12.42%)        |
	| Instrument-Agency  | 504 (6.30%)         | 156 (5.74%)         | 660 (6.16%)          |
	| Product-Producer   | 717 (8.96%)         | 231 (8.50%)         | 948 (8.85%)          |
	| Content-Container  | 540 (6.75%)         | 192 (7.07%)         | 732 (6.83%)          |
	| Entity-Origin      | 716 (8.95%)         | 258 (9.50%)         | 974 (9.09%)          |
	| Entity-Destination | 845 (10.56%)        | 292 (10.75%)        | 1137 (10.61%)        |
	| Component-Whole    | 941 (11.76%)        | 312 (11.48%)        | 1253 (11.69%)        |
	| Member-Collection  | 690 (8.63%)         | 233 (8.58%)         | 923 (8.61%)          |
	| Message-Topic      | 634 (7.92%)         | 261 (9.61%)         | 895 (8.35%)          |
	| Other              | 1,410 (17.63%)      | 454 (16.71%)        | 1864 (17.39%)        |
	| **Total**          | **8,000 (100.00%)** | **2,717 (100.00%)** | **10,717 (100.00%)** |



## Reference
* **Relation Classification via Convolutional Deep Neural Network** (COLING 2014), D Zeng et al. **[[review]](https://github.com/roomylee/paper-review/blob/master/relation_extraction/Relation_Classification_via_Convolutional_Deep_Neural_Network.md)** [[paper]](http://www.aclweb.org/anthology/C14-1220)
* **Relation Extraction: Perspective from Convolutional Neural Networks** (NAACL 2015), TH Nguyen et al. **[[review]](https://github.com/roomylee/paper-review/blob/master/relation_extraction/Relation_Extraction-Perspective_from_Convolutional_Neural_Networks.md)** [[paper]](http://www.cs.nyu.edu/~thien/pubs/vector15.pdf)
* dennybritz's cnn-text-classification-tf repository [[github]](https://github.com/dennybritz/cnn-text-classification-tf)


