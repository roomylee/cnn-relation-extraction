# Relation Extraction using Convolutional Neural Networks

Deep Learning Approach for Relation Extraction Challenge([**SemEval-2010 Task #8**: *Multi-Way Classification of Semantic Relations Between Pairs of Nominals*](https://docs.google.com/document/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/preview)) using Convolutional Neural Networks.


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



## Convolutional Neural Networks
<p align="center">
	<img width="600" height="400" src="https://user-images.githubusercontent.com/15166794/32838125-475cbdba-ca53-11e7-929c-2e27f1aca180.png">
</p>


## Reference
* **Relation Extraction: Perspective from Convolutional Neural Networks** (NAACL 2015), TH Nguyen et al. [[pdf]](http://www.cs.nyu.edu/~thien/pubs/vector15.pdf)


