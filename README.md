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

### Distribution for Dataset([Download](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?layout=list&ddrp=1&sort=name&num=50#))
* Training: 8,000 examples
* Test: 2,717 examples
* Total: 10,717 examples

| Relation           | Frequency     | Positive | IAA   |
|--------------------|---------------|----------|-------|
| Cause-Effect       | 1,331 (12.4%) | 91.2%    | 79.0% |
| Component-Whole    | 1,253 (11.7%) | 84.3%    | 70.0% |
| Entity-Destination | 1,137 (10.6%) | 80.1%    | 75.2% |
| Entity-Origin      | 974 (9.1%)    | 69.2%    | 58.2% |
| Product-Producer   | 948 (8.8%)    | 66.3%    | 84.8% |
| Member-Collection  | 923 (8.6%)    | 74.7%    | 68.2% |
| Message-Topic      | 895 (8.4%)    | 74.4%    | 72.4% |
| Content-Container  | 732 (6.8%)    | 59.3%    | 95.8% |
| Instrument-Agency  | 660 (6.2%)    | 60.8%    | 65.0% |
| Other              | 1,864 (17.4%) |
(*IAA = Inter-Annotator Agreement)


## Convolutional Neural Networks
<p align="center">
	<img width="600" height="400" src="https://user-images.githubusercontent.com/15166794/32838125-475cbdba-ca53-11e7-929c-2e27f1aca180.png">
</p>




