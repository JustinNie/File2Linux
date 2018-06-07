#Function Description  

##CountWordNet()  
* **Usage:** Count and analyze the basic information about the WordNet.
* **Input File:** None.
* **Output File:** *LemmaLengthAll.npy*, *SynsetNameAll.npy*.


##ConstructAllDict()


* **Usage:** Construct all synsets in WordNet into a dictionary.
* **Input File:** None.
* **Output File:** *SynsetDictAll.pkl*.
  

##ConstructNewDict()


* **Usage:** Construct the new dictionary, which stores the synset and lemma names according to the vocabulary. This is to make sure that all the lemma names could be found in vocabulary.
* **Input File:** *vocab.txt*.
* **Output File:** *SynsetDict.npy*, *LemmaLength.npy*, *SynsetNames.npy*, 
					  *SynsetNameNumber.pkl*, *SynsetNumberName.pkl*.


##ConstructData()


* **Usage:** Convert the word dictionary and word vectors into training data.
* **Input File:** *vectors.txt*, *SynsetDict.pkl*.
* **Output File:** *TrainVectors.npy*, *TrainSynsets.npy*.	


##ConstructNewDict()


* **Usage:** 
* **Input File:** 
* **Output File:** 


##ConstructNewDict()


* **Usage:** 
* **Input File:** 
* **Output File:** 



#Data Description  


##SynsetNameAll.npy


* **Definition:** a numpy that stores all the synset name (modified, add one more character in the last) information.
* **Format:** [item, ...], item -> car.n.010
* **Usage:** used to search synset name in the whole wordnet.
* **From:** function `CountWordNet()`.  


##LemmaLengthAll.npy  


* **Definition:** a numpy that store the number of lemmas of each synset. 
* **Format:** [3, 4, ...] -> means the synsets hava 3, 4, ... lemmas.
* **Usage:** used to count and analyze the length of lemmas of synsets.
* **From:** function `CountWordNet()`.


##SynsetDictAll.pkl


* **Definition:** a dictionary that stores the synset name and it's lemmas, which is divided into `NUMBER` lemmas in each synset.
* **Format:** 
	+ [synset name: lemmas]
	+ synset name: car.n.010 (there is one more character).
	+ lemmas: have NUMBER items. Repeat the item when the original is less than `NUMBER`.
* **Usage:** used to search the lemmas with index synset name.
* **From:** function `ConstructDict()`.


##LemmaLength.npy


* **Definition:** a numpy that store the number of lemmas of each synset according to the vocabulary.
* **Format:** [3, 4, ...] -> means the synsets hava 3, 4, ... lemmas.
* **Usage:** used to count and analyze the length of lemmas of synsets in the particular vocabulary.
* **From:** function `ConstructNewDict`.


##SynsetDict.pkl


* **Definition:** a dictionary that stores the synset name and it's lemmas, which is divided into `NUMBER` lemmas in each synset, according to the vocabulary.
* **Format:** 
	+ [synset name: lemmas].
	+ synset name: car.n.010 (there is one more character).
	+ lemmas: have NUMBER items. Repeat the item when the original is less than `NUMBER`.
* **Usage:** used to search the lemmas with index synset name in the vocabulary.
* **From:** function `ConstructNewDict()`.

##SynsetNameNumber.pkl


* **Definition:** a dictionary that search the particular number from the index synset name in the vocabulary.
* **Format:** 
	+ [name: number].
	+ name: the synset name, like `car.n.010`.
	+ number: the particular number, from the beginning 0, then increment it.
* **Usage:** create the particular number of each synset name, in order to store it in torch format.
* **From:** function `ConstructNewDict()`.


##SynsetNumberName.pkl


* **Definition:** a dictionary that search the synset name from the particular number in the vocabulary.
* **Format:** 
	+ [number: name].
	+ name: the synset name, like `car.n.010`.
	+ number: the particular number of synset name.
* **Usage:** index the synset name from the corresponding number.
* **From:** function `ConstructNewDict()`.


##TrainVectors.npy


* **Definition:** a numpy that stores the training vectors.
* **Format:** 
	+ (sample number, `NUMBER` * `IN_VECTOR`)
	+ **sample number**: the number of synset names used to train.
	+ `NUMBER`: the number of word in each row.
	+ `IN_VECTOR`: the length of vector of the word in the original.
* **Usage:** used to training, and it's corresponding to the *TrainSynsets.npy*.
* **From:** function `ConstructData()`.


##TrainSynsets.npy


* **Definition:** a numpy that stores the training synsets.
* **Format:** 
	+ [0, 1, ...], whose length is **sample number**. 
	+ the number is related to the corresponding synset name.
* **Usage:** used to index the synset name related to the **TrainVectors.npy**.
* **From:** function `ConstructData()`.


##SynsetDictAll.npy


* **Definition:** 
* **Format:** 
* **Usage:**
* **From:** 


##SynsetDictAll.npy


* **Definition:** 
* **Format:** 
* **Usage:**
* **From:** 




