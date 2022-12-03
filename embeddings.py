import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd
class train_doc2vec():
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.skills = {}

    def get_skills(self, data):
        temp_skill = []
        #generate a 2-D list skills and name from the resume
        for j in data:
            if j['label'] == ['Skills']:
                temp_skill.append(j['points'][0]['text'])
            if j['label'] == ['Name']:
                temp_name = j['points'][0]['text']
            else:
                temp_name = "No name"

        #clean the list of skills
        for i,j in enumerate(temp_skill):
            j = j.replace("•","")
            j = j.replace('\n',",")
            j = re.sub("[^A-Za-z0-9+-, ]","",j)
            j = j.split(',')
            j = [x for x in j if x!= '']
            temp_skill[i] = j
            
        temp_s = []

        #Convert the 2-D list into a 1-D list
        for j in temp_skill:
            for i in j:
                temp_s.append(i)

        
        return (temp_name, temp_s)

    def skills_list(self):
        df = pd.read_json(self.file_path, lines = True)
        data = df["annotation"]
        #Create a dictionary of skills with key as name and value as the list of skills
        for i in data:
            name,skill = self.get_skills(i)
            self.skills[name] = skill

    # This is a function to train the doc2vec model using the skills dictionary
    def doc2vec_similarity_train(self):
        tokenized_dict = {}

        # Here we create a tokenized dictionary with key as name and values as tokens from the skills
        for n,s in self.skills.items():
            tokenized_list = []
            for i in s:
                x = word_tokenize(i.lower())
                for j in x:
                    tokenized_list.append(j)
            tokenized_dict[n] = tokenized_list
        
        # This tags each document (tokens of skills) to feed to the Doc2Vec model as input
        tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_dict.values())]

        # we create the doc2Vec model that outputs a vector of length 40. 
        # Window size = 3 for the continours bag of words
        # We don't count words with count less than 1 
        # and we train for 100 epochs
        self.model = Doc2Vec(vector_size=40,window = 3, min_count = 1, epochs = 100)
        
        #build and train the model
        self.model.build_vocab(tagged_data)
        self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        
    
    def get_model(self):
        self.skills_list()
        self.doc2vec_similarity_train()

        return self.model
