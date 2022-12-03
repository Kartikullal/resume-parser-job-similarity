
import spacy
import ast
import random

f = open("data/skill.txt", "r")
l = f.read()
patterns = l
res = ast.literal_eval(patterns)

def train_spacy(train_data):
    
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy

    # adding 'parser', 'ner','tagger' and 'entity_ruler' pipeline components
    if 'ner' not in nlp.pipe_names and 'parser' not in nlp.pipe_names and 'tagger' not in nlp.pipe_names:

        parser = nlp.create_pipe('parser')
        nlp.add_pipe(parser, last = True)

        tagger = nlp.create_pipe('tagger')
        nlp.add_pipe(tagger, last = True)

        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

        ruler = nlp.create_pipe('entity_ruler')
        nlp.add_pipe(ruler, last = True)
        ruler.add_patterns(res)
        
        
    # add labels
    for _, annotations in train_data:
         for ent in annotations.get("entities"):
            ner.add_label(ent[2])
            
    # get names of other pipes to disable them during training
    loss_arr = []
    #other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    #with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(20):
        print("Starting iteration " + str(itn))
        random.shuffle(train_data)
        losses = {}
        for text, annotations in train_data:
            nlp.update(
                [text],  # batch of texts
                [annotations],  # batch of annotations
                drop=0.2,  # dropout - make it harder to memorise data
                sgd=optimizer,  # callable to update weights
                losses=losses)
        loss_arr.append(losses["ner"])
        print(losses)
    return nlp,loss_arr