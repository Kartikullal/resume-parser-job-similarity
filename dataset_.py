
# import logging
import json
import re
import random 
import math
# JSON formatting functions


class get_data():
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.training_data = []
        self.tidy_data = []
        self.clean_data = []


    def json_to_spacy(self):
        lines=[]

        
        with open(self.file_path, 'r') as f:
            lines = f.readlines()

        #For each json line
        for line in lines:
            data = json.loads(line)

            #add content key in data dictionary
            text = data['content'].replace("\n", " ")
            entities = []

            #add annotation key in dictionary
            data_annotations = data['annotation']
            if data_annotations is not None:
                for annotation in data_annotations:
                    #only a single point in text annotation.
                    point = annotation['points'][0]
                    labels = annotation['label']
                    # handle both list of labels or a single label.
                    if not isinstance(labels, list):
                        labels = [labels]

                    for label in labels:
                        p_start = point['start']
                        p_end = point['end']
                        p_text = point['text']

                        #find the left and right white spaces and remove them
                        lstrip_diff = len(p_text) - len(p_text.lstrip())
                        rstrip_diff = len(p_text) - len(p_text.rstrip())

                        #move the pointer for white spaces
                        if lstrip_diff != 0:
                            p_start = p_start + lstrip_diff
                        if rstrip_diff != 0:
                            p_end = p_end - rstrip_diff

                        #add the updates locations of the entities
                        entities.append((p_start, p_end + 1 , label))
            self.training_data.append((text, {"entities" : entities}))

    def trim_entity_spans(self) -> list:
        #Removes leading and trailing white spaces from entity spans.
        #Returns The cleaned data.

        inval_span_tokens = re.compile(r'\s')

        for text, annotations in self.training_data:
            entities = annotations['entities']
            val_entities = []
            for start, end, label in entities:
                val_start = start
                val_end = end

                # remove the whitespaces in the entity spans
                while val_start < len(text) and inval_span_tokens.match(
                        text[val_start]):
                    val_start += 1
                while val_end > 1 and inval_span_tokens.match(
                        text[val_end - 1]):
                    val_end -= 1
                val_entities.append([val_start, val_end, label])
            self.tidy_data.append([text, {'entities': val_entities}])
     
    def clean_ents(self):
    
        for text, annotation in self.tidy_data:
            
            ents = annotation.get('entities')
            ents_copy = ents.copy()
            
            # append ent only if it is longer than its overlapping ent
            i = 0
            for ent in ents_copy:
                j = 0
                for overlapping_ent in ents_copy:
                    # Skip self
                    if i != j:
                        e_start, e_end, oe_start, oe_end = ent[0], ent[1], overlapping_ent[0], overlapping_ent[1]
                        # Delete any ent that overlaps, keep if longer
                        if ((e_start >= oe_start and e_start <= oe_end) \
                        or (e_end <= oe_end and e_end >= oe_start)) \
                        and ((e_end - e_start) <= (oe_end - oe_start)):
                            ents.remove(ent)
                    j += 1
                i += 1
            self.clean_data.append((text, {'entities': ents}))
                    
        


    def get_data(self):

        self.json_to_spacy()
        self.trim_entity_spans()
        self.clean_ents()

        return self.clean_data

    def train_test_split(self, data, test_size, random_state):

        random.Random(random_state).shuffle(data)
        test_idx = len(data) - math.floor(test_size * len(data))
        train_set = data[0: test_idx]
        test_set = data[test_idx: ]

        return train_set, test_set