from dataset_ import get_data
from cutstom_ner import train_spacy
from plot_loss import plot_loss
from spacy.gold import GoldParse
from evaluate_ner import doc_to_bilou, ner_report

file_path = 'data/data.json'
data = get_data(file_path)

train_data = data.get_data()

train_set, test_set = data.train_test_split(train_data, test_size= 0.1, random_state= 42)




nlp, loss_arr = train_spacy(train_set)

nlp.to_disk('model/ner_model')

plot_loss(loss_arr)

y_test = []
y_pred = []

for text, annots in test_set:
    
    gold = GoldParse(nlp.make_doc(text), entities = annots.get("entities"))
    ents = gold.ner
    pred_ents = doc_to_bilou(nlp, text)
    
    y_test.append(ents)
    y_pred.append(pred_ents)

report, accuracy = ner_report(y_test, y_pred)
print(report)
print(accuracy)