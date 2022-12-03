import spacy 
import sys
import fitz
import ast 
import re
from similarity_test import doc2vec_similarity_test
from embeddings import train_doc2vec
nlp = spacy.load('model/ner_model')


fname = 'KartikUllal_resume_nlp.pdf'
resume = fitz.open(fname)



text = ''
for page in resume:
    text = text + str(page.get_text())
tx = " ".join(text.split('\n'))


doc = nlp(tx)


user_skills = []
for ent in doc.ents:
    if ent.label_ == 'SKILL':
        user_skills.append(ent.text)


print(user_skills)

def clean_job(description):
  description = description.lower()
  description= description.replace("•"," ")
  description = description.replace('\n',",")
  description = re.sub("[^A-Za-z0-9,]"," ",description)

  return description

#job description for data science role
job_req = "To qualify you must have a 1. Masters degree in a quantitative discipline (Biomedical Informatics, Computer Science, Machine Learning, Applied Statistics, Mathematics or similar field, Proficiency in at least one programming language (Python, R) and machine learning tools (scikit learn, R), Knowledge of predictive modeling and machine learning concepts, including design, development, evaluation, deployment and scaling to large datasets, Familiarity with computing models for big data Hadoop / MapReduce, Spark etc., Knowledge of databases (Relational / SQL, NOSQL, MongoDB, etc.), Good grasp of software engineering principles. Experience in integrating modern software architectures, Knowledge and some experience in operational aspects of software development and deployment, including automation, testing, virtualization and container technology, Knowledge of clinical and operational aspects of healthcare delivery, Excellent written and oral communication skills for a variety of audiences, Preferred Qualifications, PhD degree in a quantitative field (Biomedical Informatics, Computer Science, Machine Learning, Applied Statistics, Mathematics or similar field) + 2 years experience, Demonstrated skills in design and implementation of complex machine learning models, Demonstrated knowledge of software engineering and operational skills through prior projects."

job_req = clean_job(job_req)

doc = nlp(job_req)


job_skills = []
for ent in doc.ents:
    if ent.label_ == 'SKILL':
        job_skills.append(ent.text)


print(job_skills)
file_path = 'data/data.json'
trainer = train_doc2vec(file_path)
model = trainer.get_model()


print(doc2vec_similarity_test(model, job_skills,user_skills))