from nltk.tokenize import word_tokenize
from similarity import cosine

def doc2vec_similarity_test(model, req, skill):
    tokenized_req = []
    tokenized_skill = []
    #tokenizes the job requirements
    for i in req:
        x = word_tokenize(i.lower())
        for j in x:
                tokenized_req.append(j)
    #tokenizes the resume skills
    for i in skill:
        x = word_tokenize(i.lower())
        for j in x:
                tokenized_skill.append(j)
    
    #calculates vector for resume skills
    skill_vector = model.infer_vector(tokenized_skill)
    #calculates vector for job requirement skills
    req_vector = model.infer_vector(tokenized_req)
    
    #calculates similarity
    similarity = cosine(skill_vector, req_vector)
    
    return similarity
    