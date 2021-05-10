""" 分词模块 """
from collections import defaultdict

from tqdm import tqdm
from src.Utils.TF_IDF import TF_IDF

def CutWords(): 
   rescaledData,cvModel=TF_IDF()
   Features=rescaledData.select("features").toPandas() 
   word_idf=defaultdict(float)
   vacabulary=cvModel.vocabulary
   features_dict=Features.to_dict()
   features=features_dict["features"]
   docs=features.keys()
   for doc_num in tqdm(docs):
       temp_features=features[int(doc_num)]
       values=temp_features.values
       indices=temp_features.indices
       for i in indices:
           word_idf[vacabulary[int(i)]]=temp_features[int(i)]
   return word_idf