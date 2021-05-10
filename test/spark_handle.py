#%%
from collections import defaultdict
from re import template,sub
from pyecharts.charts.basic_charts.wordcloud import WordCloud
from pyspark import SparkConf,SparkContext
from pyspark.ml.feature import CountVectorizer, IDF,HashingTF,Tokenizer
from pymongo import MongoClient
from pyspark.sql.functions import json_tuple
from pyspark.sql.session import SparkSession
import numpy as np
import jieba
import json
from pyecharts import options as opts
from tqdm import tqdm
np.set_printoptions(threshold=np.inf)
#%%
""" mongodb连接 """
host = 'localhost'   # 你的ip地址
client = MongoClient(host, 27017)  # 建立客户端对象
db = client["Gootworms"]  # 连接mydb数据库，没有则自动创建
#%%
sentence_list=[]
result=db["Result"]
counter=0
for i in result.find():
    if "content" in i:
            if i["content"] is None:continue
            content=str(list(jieba.cut(i["content"].replace(" ","")))).\
                replace(","," ").replace("'","")
            # print(content)
            # 去除特殊字符
            temp_tupple=(counter,sub(r"[a-zA-Z]+","",content[1:-2]))
            sentence_list.append(temp_tupple)
            counter+=1
#%%            
""" 初始化spark并创建dataframe """
conf=SparkConf().setMaster("local").setAppName("NewsAnalys")
sc=SparkContext(conf=conf)
spark=SparkSession(sc)
sentenceData=spark.createDataFrame(sentence_list).toDF("label","sentence").distinct()
# %%
tokenizer=Tokenizer(inputCol="sentence",outputCol="words")
wordsData=tokenizer.transform(sentenceData)
#%%
wordsData.show(1)
#%%
""" TF哈希结果（无法将词频对应上单词)"""
hashingTF=HashingTF(inputCol="words",outputCol="rawFeatures")
featurizeData=hashingTF.transform(wordsData)
featurizeData.select("words","rawFeatures").show(truncate=False)
#%%
""" CountVectorizer词频统计(可以将词频对应上单词)"""
countVector=CountVectorizer(inputCol="words",outputCol="rawFeatures",minDF=2)
cvModel=countVector.fit(wordsData)
cv_df=cvModel.transform(wordsData)
cv_df.show(4,False)
#%%
# voc=cvModel.vocabulary
# getKeywordFunc=udf()
# %%
""" IDF模型训练 """
idf=IDF(inputCol="rawFeatures",outputCol="features")
idfModel=idf.fit(cv_df)
rescaledData=idfModel.transform(cv_df)
# %%
list=rescaledData.collect()
# with open("./collect_file.txt","w+") as f:
#     f.write(str(list))
# %%
Features=rescaledData.select("features").toPandas()
Words=rescaledData.select("words").toPandas()
#%%
features_dict=Features.to_dict()
# with open("./features_dict.txt","w") as f:
#     f.write(str(features_dict["features"]))
# %%
features_numpy=np.array(Features)
#%%
word_idf=defaultdict(float)
vacabulary=cvModel.vocabulary
features_dict=Features.to_dict()
features=features_dict["features"]
docs=features.keys()
#%%
for doc_num in tqdm(docs):
    temp_features=features[int(doc_num)]
    values=temp_features.values
    indices=temp_features.indices
    for i in indices:
        word_idf[vacabulary[int(i)]]=temp_features[int(i)]
jsObj = json.dumps(word_idf)
with open("./word_idf.json","w+") as f:
    f.write(jsObj)
# with open("./word_idf.json","r") as f:
#     word_idf=json.loads(f.read())

#%%
word_cloud_list=[]
for k,v in word_idf.items():
    word_cloud_list.append((k,v))
with open("./word_cloud_list.txt","w") as f:
    f.write(str(word_cloud_list))

#%%
#%%
with open("./features_numpy.txt","w+") as f:
    f.write(str(features_numpy[0][0]))
#%%
""" 计算两两文本间的idf-tf的余弦相似度 """
res_list=[]
for i in range(496):
    for j in range(i+1,496):
        vec1=features_numpy[i][0].toArray()
        vec2=features_numpy[j][0].toArray()
        num=vec1.dot(vec2.T)
        denom=np.linalg.norm(vec1)*np.linalg.norm(vec2)
        res_list.append(((i,j),num/denom))
# %%
# with open("cosineSim.txt","w+") as f:
#     f.write(str(res_list))
# %%
# 渲染图
def wordcloud_base() -> WordCloud:
    c = (
        WordCloud()
        .add("", word_cloud_list, word_size_range=[20, 100], shape='diamond')  # SymbolType.ROUND_RECT
        .set_global_opts(title_opts=opts.TitleOpts(title='WordCloud词云'))
    )
    return c

# 生成图
wordcloud_base().render('./词云图.html')

# %%
