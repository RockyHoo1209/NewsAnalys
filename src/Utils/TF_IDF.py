""" TF-IDF算法分析 """
from re import template,sub
from pyspark import SparkConf,SparkContext
from pyspark.ml.feature import CountVectorizer, IDF,Tokenizer
from pyspark.sql.functions import json_tuple
from pyspark.sql.session import SparkSession
import jieba
from src.DB.Mongodb import GetCollection

def Cul_Freq():
    sentence_list=[]
    result=GetCollection("Gootworms","Result")
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
    return sentence_list

def TF_IDF():
    """ 使用 Tokenizer按空格分割句子，形成dataframe"""
    sentence_list=Cul_Freq()
    conf=SparkConf().setMaster("local").setAppName("NewsAnalys")
    sc=SparkContext(conf=conf)
    spark=SparkSession(sc)
    sentenceData=spark.createDataFrame(sentence_list).toDF("label","sentence").distinct()
    tokenizer=Tokenizer(inputCol="sentence",outputCol="words")
    wordsData=tokenizer.transform(sentenceData)
    """ idf模型训练 """
    countVector=CountVectorizer(inputCol="words",outputCol="rawFeatures",minDF=2)
    cvModel=countVector.fit(wordsData)
    cv_df=cvModel.transform(wordsData)
    idf=IDF(inputCol="rawFeatures",outputCol="features")
    idfModel=idf.fit(cv_df)
    rescaledData=idfModel.transform(cv_df)
    return rescaledData,cvModel