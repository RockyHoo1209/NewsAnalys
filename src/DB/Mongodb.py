from pymongo import MongoClient


def GetCollection(db_name:str,collect_name:str):
    host = 'localhost'   # 你的ip地址
    client = MongoClient(host, 27017)  # 建立客户端对象
    db = client[db_name]  # 连接mydb数据库，没有则自动创建
    return db[collect_name]
    