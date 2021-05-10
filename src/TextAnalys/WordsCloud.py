""" 词云可视化展示 """
from pyecharts import options as opts
from pyecharts.charts.basic_charts.wordcloud import WordCloud
from src.Utils.CutWords import CutWords

def ShowWordCloud():
    def wordcloud_base(word_cloud_list) -> WordCloud:
        c = (
            WordCloud()
            .add("", word_cloud_list, word_size_range=[20, 100], shape='diamond')  # SymbolType.ROUND_RECT
            .set_global_opts(title_opts=opts.TitleOpts(title='WordCloud词云'))
        )
        return c
    word_cloud_list=[]
    word_idf=CutWords()
    for k,v in word_idf.items():
        word_cloud_list.append((k,v))
    wordcloud_base(word_cloud_list).render('./词云图.html')