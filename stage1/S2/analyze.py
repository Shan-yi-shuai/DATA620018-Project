import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import seaborn as sns
import re

# 数据加载
df1 = pd.read_csv('/opt/tiger/yantu-swift/VAST/VAST/csv-1700-1830.csv', parse_dates=['date(yyyyMMddHHmmss)'], encoding='ISO-8859-1')
df2 = pd.read_csv('/opt/tiger/yantu-swift/VAST/VAST/csv-1831-2000.csv', parse_dates=['date(yyyyMMddHHmmss)'], encoding='ISO-8859-1')
df3 = pd.read_csv('/opt/tiger/yantu-swift/VAST/VAST/csv-2001-2131.csv', parse_dates=['date(yyyyMMddHHmmss)'], encoding='ISO-8859-1')
df = pd.concat([df1, df2, df3])

def sudden_change():
    # 检测消息突发量
    msg_count = df.resample('10T', on='date(yyyyMMddHHmmss)').size()
    std_dev = msg_count.std()
    alerts = msg_count[msg_count > 3*std_dev]  # 三倍标准差作为异常阈值
    print("alerts: ", alerts) ## 输出异常时间段及其消息量


def plot_timeseries(df):
    # 按10分钟间隔统计消息量
    mb_counts = df[df['type'] == 'mbdata'].resample('10T', on='date(yyyyMMddHHmmss)').size()
    cc_counts = df[df['type'] == 'ccdata'].resample('10T', on='date(yyyyMMddHHmmss)').size()

    plt.figure(figsize=(15,6))
    mb_counts.plot(label='Social Media Posts', color='royalblue')
    cc_counts.plot(label='Emergency Calls', color='crimson', linestyle='--')
    
    # 添加消息突发的关键时间点标记（来源：sudden_change（））
    for time_str in ['2014-01-23 18:40:00', '2014-01-23 19:40:00']:
        timestamp = pd.to_datetime(time_str)
        plt.scatter(timestamp, mb_counts.get(timestamp, 0), 
                   color='red', marker='*', s=150, zorder=5)
        plt.scatter(timestamp, cc_counts.get(timestamp, 0),
                   color='red', marker='*', s=150, zorder=5)

    # 设置横轴刻度
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MinuteLocator(interval=10))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    
    plt.title('Event Timeline Analysis')
    plt.ylabel('Message Volume')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()  # 防止标签重叠
    plt.savefig('/opt/tiger/yantu-swift/path/to/images/事件时间序列变化图.png')
    plt.show()


# 关键词词云与情感分析
def text_analysis(df):
    # 生成词云
    text = ' '.join(df[df['type']=='mbdata']['message'].dropna())
    wordcloud = WordCloud(width=800, height=400).generate(text)
    
    # 情感分析：使用TextBlob计算每条消息的情感极性（-1到1）
    def get_sentiment(text):
        return TextBlob(str(text)).sentiment.polarity
    df['sentiment'] = df['message'].apply(get_sentiment)
    
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    
    plt.subplot(1,2,2)
    sns.histplot(df[df['type']=='mbdata']['sentiment'], bins=10, kde=True)
    plt.title('Sentiment Distribution')
    plt.show()
    plt.savefig('/opt/tiger/yantu-swift/path/to/images/关键词词云与情感分析.png')


def analyze_ccdata_risk(df):
    # 筛选ccdata数据
    cc_df = df[df['type'] == 'ccdata'].copy()
    with open('/opt/tiger/yantu-swift/path/to/ccdata_messages.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(cc_df['message'].dropna().astype(str)))

    # 定义风险等级判断规则
    def classify_risk(message):
        message = message.lower()
        # 高风险：涉及暴力犯罪或重大灾害
        high_risk = r'\b(assault|shooting|hostage|riot|fire|bomb|explosion|armed|stabbing|active shooter|homicide|injured officer|building fire|officer down)\b'
        # 中风险：财产犯罪或潜在危险
        med_risk = r'\b(disturbance|alarm|crime scene|investigation|vandalism|theft|suspicious|break-in|burglary|dwelling of interest|subject stop|shots fired)\b'
        # 低风险：其他
        # low_risk = r'\b(traffic stop|park check|keep the peace|parking violation|routine patrol|business check|secure no crime|suspicious circumstances)\b'
        
        if re.search(high_risk, message):
            return 'high risk'
        elif re.search(med_risk, message):
            return 'medium risk'
        # 剩余未匹配的归为低风险（如crowd control等）
        return 'low risk' 
    
    # 应用分类
    cc_df['risk_level'] = cc_df['message'].apply(classify_risk)
    
    # 统计各风险等级数量
    risk_counts = cc_df['risk_level'].value_counts()
    print(risk_counts)
    # 可视化
    plt.figure(figsize=(10,6))
    risk_counts.plot(kind='bar', color=['#ff9999','#66b3ff','#99ff99'])
    plt.title('Emergency Call Risk Level Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.savefig('/opt/tiger/yantu-swift/path/to/images/风险等级分布.png')
    plt.show()



def user_behavior_analysis():
    # 用户行为分析
    top_users = df['author'].value_counts().head(10)
    plt.figure(figsize=(10,6))
    top_users.plot(kind='barh')
    plt.title('TOP10 active user')
    plt.savefig('/opt/tiger/yantu-swift/path/to/images/用户行为分析.png')

def repost():
    # 构建转发网络
    retweets = df[df['message'].str.startswith('RT @')]
    # 创建包含发送者和接收者的DataFrame
    network_df = pd.DataFrame({
        'sender': retweets['author'],
        'receiver': retweets['message'].str.extract(r'RT @(\w+)')[0]}).dropna()

    # 新增：统计转发频次
    freq_network = network_df.groupby(['sender', 'receiver']).size().reset_index(name='counts')
    strong_links = freq_network[freq_network['counts'] >= 4]  # 筛选互转≥2次的关系

    # 绘制高频子图
    import networkx as nx
    plt.figure(figsize=(12,8))
    subgraph = nx.from_pandas_edgelist(strong_links, 'sender', 'receiver', edge_attr='counts')
    
    # 设置可视化参数
    edge_width = [d['counts']*0.8 for (u,v,d) in subgraph.edges(data=True)]
    nx.draw(subgraph,
           node_size=70,
           width=edge_width,
           edge_color='firebrick',
           with_labels=True,
           font_size=8)
    plt.title('High-frequency Retweet Relationships')
    plt.savefig('/opt/tiger/yantu-swift/path/to/images/高频转发子图.png')
    plt.close()

    # 原始完整网络图绘制
    G = nx.from_pandas_edgelist(network_df, 
                            source='sender', 
                            target='receiver')
    plt.figure(figsize=(12,8))
    nx.draw(G, node_size=50, alpha=0.5, with_labels=True)
    plt.savefig('/opt/tiger/yantu-swift/path/to/images/转发网络图.png')
    plt.close()


def valid_event():
    # 基于权威账号的规则
    authorities = ['AbilaPost', 'POK', 'KronosStar']
    # 有效事件特征
    valid_event = df[
        (df['type'] == 'ccdata') |  # 报警记录
        (df['author'].isin(authorities)) |  # 权威账号
        (df['message'].str.contains('#APD|#standoff|#riot|#shooting|#evacuate')) |  # 关键标签
        (df['latitude'].notna() & df['longitude'].notna())  # 含地理坐标
    ]
    print("valid_event: ", valid_event)
    # 新增保存有效事件到CSV
    valid_event.to_csv('/opt/tiger/yantu-swift/VAST/VAST/valid_events.csv', 
                      index=False, encoding='utf-8')

def advertisement():
    # 广告特征
    spam_pattern = r'(http|\.kronos/|click here|#followme)'
    spam = df[df['message'].str.contains(spam_pattern, na=False)]
    print("spam: ", spam)
    # 新增保存广告数据到CSV
    spam.to_csv('/opt/tiger/yantu-swift/VAST/VAST/advertisements.csv', 
               index=False, encoding='utf-8')


if __name__ == "__main__":
    # sudden_change()
    # plot_timeseries(df)
    # text_analysis(df)
    # analyze_ccdata_risk(df)
    # user_behavior_analysis()
    repost()
    # valid_event()
    # advertisement()


