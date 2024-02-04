import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from playsound import playsound

f = open('true_out.csv', 'r')
data = f.read().split(", ")
data = list(map(int,data))
f.close()
T = 60#1分あたりの衝突回数
f = 0.05#1サンプル取得の時間
h = 20#サンプル周波数
update_time = 30#更新する時間単位
e = 0
mean_100 = 0
near = 100#最近の平均

hist_range = int(T/f)#ヒストグラムにする幅
start = 0
end = hist_range

update_range = int(update_time/f)
hist = []
rank = []
for i in range(near):
    rank.append(10)
    
def main():
    #plt.style.use('dark_background')
    fig = plt.figure(figsize = (20,4), facecolor='white')
    ani = FuncAnimation(fig, plot_hist, interval=250)
    plt.show()
    #plt.plot(hist)
    
    #上半分
    plt.subplot(211)
    value = [0,0,0,0]
    for i in range(len(hist)-2):
        if rank[i+1] == 0:
            col = 'red'
            value[0] = value[0] + 1
        elif rank[i+1] == 1:
            col = 'orange'
            value[1] = value[1] + 1
        elif rank[i+1] == 2:
            col = 'green'
            value[2] = value[2] + 1
        elif rank[i+1] == 3:
            col = 'blue'
            value[3] = value[3] + 1
        else:
            col = 'black'
        plt.plot([i,i+1],hist[i:i+2], color=col,label=col)
    plt.title("作業効率の変化")
    plt.tick_params(labelbottom=False)
    plt.xlabel("時間")
    plt.ylabel("作業効率")
    
    #左下
    plt.subplot(223)
    #plt.title("評価割合")
    plt.axis("off")
    #plt.text(0, 0.8,"　　　Excellent",size=30,color='r')
    labels = ["Excellent","Good","Poor","Bad"]
    colors = ["red","orange","green","blue"]
    plt.pie(value, startangle=90, counterclock=False, autopct='%.1f%%', labels=labels, colors=colors)
    print(value)
    #plt.legend()
    
    #右下
    plt.subplot(224)
    plt.text(0, 0.90,"コメント")
    plt.text(0, 0.75,"テスト採点お疲れ様でした。")
    good = value[0] + value[1]
    bad = value[2] + value[3]
    if good > bad:
        plt.text(0, 0.6,str(round(good/(good+bad)*100,1))+"%の時間、高評価を得ました。")
        plt.text(0, 0.45,"よく集中できています！")
    else:
        plt.text(0, 0.6,str(round(good/(good+bad)*100,1))+"%の時間、高評価を得ました。")
        plt.text(0, 0.45,"低評価の方が多くなっています。がんばりましょう！")
    
    
    #plt.text(0, 0.3,"テスト採点お疲れ様でした。")
    plt.axis("off")
    plt.show()



def plot_hist(i):
    if i != 0:
        plt.clf() 
    #左半分
    plt.subplot(121)
    plt.axis("off")
    #メーター
    radius = 1
    #円
    x = radius * np.cos(np.linspace(0, np.pi, 1000))
    y = radius * np.sin(np.linspace(0, np.pi, 1000))
    plt.plot(x, y, color = "k")
    
    for second in range(61):
        line_length = 0.1 if second % 10 == 0 else 0.05
        line_width = 2 if second % 10 == 0 else 1
        x1 = np.cos(np.radians(180 * (second / 60))) * radius
        x2 = np.cos(np.radians(180 * (second / 60))) * (radius - line_length)
        y1 = np.sin(np.radians(180 * (second / 60))) * radius
        y2 = np.sin(np.radians(180 * (second / 60))) * (radius - line_length)
        plt.plot([x1, x2], [y1, y2], lw = line_width, color = "k")
    
    for hour in range(0, 70, 10):
        x = np.cos(np.radians(180 * (hour / 60))) * radius * 0.8
        y = np.sin(np.radians(180 * (hour / 60))) * radius * 0.8
        plt.text(x,y,str(60-hour),fontsize=16,verticalalignment="center",horizontalalignment="center")
    global start,end,e,mean_100
    if i == 0:
        e = sum(data[i:i+hist_range])#data[int(i*0.25/f):
    else:
        plus_begin = int((i-1)*0.25/f)+hist_range
        plus = sum(data[plus_begin:plus_begin+5])
        plus = sum(data[end:end+5])
        minus = sum(data[start:start+5])
        end = end + 5
        start = start + 5
        e = e - minus + plus 
    hist.append(e)
    print(e)
    mean=np.mean(hist)
    x = np.cos(np.radians(180 * ((60-e) / 60))) * radius
    y = np.sin(np.radians(180 * ((60-e) / 60))) * radius
    plt.plot([0,x],[0,y],color = "r",label="現在の作業効率")
    
    x = np.cos(np.radians(180 * ((60-mean) / 60))) * radius
    y = np.sin(np.radians(180 * ((60-mean) / 60))) * radius
    plt.plot([0,x],[0,y],color = "g",label="平均")
    plt.legend(bbox_to_anchor=(0.5, 0), loc='center', borderaxespad=0, fontsize=10)
    plt.title('作業効率メーター',{'fontsize':15})     

    #if len(hist)>near:
    #    new_hist=hist[(len(hist)-near):]
    #    mean_100=np.mean(new_hist)
    #    x = np.cos(np.radians(180 * ((60-mean_100) / 60))) * radius
    #    y = np.sin(np.radians(180 * ((60-mean_100) / 60))) * radius
    #    plt.scatter(x,y,color = "orange")
    
    e#現在
    mean#全体平均
    mean_100
    
    #右半分
    plt.subplot(122)
    plt.text(0, 0.8,"評価：",size=30)
    if len(hist) > near:
        if mean < mean_100:
            if mean_100 < e:
                plt.text(0, 0.8,"　　　Excellent",size=30,color='r')
                rank.append(0)
            elif mean_100 >= e and mean < e:
                plt.text(0, 0.8,"　　　Good",size=30,color='y')
                rank.append(1)
            else:
                #plt.text(0, 0.8,"　　　降下中!",size=30,color='r')
                plt.text(0, 0.8,"　　　Poor",size=30,color='g')
                rank.append(2)
        else:
            if mean_100 < e and e < mean:
                plt.text(0, 0.8,"　　　Poor",size=30,color='g')
                rank.append(2)
            elif mean_100 >= e:
                plt.text(0, 0.8,"　　　Bad",size=30,color='b')
                rank.append(3)
            else:
                #plt.text(0, 0.8,"　　　上昇中!",size=30,color='r')
                plt.text(0, 0.8,"　　　Good",size=30,color='y')
                rank.append(1)
    else:
        plt.text(0, 0.8,"　　　---",size=30)
    plt.text(0, 0.6,"作業効率："+str(e),size=30)
    if len(hist)>near:
        plt.text(0, 0.4,"直近の平均："+str(mean_100),size=30)
    else:
        plt.text(0, 0.4,"直近の平均：---",size=30)
    plt.text(0, 0.2,"全体の平均："+str(round(mean,2)),size=30)
    plt.axis("off")
    
if __name__=="__main__":
    main()