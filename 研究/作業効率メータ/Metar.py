import csv
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from matplotlib.animation import FuncAnimation

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

hist_range = int(T/f)#ヒストグラムにする幅
start = 0
end = hist_range

update_range = int(update_time/f)
hist = []

def main():
    fig = plt.figure(figsize = (20,4), facecolor='white')
    ani = FuncAnimation(fig, plot_hist, interval=250)
    plt.show()
    plt.plot(hist)
    plt.title("作業効率の変化")
    plt.show()

def plot_hist(i):
    if i != 0:
        plt.clf() 
    #fig,ax = plt.subplots(1,2,tight_layout=True)
    #左半分
    plt.subplot(121)
    #ax[0].gca().axis('off')
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

    
    #plt.axvline(x=e+0.5, ymin=0, ymax=100,color="red",label="現在の作業効率")
    #plt.axvline(x=mean, ymin=0, ymax=100,color="lightgreen",label="全体の平均")
    if len(hist)>100:
        new_hist=hist[(len(hist)-100):]
        mean_100=np.mean(new_hist)
        x = np.cos(np.radians(180 * ((60-mean_100) / 60))) * radius
        y = np.sin(np.radians(180 * ((60-mean_100) / 60))) * radius
        plt.scatter(x,y,color = "orange")
        #plt.axvline(x=mean_100+0.5, ymin=0, ymax=100,color="orange",label="直近の平均")
        #past_hist=hist[:len(hist)-100]
        #plt.hist([past_hist,new_hist],bins=20,range=(30,50),ec="black",color=['#1f77b4', '#ff7f0e'],stacked=True)
    #else:
        #plt.hist(hist,bins=20,range=(30,50),ec="black",color='#1f77b4',stacked=True)
    #plt.xticks(np.arange(30, 55, 5))
    #plt.legend()
    #plt.title('作業効率ヒストグラム',{'fontsize':15})     
    #plt.xlabel('1分間の採点回数(作業効率)',{'fontsize':15}) 
    #plt.ylabel('度数',{'fontsize':15}) 
    
    e#現在
    mean#全体平均
    mean_100
    
    #右半分
    plt.subplot(122)
    plt.text(0, 0.8,"評価：",size=30)
    if len(hist) > 100:
        if mean < mean_100:
            if mean_100 < e:
                plt.text(0, 0.8,"　　　Excellent",size=30,color='r')
            elif mean_100 >= e and mean < e:
                plt.text(0, 0.8,"　　　Good",size=30,color='y')
            else:
                #plt.text(0, 0.8,"　　　降下中!",size=30,color='r')
                plt.text(0, 0.8,"　　　Poor",size=30,color='g')
        else:
            if mean_100 < e and e < mean:
                plt.text(0, 0.8,"　　　Poor",size=30,color='g')
            elif mean_100 >= e:
                plt.text(0, 0.8,"　　　Bad",size=30,color='b')
            else:
                #plt.text(0, 0.8,"　　　上昇中!",size=30,color='r')
                plt.text(0, 0.8,"　　　Good",size=30,color='y')
    else:
        plt.text(0, 0.8,"　　　---",size=30)
    plt.text(0, 0.6,"作業効率："+str(e),size=30)
    if len(hist)>100:
        plt.text(0, 0.4,"直近の平均："+str(mean_100),size=30)
    else:
        plt.text(0, 0.4,"直近の平均：---",size=30)
    plt.text(0, 0.2,"全体の平均："+str(round(mean,2)),size=30)
    plt.axis("off")
    
if __name__=="__main__":
    main()