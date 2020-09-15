
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from pylab import rcParams
from scipy.spatial import ConvexHull
import random
import os
from PIL import Image

# mpl.use('PS')
mpl.rcParams['text.usetex'] = True
# rcParams['figure.figsize'] = 8.5, 5
rcParams['figure.figsize'] = 5, 5

# plt.gcf().subplots_adjust(right=0.5)
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 12}

mpl.rc('font', **font)

def plot_dataset():
    folders = glob("Images/*")

    class_names = [i.replace('Images/','') for i in folders]
    class_names = [i.replace('_', ' ') for i in class_names]
    freq = []

    for path in folders:
        images = glob(f'{path}/*.jpg')
        images = [i.replace(f'{path}/', '') for i in images]
        freq.append(len(images))


    df = pd.DataFrame(list(zip(class_names, freq)),
                      columns=['Class Names', 'Freq'])
    print(df)

    bars =  df['Class Names']
    y_pos = np.arange(len(bars))
    height = df['Freq']
    plt.bar(y_pos, height)

    # # Decoration
    plt.title(f"Frequency Distribution of Indoor Classes", fontsize=12)
    plt.xlabel('Indoor Scenes')
    plt.ylabel("Frequency")
    plt.xticks(y_pos, bars)
    plt.xticks(rotation=90)
    plt.show()

    plt.savefig('plots/class_imbalance_1.eps', format='eps')

# Encircle
def encircle(x,y, ax=None, **kw):
    if not ax: ax=plt.gca()
    p = np.c_[x,y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices,:], **kw)
    ax.add_patch(poly)

def plot_all_results():
    df = pd.read_csv('results/results.csv')

    print(df.loc[df['Feature Extractor'] == 'resnext101', 'Test Time'])
    print(df.loc[df['Feature Extractor'] == 'resnext1010', 'Test F1 Score'])

    sns.scatterplot(data=df[df.Classifier=='nn'], x="Test Time", y="Test F1 Score", marker='D', label='Neural Network')
    sns.scatterplot(data=df[df.Classifier == 'svm'], x="Test Time", y="Test F1 Score", marker='v', label='SVM')
    sns.scatterplot(data=df[df.Classifier == 'naive-bayes'], x="Test Time", y="Test F1 Score", marker='x', label='Naive Bayes')
    sns.scatterplot(data=df[(df.Classifier == 'decision-tree')], x="Test Time", y="Test F1 Score", marker='2', label='Decision Tree')
    sns.scatterplot(data=df[(df.Classifier == 'random-forest')], x="Test Time",
                    y="Test F1 Score", marker='1', label='Random Forest')
    sns.scatterplot(data=df[df.Classifier == 'knn'], x="Test Time", y="Test F1 Score", marker='^', label='K-Nearest Neighbour')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f"Test F1 Score vs Inference Time", fontsize=12)
    plt.ylabel('Test F1 Score [0-1]')
    plt.xlabel("Inference Time (s)")
    plt.savefig(f'plots/all.eps', format='eps', bbox_inches='tight')
    plt.show()

def plot_nn(type):
    if type=='nn':
        marker = 'D'
        label = 'Neural Network'
    elif type=='knn':
        marker = '^'
        label = 'K-Nearest Neighbour'
    elif type == 'svm':
        marker = 'v'
        label = 'Support Vector Machine'
    elif type == 'naive-bayes':
        marker = 'x'
        label = 'Naive Bayes'

    df = pd.read_csv('results/results.csv')

    df = df[df.Classifier == type]
    df = df.sort_values('Test F1 Score', ascending=False)

    df = df.iloc[0:len(df) if len(df) <= 16 else 16]

    print(df)

    x = df['Test Time'].to_list()
    y = df['Test F1 Score'].to_list()
    labels = df['Experiment Name'].to_list()

    for i in range(len(x)):
        print(x[i])
        ax = sns.scatterplot(x=[x[i]], y=[y[i]], marker='D',  label=labels[i])

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f"Test F1 Score vs Inference Time for {label}", fontsize=12)
    plt.ylabel('Test F1 Score [0-1]')
    plt.xlabel("Inference Time (s)")

    plt.savefig(f'plots/{type}.eps', format='eps', bbox_inches='tight')
    plt.show()



def plot_dt():

    df = pd.read_csv('results/results.csv')
    df = df[(df.Classifier == 'decision-tree') | (df.Classifier == 'random-forest')]

    dt = df[(df.Classifier == 'decision-tree')].sort_values('Test F1 Score', ascending=False).iloc[0:8]
    rf = df[(df.Classifier == 'random-forest')].sort_values('Test F1 Score',  ascending=False).iloc[0:8]

    x = dt['Test Time'].to_list()
    y = dt['Test F1 Score'].to_list()
    labels = dt['Experiment Name'].to_list()

    for i in range(len(x)):
        ax = sns.scatterplot(x=[x[i]], y=[y[i]], marker='D',  label='decison-tree ' + labels[i])


    x = rf['Test Time'].to_list()
    y = rf['Test F1 Score'].to_list()
    labels = rf['Experiment Name'].to_list()

    for i in range(len(x)):
        ax = sns.scatterplot(x=[x[i]], y=[y[i]], marker='X',  label='random-forest ' + labels[i])

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f"Test F1 Score vs Inference Time for Decison Tree and Random Forest", fontsize=12)
    plt.ylabel('Test F1 Score [0-1]')
    plt.xlabel("Inference Time (s)")
    plt.savefig(f'plots/decison_tree.eps', format='eps', bbox_inches='tight')
    plt.show()

def plot_dataset_images():
    images = []
    train_folder = 'Images'
    for folder in os.listdir(train_folder):
        try:
            for image in os.listdir(train_folder + '/' + folder):
                images.append(os.path.join(train_folder, folder, image))
        except NotADirectoryError:
            continue

    plt.figure(1, figsize=(15, 9))
    plt.tight_layout()

    n = 0
    for i in range(16):
        n += 1
        random_img = random.choice(images)

        title = random_img.split('/')[1]
        title = title.replace('_',' ')
        print(title)
        imgs = Image.open(random_img)
        plt.subplot(4, 4, n)
        plt.title(title)
        plt.axis('off')
        plt.imshow(imgs)

    plt.savefig(f'plots/dataset.eps', format='eps')
    plt.show()


if __name__ == '__main__':
    plot_dataset()
    plot_all_results()
    plot_nn('knn')
    plot_nn('nn')
    plot_nn('naive-bayes')
    plot_nn('svm')
    plot_dt()
    plot_dataset_images()
