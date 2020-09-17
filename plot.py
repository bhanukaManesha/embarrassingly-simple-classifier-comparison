
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

def plot_all_results():
    rcParams['figure.figsize'] = 6,6
    df = pd.read_csv('results/results.csv')

    print(df.loc[df['Feature Extractor'] == 'resnext101', 'Test Time'])
    print(df.loc[df['Feature Extractor'] == 'resnext1010', 'Test F1 Score'])

    sns.scatterplot(data=df[df.Classifier=='nn'], x="Test Time", y="Test Accuracy", marker='.', label='Neural Network')
    sns.scatterplot(data=df[df.Classifier == 'svm'], x="Test Time", y="Test Accuracy", marker='+', label='SVM')
    sns.scatterplot(data=df[df.Classifier == 'naive-bayes'], x="Test Time", y="Test Accuracy", marker='x', label='Naive Bayes')
    sns.scatterplot(data=df[(df.Classifier == 'decision-tree')], x="Test Time", y="Test Accuracy", marker='2', label='Decision Tree')
    sns.scatterplot(data=df[(df.Classifier == 'random-forest')], x="Test Time",
                    y="Test Accuracy", marker='1', label='Random Forest')
    sns.scatterplot(data=df[df.Classifier == 'knn'], x="Test Time", y="Test Accuracy", marker='^', label='K-Nearest Neighbour')

    plt.legend(loc='center right', bbox_to_anchor = [0.75, 0.45])
    plt.title(f"Test Accuracy vs Inference Time per Image", fontsize=12)
    plt.xlabel("Inference Time per Image (s)")
    plt.ylabel("Test Accuracy [0-1]")
    plt.savefig(f'plots/all.eps', format='eps', bbox_inches='tight')
    plt.show()

def plot_classifier(type, ax):
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
    elif type == 'decision-tree':
        marker = 'x'
        label = 'Decision Tree'
    elif type == 'random-forest':
        marker = 'x'
        label = 'Random Forest'

    df = pd.read_csv('results/results.csv')

    df = df[df.Classifier == type]
    df = df.sort_values('Test Accuracy', ascending=False)

    resnet_df = df[df['Feature Extractor'] == 'resnext101']
    mnasnet_df = df[df['Feature Extractor'] == 'mnasnet1_0']

    resnet_df = resnet_df.iloc[0:len(resnet_df) if len(resnet_df) <= 6 else 6]
    mnasnet_df = mnasnet_df.iloc[0:len(mnasnet_df) if len(mnasnet_df) <= 6 else 6]


    resnext_x = resnet_df['Test Time'].to_list()
    resnext_y = resnet_df['Test Accuracy'].to_list()
    resnext_labels = resnet_df['Experiment Name'].to_list()

    for k in range(len(resnext_x)):
        ax = sns.scatterplot(x=[resnext_x[k]], y=[resnext_y[k]], marker='X',  label=resnext_labels[k], ax=ax)

    mnasnet_x = mnasnet_df['Test Time'].to_list()
    mnasnet_y = mnasnet_df['Test Accuracy'].to_list()
    mnasnet_labels = mnasnet_df['Experiment Name'].to_list()

    for k in range(len(mnasnet_x)):
        ax = sns.scatterplot(x=[mnasnet_x[k]], y=[mnasnet_y[k]], marker='+',  linewidth=1,  label=mnasnet_labels[k], ax=ax)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title(f"Test Accuracy vs Inference Time per Image for {label}", fontsize=12)
    return ax

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
    for i in range(8):
        n += 1
        random_img = random.choice(images)

        title = random_img.split('/')[1]
        title = title.replace('_',' ')
        print(title)
        imgs = Image.open(random_img)
        plt.subplot(1, 8, n)
        plt.title(title)
        plt.axis('off')
        plt.imshow(imgs)

    plt.savefig(f'plots/dataset-2.eps', format='eps', bbox_inches='tight')
    plt.show()

def subplot_all():
    rcParams['figure.figsize'] = 15, 15
    fig, axs = plt.subplots(3, 2)
    fig.subplots_adjust(wspace=0.8)

    for i, ax in enumerate(np.ravel(axs)):

        if i == 0:
            print(i, ax)
            plot_classifier('knn', ax)
        elif i == 1:
            print(i, ax)
            plot_classifier('nn', ax)
        elif i == 4:
            print(i, ax)
            plot_classifier('naive-bayes', ax)
        elif i == 2:
            print(i, ax)
            plot_classifier('svm', ax)
        elif i == 3:
            print(i, ax)
            plot_classifier('decision-tree',ax)
        elif i == 5:
            print(i, ax)
            plot_classifier('random-forest',ax)

    # Set common labels
    fig.text(0.5, 0.04, 'Inference Time per Image (s)', ha='center', va='center')
    fig.text(0.06, 0.5, 'Test Accuracy [0-1]', ha='center', va='center', rotation='vertical')

    plt.savefig(f'plots/combined.eps', format='eps', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # plot_dataset()
    plot_all_results()
    subplot_all()
    # plot_dataset_images()
