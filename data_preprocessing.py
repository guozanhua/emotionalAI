import pandas as pd
import shutil, os
import math

video_dict = {}
split_dict = {}
exception_arr = []
root_dir = 'data/'
directories = [root_dir+'train/', root_dir+'test/']
Excel_file = 'data/VideoEmotionDataset-TrainTestSplits.xlsx'


def prepare_video_data():

    video_data = pd.read_excel(Excel_file, sheetname=0)
    for index, row in video_data.iterrows():
        if index == 0:
            continue
        video_dict[index] = row['Video Name and Directory']

def prepare_train_test_data(split_value='Split 2'):
    split_dict = {
        'Split 1': [0, 1],
        'Split 2': [2, 3],
        'Split 3': [4, 5],
        'Split 4': [6, 7],
        'Split 5': [8, 9],
        'Split 6': [10, 11],
        'Split 7': [12, 13],
        'Split 8': [14, 15],
        'Split 9': [16, 17],
        'Split 10': [18, 19]
    }
    split_data = pd.read_excel(Excel_file, sheetname=1, header=[0, 1], parse_cols=split_dict[split_value])
    # for index, row in split_data[split_value].iteritems():
    for i in split_data.itertuples():
        category_folder = []
        if not math.isnan(i[0]):
            # print("Train:",i[0])
            train_video =  video_dict[int(i[0])]
            category_folder.append(train_video.split('/')[0])
            # print(video_dict[int(i[0])])
        else: 
            train_video = None
            category_folder.append(None)
        
        if not math.isnan(i[1]):
            # print("test:", i[1])
            test_video =  video_dict[int(i[1])]
            category_folder.append(test_video.split('/')[0])
            # print(video_dict[int(i[1])])
        else:
            test_video = None
            category_folder.append(None)
        
        for cate, dirr in zip(category_folder, directories):
            if cate is None:
                continue
            if not os.path.exists(dirr+cate):
                os.makedirs(dirr+cate)
        try:
            if train_video is not None:
                shutil.copyfile(root_dir+train_video, directories[0]+category_folder[0]+'/'+train_video.split('/')[2])
            if test_video is not None:
                shutil.copyfile(root_dir+test_video, directories[1]+category_folder[1]+'/'+test_video.split('/')[2])
        except IOError as e:
            exception_arr.append(e)
            


def main():
    for directory in directories: 
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
        else:
            os.makedirs(directory)
    prepare_video_data()
    prepare_train_test_data()
if __name__ == "__main__":
    main()