import os
import pandas as pd

phrases = ['Happily Surprised', 'Sadly Fearful', 'Sadly Angry', 'Sadly Surprised', 'Fearfully Surprised', 'Angrily Surprised', 'Disgustedly Surprised']


def count_phrases(file_path):
    counts = {}
    with open(file_path, 'r') as file:
        text = file.read()
        for phrase in phrases:
            count = text.count(phrase)
            if count > 0:
                counts[phrase] = count
    return counts

def step1():
    data = []
    directory = './abaw-test/images_aligned'
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.jpg'):
                jpg_path = os.path.join(root, filename)
                txt_path = os.path.splitext(jpg_path)[0] + '.txt'
                if os.path.exists(txt_path):
                    label_counts = count_phrases(txt_path)
                    if len(label_counts) > 0:
                        label = max(label_counts, key=label_counts.get)
                        if 'Happily Surprised' in label_counts:
                            label = 'Happily Surprised'
                        elif 'Sadly Fearful' in label_counts:
                            label = 'Sadly Fearful'
                        label_index = phrases.index(label)
                        data.append((jpg_path, label_index))
                    else:
                        data.append((jpg_path, -1))
                else:
                    data.append((jpg_path, -1))

    df = pd.DataFrame(data, columns=["img_name", "label"])
    df_sorted = df

    df_sorted["img_name"] = df_sorted["img_name"].apply(lambda x: x.split("/")[-2] + '/' + x.split("/")[-1])
    df_sorted = df.sort_values(by="img_name")

    value_counts = df_sorted['label'].value_counts()
    print(value_counts)

    df_sorted.to_csv("abaw_all.csv", index=False)
    
    
def step2():
    df = pd.read_csv('abaw_all.csv')
    part_list = []

    for i, row in df.iterrows():
        if row['label'] == -1:
            image_name = row['img_name']
            first_part = int(image_name.split("/")[0])
            second_part = int(image_name.split("/")[1].split(".")[0])
            part_list.append((first_part, second_part))
            
    part_list.sort(key=lambda x: (x[0], x[1]))

    for first_part, second_part in part_list:
        if second_part > 1:
            fixed_img_name = f"{first_part:02d}/{second_part-1}.jpg"
        else:
            fixed_img_name = f"{first_part-1:02d}/{second_part+1}.jpg"
        fixed_label = df.loc[df['img_name'] == fixed_img_name, 'label'].values[0]
        df.loc[(df['img_name'] == f"{first_part:02d}/{second_part}.jpg") & (df['label'] == -1), 'label'] = fixed_label

    label_counts = df['label'].value_counts()
    print(label_counts)

    df.to_csv('abaw_all_fixed.csv', index=False)
    
def step3():
    df = pd.read_csv('abaw_all_fixed.csv')
    part_list = []

    for i, row in df.iterrows():
        image_name = row['img_name']
        label = row['label']
        first_part = int(image_name.split("/")[0])
        second_part = int(image_name.split("/")[1].split(".")[0])
        part_list.append((first_part, second_part, label))

    part_list.sort(key=lambda x: (x[0], x[1]))

    data = []
    for first_part, second_part, label in part_list:
        data.append([f"{first_part:02d}/{second_part}.jpg", label])

    df = pd.DataFrame(data, columns=["img_name", "label"])

    df.to_csv("abaw_all_fixed_sorted.csv", index=False)
    
step1()
step2()
step3()