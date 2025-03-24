# 数据集下载，并且解压
import pandas as pd
import os   # 用于操作系统文件路径
import requests #用于处理网络请求
import zipfile  # 用于解压ZIP文件
import shutil # 用于高级文件操作（删除目录）
import glob
import json
def download_dataset(download_list_file_path,data_path):

    download_list = pd.read_csv(download_list_file_path)
    mypath = data_path
    if not os.path.exists(mypath):
        os.makedirs(mypath)

    for i, link in enumerate(download_list['link']):
        file_name = download_list['file'][i]
        print(file_name, '\t', link)
        file_path = mypath + file_name
        # 如果数据集还没有下载，就下载文件（避免重复下载）
        if not os.path.exists(file_path):
            # 发起HTTP请求（stream=True表示流式下载）
            response = requests.get(link,stream=True)
            print("downloading to ",file_path)
            # 分块写入文件（防止内存溢出)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk: f.write(chunk)
    # 需要解压的压缩包列表
    zip_list = ['mchar_train', 'mchar_test_a', 'mchar_val']

    for zip_name in zip_list:
        # 构建解压路径
        target_path = mypath + zip_name
        # 检查是否已解压过
        if not os.path.exists(target_path):
            print("解压文件",zip_name)
            # 打开解压文件
            with zipfile.ZipFile(mypath + zip_name + '.zip','r') as zf:
                # 解压到指定目录
                zf.extractall(mypath)

    # 清理Mac系统自动生成的__MACOSX目录（如果存在）
    maccox_dir = mypath + '__MACOSX'
    if os.path.exists(maccox_dir):
        print("删除__MACCOX目录")
        shutil.rmtree(maccox_dir)

#定义目录路径
data_dir = {
    'train_data': './tcdata/mchar_train/',
    'val_data': './tcdata/mchar_val/',
    'test_data': './tcdata/mchar_test_a/',
    'train_label': './tcdata/mchar_train.json',
    'val_label': './tcdata/mchar_val.json',
    'submit_file': './tcdata/mchar_sample_submit_A.csv'
}

#统计train,val,test数据集的个数
def data_summary():
  train_list = glob.glob(data_dir['train_data']+'*.png')
  test_list = glob.glob(data_dir['test_data']+'*.png')
  val_list = glob.glob(data_dir['val_data']+'*.png')
  print('train image counts: %d'%len(train_list))
  print('val image counts: %d'%len(val_list))
  print('test image counts: %d'%len(test_list))


def look_submit():
    df = pd.read_csv(data_dir['submit_file'])
    print(df.head())


def label_summary():
    marks = json.loads(open(data_dir['train_label'], 'r').read())
    dicts={}
    for img,mark in marks.items():
        if len(mark['label']) not in dicts:
            dicts[len(mark['label'])] = 0
        dicts[len(mark['label'])] += 1

    dicts = sorted(dicts.items(), key=lambda x: x[0])
    for k,v in dicts:
        print('%d个数字的图片数目: %d' % (k, v))