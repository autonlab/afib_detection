import datetime as dt
import json
import numpy as np
import os
import pandas as pd

def parseAnnotation(x):
    x = json.loads(x)
    res = pd.Series([x[k] for k in x.keys()])
    return res
    
def parseAndAddAnnotationsToDf(path):
    column_names = 'id,user_id,project_id,file_id,pattern_id,pattern_set_id,series,left,right,top,bottom,label,created_at,id.1,project_id.1,path'.split(',')
    df = pd.read_csv(path)
    print(list(df.columns))
    assert(list(df.columns) == column_names)

    userMap = {
        36: 'kaufmanr',
        7: 'gcler',
        20: 'rooneys'
    }
    df['annotation'] = df['label']
    print(df.columns)
    df['user_id'] = df['user_id'].apply(lambda x: userMap[x])

    df['start'] = df['left'].apply(dt.datetime.fromtimestamp)
    df['stop'] = df['right'].apply(dt.datetime.fromtimestamp)

    df['fin_study_id'] = df['path'].apply(lambda x: x.split(os.sep)[-1].split('_')[-1].split('.')[0])
    df[[x for x in json.loads(df['annotation'][0]).keys()]] = df['annotation'].apply(parseAnnotation)
    # df = df.drop(columns=['annotation','id.1', 'pattern_id', 'pattern_set_id', 'top', 'bottom', 'created_at', 'project_id.1', 'label'])
    df = df.drop(columns=['annotation','id.1', 'pattern_id', 'pattern_set_id', 'top', 'bottom', 'project_id.1', 'label'])
    return df

def filterDF(df):
    # shared = pd.read_csv('./assets/stratified_gold_shared2.csv')
    # rest = pd.read_csv('./assets/stratified_gold_rest2.csv')
    # all = pd.concat([shared, rest])
    # df = df[df['fin_study_id'].astype(np.int64).isin(all['fin_study_id'])]
    df['isduplicate'] = df.duplicated(subset=['fin_study_id', 'start'], keep='last')
    df = df[ ~df['isduplicate']]
    df = df[df['notes'] != 'no data']
    return df

def idk(df):
    res = list()
    for idx, row in df.iterrows():
        if type(row['rhythm_label']) == float:
            rowContent = row['pvc_presence']
            if ',' in row['pvc_presence']:
                rhythm = rowContent.split(',')[0].split()[0]
                rest = rowContent.split(',')[1:]
            elif ';' in row['pvc_presence']:
                rhythm = rowContent.split(';')[0].split()[0]
                rest = rowContent.split(';')[1:]
            else:
                rhythm = rowContent.split()[0]
            if rhythm.lower() == 'atrial':
                row['label'] = 'ATRIAL_FIBRILLATION'
            elif rhythm.lower() == 'sinus':
                row['label'] = 'SINUS'
            elif rhythm.lower() == 'other':
                row['label'] = 'OTHER'
            elif rhythm.lower() == 'noise':
                row['label'] = 'NOISE'
            else:
                continue
        else:
            row['label'] = '_'.join(row['rhythm_label'].split()).upper()
        res.append(row)
    result = pd.DataFrame(res)
    return result

if __name__ == '__main__':
    # parsed = parseAndAddAnnotationsToDf('./mostrecentannotations.csv')
    # print(len(parsed))
    # parsed = parsed[parsed['user_id'] == 'rooneys']
    # print(len(parsed))
    # print(len(parsed))
    # parsed = filterDF(parsed)
    # print(len(parsed))
    # print(parsed.columns)
    # print(parsed)
    # parsed.to_csv('final_annotations.csv', index=False)
    # idk(pd.read_csv('final_annotations.csv')).to_csv('final_annotations.csv', index=False)
    # parsed = filterDF(parsed)
    # parsed.to_csv('./filtered_annotations_2022-04-12.csv')