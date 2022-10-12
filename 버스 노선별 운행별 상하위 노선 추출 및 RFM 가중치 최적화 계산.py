#!/usr/bin/env python
# coding: utf-8

# In[249]:


import pandas as pd


# In[250]:


df = pd.read_csv("충청남도 천안시_시내버스운수업체별노선현황_20220923.csv", encoding = 'CP949')
df


# # 1회운행거리, 1일운행횟수 컬럼 각 상하위 10개 버스 노선 추출

# In[251]:


# onetime_distance 상위 10개

df1 = df.groupby('노선번호').agg(onetime_distance = ('1회운행거리(km_편도기준)', 'max'),
                            perday_number = ('1일운행횟수(회_편도기준)', 'max')).sort_values(['onetime_distance', 'perday_number']
                                                                                   ,ascending = False)
df1.head(10)

# 532, 381, 380, 382, 383, 166, 540, 165, 640, 391


# In[252]:


# onetime_distance 하위 10개

df2 = df.groupby('노선번호').agg(onetime_distance = ('1회운행거리(km_편도기준)', 'max'),
                            perday_number = ('1일운행횟수(회_편도기준)', 'max')).sort_values(['onetime_distance', 'perday_number']
                                                                                   ,ascending = True)
df2.head(10)

# 462, 850, 662, 651, 840, 431, 99, 93, 98, 115


# In[253]:


# perday_number 상위 10개

df3 = df.groupby('노선번호').agg(onetime_distance = ('1회운행거리(km_편도기준)', 'max'),
                            perday_number = ('1일운행횟수(회_편도기준)', 'max')).sort_values(['perday_number', 'onetime_distance']
                                                                                   ,ascending = False)
df3.head(10)

# 12, 14, 13, 1, 11, 400, 81, 3, 90, 20


# In[254]:


# perday_number 하위 10개

df4 = df.groupby('노선번호').agg(onetime_distance = ('1회운행거리(km_편도기준)', 'max'),
                            perday_number = ('1일운행횟수(회_편도기준)', 'max')).sort_values(['perday_number', 'onetime_distance']
                                                                                   ,ascending = True)
df4.head(10)

# 99, 93, 98, 91, 590, 104, 492, 373, 105, 702


# In[287]:


df4.to_clipboard(index=False)


# In[ ]:





# # RFM 가중치 계산

# ## 최적 가중치를 찾기 위한 함수 및 사전 준비

# In[272]:


import pandas as pd
rfm_score = pd.read_csv('충청남도 천안시_시내버스운수업체별노선현황_20220923.csv',  encoding = 'CP949') ## 데이터 불러오기


## 필요 변수 추출
rfm_score = rfm_score[['노선번호', '1회운행거리(km_편도기준)', '1일운행횟수(회_편도기준)']]


import pandas as pd
import numpy as np
 
from tqdm import tqdm


def get_score(level, data, reverse = False):
    

    score = [] 
    for j in range(len(data)): 
        for i in range(len(level)): 
            if data[j] <= level[i]: 
                score.append(i+1) 
                break 
            elif data[j] > max(level): 
                score.append(len(level)+1) 
                break 
            else: 
                continue
    if reverse:
        return [len(level)+2-x for x in score]
    else:
        return score 
 
grid_number = 100 ## 눈금 개수, 너무 크게 잡으면 메모리 문제가 발생할 수 있음.
weights = []
for j in range(grid_number+1):
    weights += [(i/grid_number,j/grid_number,(grid_number-i-j)/grid_number)
                  for i in range(grid_number+1-j)]
num_class = 5 ## 클래스 개수
class_level = np.linspace(1,5,num_class+1)[1:-1] ## 클래스를 나누는 지점을 정한다.
total_amount_of_sales = rfm_score['1회운행거리(km_편도기준)'].sum() ## 1회운행거리 총합 = 총 운행거리
print(class_level)



# ## 최적 가중치를 찾는 코드 

# In[285]:


max_std = 0 ## 표준편차 초기값
for w in tqdm(weights,position=0,desc = '[Finding Optimal weights]'):
    ## 주어진 가중치에 따른 1회운행거리별 점수 계산
    score = w[0]*rfm_score['노선번호'] +                         w[1]*rfm_score['1회운행거리(km_편도기준)'] +                         w[2]*rfm_score['1일운행횟수(회_편도기준)'] 
    rfm_score['Class'] = get_score(class_level,score,True) ## 점수를 이용하여 고객별 등급 부여
    ## 등급별로 1회운행거리를 집계한다.
    grouped_rfm_score = rfm_score.groupby('Class')['1회운행거리(km_편도기준)'].sum().reset_index()

    ## 제약조건 추가 - 등급이 높은 노선들의 1회운행거리가 낮은 등급의 노선들보다 커야한다.
    grouped_rfm_score = grouped_rfm_score.sort_values('Class')
    
    temp_monetary = list(grouped_rfm_score['1회운행거리(km_편도기준)'])
    if temp_monetary != sorted(temp_monetary,reverse=True):
        continue
    
    
    ## 클래스별 1회운행거리를 총구매금액으로 나누어 클래스별 매출 기여도 계산
    grouped_rfm_score['1회운행거리(km_편도기준)'] = grouped_rfm_score['1회운행거리(km_편도기준)'].map(lambda x : x/total_amount_of_sales)
    std_sales = grouped_rfm_score['1회운행거리(km_편도기준)'].std() ## 매출 기여도의 표준편차 계산
    if max_std <= std_sales:
        max_std = std_sales ## 표준편차 최대값 업데이트
        optimal_weights = w  ## 가중치 업데이트


# In[286]:


print(optimal_weights)


# ## 최적 가중치 결과
# - 노선번호,	1회운행거리(km_편도기준), 1일운행횟수(회_편도기준)	순으로
# - (0.01, 0.61, 0.38)

# # 최적 가중치 계산 결과값을 통해 버스 노선 이용량 상하위 10개 추출

# - 1회운행거리가 길고, 1일 운행횟수도 많은 상위 10개 순위 추출
# - 이동 교통량이 많은 지역 (버스 이용량 승하차 인원 많다)

# In[303]:


df2 = df1.assign(total = df1['onetime_distance'] * 0.61 + df1['perday_number'] * 0.38).sort_values('total', ascending=False)
df2.head(10)


# - 12, 14, 13, 1, 400, 3, 11, 201, 81, 20 

# - 1회운행거리가 짧고, 1일 운행횟수도 작은 상위 10개 순위 추출
# - 이동 교통량이 적은 지역 (버스 이용량 승하차 인원 적다)

# In[302]:


# 가중치 파생변수 만들기

df2 = df1.assign(total = df1['onetime_distance'] * 0.61 + df1['perday_number'] * 0.38).sort_values('total', ascending=True)

df2.head(10)


# - 99, 93, 98, 431, 850, 91, 462, 651, 231, 662
