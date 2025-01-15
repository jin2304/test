import pandas as pd
import re

# 엑셀 파일 읽기
df = pd.read_excel('data/1.raw/일반.xlsx', names=['post_hashtags', 'post_texts'])

# 열 이름 변경


# 라벨을 0로 설정
df['label'] = 0
df['post_hashtags']= ''

# 광고성 해시태그 목록 정의
advertising_tags = [
    r'\b#광고\b', r'\b#광고입니다\b', r'\b#제품제공\b',
    r'\b#협찬\b', r'\b#협찬리뷰\b', r'\b#협찬광고\b',
    r'\b#협찬제품\b', '#카지노',r'\b#제품협찬\b',
    r'\b광고\b', r'\b광고입니다\b',
    r'\b제품제공\b',r'\b협찬\b', r'\b협찬리뷰\b',
    r'\b협찬광고\b',r'\b협찬제품\b', '카지노',
    r'\b제품협찬\b','010','링크',
    'dm', '디엠', '카톡',r'\b#도서협찬\b',
    r'\b도서협찬\b',r'\b협찬을\b', '.com',
    'DM', '공구', '이벤트', r'\b증정\b',
    '051','문의','https','http','좌표',
    '오피셜','official','체험단','메포',
    '유료광고','고객님후기']

# 한글 비율을 계산하는 함수 정의
def korean_ratio(text):
    korean_characters = re.findall(r'[가-힣]', text)
    return len(korean_characters) / len(text) if len(text) > 0 else 0

# 1. 'post_texts' 열에서 중복 데이터 제거
df.drop_duplicates(subset='post_texts', inplace=True)

# 2. 결측치 제거 (text 모두 결측치가 있는 행 제거)
df.dropna(subset=['post_texts', 'post_hashtags'], inplace=True)

# 3. 광고성 해시태그가 포함된 데이터 필터링
df = df[~df['post_hashtags'].str.contains('|'.join(advertising_tags), na=False)]
df = df[~df['post_texts'].str.contains('|'.join(advertising_tags), na=False)]

# 4. 텍스트 길이가 10글자 이하인 행 제거
df = df[df['post_texts'].str.len() > 10]

# 5.'post_texts' 열에서 한글 비율 계산
df['korean_ratio'] = df['post_texts'].apply(korean_ratio)

# 6.한글 비율이 5% 이하인 행 제거
df = df[df['korean_ratio'] > 0.05]

# korean_ratio 열 삭제 (필요 없는 경우)
df.drop(columns='korean_ratio', inplace=True)

# 정제된 데이터를 새로운 엑셀 파일로 저장
df.to_excel('data/2.정제/일반_정제3.xlsx', index=False)


