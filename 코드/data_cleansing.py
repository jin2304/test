import pandas as pd
import re

# 엑셀 파일 읽기
df = pd.read_excel('data/1.raw/광고.xlsx', names=['post_hashtags', 'post_texts'])

# 열 이름 변경


# 라벨을 1로 설정
df['label'] = 1
df['post_hashtags'] = ''

# 일반 게시글에 포함될 수 있는 키워드 정의
advertising_tags = [
    '#내돈내산', '#내돈내산후기', '#제품리뷰',
    '#리뷰', '#찐리뷰', '#찐리뷰어',
    '#리뷰그램', '#찐후기', '#광고아님',
    '#협찬아님', '#광고협찬아님', '#광고면좋겠다',
    '#협찬이면좋겠다', '#솔직후기', '광고협찬아님',
    '광고면좋겠다', '협찬이면좋겠다', '솔직후기',
    '내돈내산','찐리뷰', '협찬아님'
]

# 한글 비율을 계산하는 함수 정의
def korean_ratio(text):
    korean_characters = re.findall(r'[가-힣]', text)
    return len(korean_characters) / len(text) if len(text) > 0 else 0

# 1. 'post_texts' 열에서 중복 데이터 제거
df.drop_duplicates(subset='post_texts', inplace=True)

# 2. 결측치 제거 (text 모두 결측치가 있는 행 제거)
df.dropna(subset=['post_texts', 'post_hashtags'], inplace=True)

# 3. 일반 해시태그가 포함된 데이터 필터링
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
df.to_excel('data/2.정제/광고_정제3.xlsx', index=False)
