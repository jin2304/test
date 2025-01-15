# model_train코드, 훈련코드
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, Layer
import tensorflow.keras.backend as K
import re
import emoji
import MeCab
import json



# GPU 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print("GPU Name:", gpu.name)
        print("GPU Details:", tf.config.experimental.get_device_details(gpu))
else:
    print("No GPU available.")


# MeCab 설정
mecab = MeCab.Tagger("-d C:/mecab/mecab-ko-dic")


# 한국어 불용어 리스트
korean_stopwords = [
    '의', '가', '이', '은', '는', '들', '에', '와', '한', '하다', '것', '그', '되', '수',
    '아니', '같', '그렇', '위하', '말', '일', '때', '있', '하', '보', '않', '없', '감', '편', '좋', '아요', '번', '트', '게다가',
    '나', '사람', '주', '등', '년', '지', '로', '대하', '오', '그리고', '그래서', '크', 'ketohye', '에요', '차', '얼', '핑', '이제',
    '중', '에서', '넬로', '잘', '고', '을', '으로', '게', '를', '도', '다', '어', '기', '해', '후', '많', '다고', '링', '네요', '으면', 't',
    '아', '면', '니', '는데', '었', '과', '더', '시', '내산', '팔', '개', '문', '입니다', '백', '맛', '음', '노티', '마',
    '스타', '먹', '인', '만', '까지', '입', '할', '빗', '데', '다가', '천', '점', '넘', '명', '랑', '이나', '외', '아서', '덕스',
    '았', '습니다', '거', '세요', '적', '했', '님', '라', '어서', '봤', '맘', '용', '희', '맥', '함',
    '여름', '분', '안', '해요', '지만', '스', '신', '제', '집', '던', '용쓰', '네', '성', '받', '면서', '원', '아기', '해서',
    '아이', '저', '서', '살', '로워', '덕', '맞', '요', '겠', '싶', '타', '쓰', '어요', '반', '두', '자', '세', '죠', '내', '사',
    '플', '였', '에게', '께', '부터', '니까', '셨', '났', '인데', '으니', '된', '엔', '그런', '왔', '늘', '며', '스럽', '듯', '해야',
    '라고', '예요', '동안', '처럼', '은데', '더니', '다는', '한다', '는데요', '써', '다면', '나와', '쌈닭','홀딱',
<<<<<<< HEAD
    '반한','치킨','편하', '강아지', '닭', '볼', '램','룩','사이','블랙','전','티','템','애', '히피', '밀키',
    '싸','믿','밀','셀','럽','구','선','뭐', '쉽', '나왔', '영','무','덤','fff','롬','먹스','셔','쿠','쥬', '든', '틱','셔서', '피', '올', '첫',
    '네', '베', '킨', '베스', '니깐', '라베라', '시연', '룬', 'mlbb', 'nail', 'getregrammer', '권', '따', 's', '따', '재', '커리', '쉴', 'ROCFIT', '여', '엠',
    '왕', '칭', 'h', 'k', '수노', '베베', "무아애", "cm","CM", '화', '브', 'mlbb', '노연', '용하', 'd', 'bitly', 'huggieshappyhug', '제니', '옐로', '소희', '마시',
    '로토토', '얇', '노즈', 'MLBB', 'mlbb', 'midowatches', 'ndmvopt', '헤', '율', '느냐', 'ssoh', 'm', '피너클', '텐', '웨', '피펫', '퐁', 'jieun', '리', '타월', '꿍'
=======
    '반한','치킨','편하', '강아지', '닭', '볼', '램','룩','사이','블랙','전','티','템','애',
    '싸','믿','밀','셀','럽','구','선','뭐', '쉽', '나왔', '영','무','덤','fff','롬','먹스','셔','쿠','쥬', '든', '틱','셔서', '피', '올', '첫',
    '네', '베', '킨', '베스', '니깐', '라베라', '시연', '룬', 'mlbb', 'nail', 'getregrammer', '권', '따', 's', '따', '재', '커리', '쉴', 'ROCFIT', '여', '엠',
    '왕', '칭', 'h', 'k', '수노', '베베', "무아애", "cm","CM", '화', '브', 'mlbb', '노연', '용하', 'd', 'bitly', 'huggieshappyhug', '제니', '옐로', '소희', '마시',
    '로토토', '얇', '노즈', 'MLBB', 'mlbb', 'midowatches', 'ndmvopt', '헤', '율', '느냐', 'ssoh', 'm', '피너클', '텐', '웨', '피펫', '퐁', 'jieun', '리', '타월', '꿍', '밀키',
    '히피',

    '부산','서면','신도림역', "dior", '제주시', '가디건', '스커트', '윤소원', '캔디', '의사', '팬츠', '레깅스', 'ddanziya', 'ppppsssh', '그린', '장난감', '치맥', '오븐',
    '샌들', '원목', '낙낙', '실내복', '디자인', '귀여운', '용사', '코팅', '책', '미소', '타임리스', '스트릿', '몬', '하기스', '선보이', '스트링',
    '네일', '머리띠', '즈', '네이처', 'moneycoon', '골프', 'nodress', '식기', '소파', '솔솔', '감성', '투', '퍼플', '팡팡', '쌩', '디엘', 'n', '변덕',
    '말차', '퀸', '돗', '신사', '패딩', '기억력', '조이', '눈물', '컷', '핏', '색상', '12', '트리', '파스타', '브라', '가족', '라운', '돼지갈비', '카리카', 'marys', '콤', '아우',
    '펫', '치', '날씨', '생맥주', '갈비', '렌즈', '외식', '마스크', '떡볶이',
>>>>>>> 38090bfa5ab13032f2de3f8db8f553cc81f6b3d0
]



# 텍스트 전처리 함수 (이모지 및 특수문자 제거)
def preprocess_text(text):
    if isinstance(text, str):
        text = emoji.replace_emoji(text, replace='')  # 이모지 제거
        #text = re.sub(r'[^가-힣0-9a-zA-Z\s]', '', text)  # 특수문자 제거
        text = re.sub(r'[^가-힣0-9a-zA-Z%*\s]', '', text)
        text = re.sub(r'[\n\r]+', ' ', text).strip()  # 줄바꿈 및 공백 제거
        return text
    return ""




#토큰 분리 함수
def split_custom_tokens(text):
    # URL 및 프로토콜 키워드 분리
    url_keywords = ["https", "http", "ftp", "www", "com"]
    url_pattern = r"(" + "|".join(url_keywords) + r")"  # 키워드 리스트를 정규식 패턴으로 변환
    text = re.sub(url_pattern, r" \1 ", text)  # 키워드 앞뒤로 공백 추가

    # 한글 키워드 분리
    korean_keywords = ["프로필", "링크", "협찬", "이벤트", "문의", "오픈", "가성비", "카톡", "공유"]
    korean_pattern = r"(" + "|".join(korean_keywords) + r")"  # 한글 키워드  -> r"(프로필|링크|협찬|이벤트|...)"
    text = re.sub(korean_pattern, r" \1 ", text)  # 키워드 앞뒤로 공백 추가

    # 영어/숫자 키워드 분리
    english_keywords = ["official", "repost", "010", "02", "055", "031", "000", "00",]
    english_pattern = r"(" + "|".join(english_keywords) + r")"  # 영어/숫자 키워드
    text = re.sub(english_pattern, r" \1 ", text)  # 키워드 앞뒤로 공백 추가

    # 중복 공백 제거 및 정리
    text = re.sub(r"\s+", " ", text).strip()  # 여러 공백을 하나로 줄이고, 양끝 공백 제거
    return text



# **토큰화 함수 (URL 분리 포함)**
def tokenize_text(text):
    if isinstance(text, str):
        # URL 분리 전처리
        text = split_custom_tokens(text)
        # 줄바꿈 및 불필요한 공백 제거
        text = re.sub(r'[\n\r]+', ' ', text).strip()
        # MeCab으로 토큰화
        tokens = mecab.parse(text).splitlines()[:-1]  # EOS 제외
        return [token.split('\t')[0] for token in tokens if token.split('\t')[0] not in korean_stopwords]  # 불용어 제거
    return text



# 데이터 로드 및 라벨 추가
ad_file_path = 'data/4.통합/광고_통합.xlsx'
non_ad_file_path = 'data/4.통합/일반_통합.xlsx'

# 광고 데이터/일반 데이터 로드 및 라벨 추가
ad_data = pd.read_excel(ad_file_path)
non_ad_data = pd.read_excel(non_ad_file_path)

# 광고 및 일반 데이터를 병합
data = pd.concat([ad_data, non_ad_data], ignore_index=True)

# 데이터 전처리 및 준비
data = data[['post_texts', 'label']].dropna()  # 필요한 열 추출 및 결측값 제거
data['original_text'] = data['post_texts']     # 원본 텍스트 보존
data['post_texts'] = data['post_texts'].astype(str)  # 문자열 변환
data['post_texts'] = data['post_texts'].apply(preprocess_text).apply(tokenize_text)  # 전처리 및 토큰화





# 데이터를 80:20 비율로 학습 및 테스트 세트로 분할
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])

X_train, y_train = train_data['post_texts'].values, train_data['label'].values
X_test, y_test = test_data['post_texts'].values, test_data['label'].values
original_test_texts = test_data['original_text'].tolist()

# 텍스트 시퀀스를 길이에 맞게 패딩
<<<<<<< HEAD
max_nb_words = 40000
=======
max_nb_words = 30000
>>>>>>> 38090bfa5ab13032f2de3f8db8f553cc81f6b3d0
max_sequence_length = 250
embedding_dim = 256

# 시퀀스 패딩
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_nb_words) #tokenizer 객체 생성
tokenizer.fit_on_texts(X_train)  # 단어의 빈도를 계산하고 고유 정수 인덱스를 할당
X_train_seq = tokenizer.texts_to_sequences(X_train) # 전체 훈련 데이터를 정수 시퀀스로 변환
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post') #post는 시퀀스 뒤에 패딩(0)을 추가.

encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)





# Attention 레이어 정의
class Attention(Layer):
    def __init__(self, hidden_dim, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        # 학습 가능한 파라미터 초기화
        self.W_k = self.add_weight(
            name='weight_matrix', shape=(input_shape[-1], self.hidden_dim), # shape=(256, 500), 입력 차원(LSTM 히튼 상태) -> u_k
            initializer='random_normal', trainable=True
        )
        self.b_k = self.add_weight(
            name='bias_vector', shape=(self.hidden_dim,),  # 1D 편향 벡터 (히든 상태 차원)
            initializer='zeros', trainable=True
        )
        self.u_s = self.add_weight(
            name='context_vector', shape=(self.hidden_dim, 1),  # 2D로 초기화
            initializer='random_normal', trainable=True
        )
        super(Attention, self).build(input_shape)

    def call(self, inputs, return_attention=False):
        # Attention 계산
        u_k = K.tanh(K.dot(inputs, self.W_k) + self.b_k)  # 변환된 히든 상태 계산 (batch_size, time_steps(단어, 시퀀스 길이), hidden_dim) ex) 32개의 문장, 각 문장 100개의 단어, 단어당 256차원의 임베딩
        score = K.dot(u_k, self.u_s)       # Attention 점수 계산 (batch_size, time_steps, 1)
        score = K.squeeze(score, axis=-1)  # 차원 축소: 불필요한 차원 제거 (batch_size, time_steps)
        attention_weights = K.softmax(score, axis=1)  # Attention 가중치 계산 (batch_size, time_steps)
        context_vector = K.batch_dot(attention_weights, inputs, axes=[1, 1])  # 문맥 벡터 생성 (batch_size, input_dim)

        if return_attention:
            return context_vector, attention_weights
        return context_vector

    def get_config(self):
        # 레이어 구성 반환
        config = super(Attention, self).get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


##### 모델 정의 #####
#입력층(300) -> 엠베딩레이어(300, 256) -> LSTM 출력층=히든 상태(300, 256) → Attention 메커니즘(256) → 문맥 벡터(context_vector) → (256)Dense Layer(2) → 최종 예측값.
input_layer = Input(shape=(max_sequence_length,)) #정수 시퀀스 입력 #input_layer = Input(shape=(time_steps, input_dim))
embedding_layer = Embedding(max_nb_words, embedding_dim, input_length=max_sequence_length)(input_layer) #Embedding Layer를 사용하여 실수형 벡터로 변환.

spatial_dropout = SpatialDropout1D(0.2)(embedding_layer)
lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(spatial_dropout)
print("lstm_layer", lstm_layer) # LSTM 출력층(300) = 히든 상태 벡터 시퀀스 ex) [h1, h2, h3 ... ]

attention_layer = Attention(hidden_dim=500, name="attention") # hidden_dim: 변환될 히든 상태 차원(u_k)
context_vector, attention_weights = attention_layer(lstm_layer, return_attention=True)
output_layer = Dense(2, activation='softmax')(context_vector)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


##### 모델 학습 #####
epochs = 3
batch_size = 64
history = model.fit(X_train_pad, y_train_enc, validation_data=(X_test_pad, y_test_enc), epochs=epochs, batch_size=batch_size, verbose=2)

# 모델 훈련 완료 후 저장
<<<<<<< HEAD
model.save("data/6.훈련결과/model.h5")
print("모델이 'model.h5' 파일에 저장되었습니다.")

# Tokenizer 저장
with open("data/6.훈련결과/tokenizer.json", "w", encoding="utf-8") as f:
=======
model.save("model.h5")
print("모델이 'model.h5' 파일에 저장되었습니다.")

# Tokenizer 저장
with open("tokenizer.json", "w", encoding="utf-8") as f:
>>>>>>> 38090bfa5ab13032f2de3f8db8f553cc81f6b3d0
    f.write(tokenizer.to_json())
print("Tokenizer가 'tokenizer.json' 파일에 저장되었습니다.")








#####  샘플 데이터를 사용한 예측 수행 (라벨 포함) #####
sample_data = test_data.sample(200, random_state=42)
sample_texts = sample_data['post_texts'].tolist()          # 전처리된 텍스트
sample_labels = sample_data['label'].tolist()              # 실제 라벨
sample_original_texts = sample_data['original_text'].tolist()  # 전처리 전 원본 텍스트

sample_sequences = tokenizer.texts_to_sequences(sample_texts)
sample_padded = pad_sequences(sample_sequences, maxlen=max_sequence_length)






# Attention 가중치를 포함하는 모델 생성(학습 데이터 Attention 가중치 유지)
attention_model = Model(inputs=model.input, outputs=[model.output, attention_weights])
#attention_weights: Attention 레이어에서 계산된 가중치.

# 가중치 예측
predictions, all_attention_weights = attention_model.predict(sample_padded)

train_predictions, train_attention_weights = attention_model.predict(X_train_pad, batch_size=batch_size)
test_predictions, test_attention_weights = attention_model.predict(X_test_pad, batch_size=batch_size)

# 학습 및 테스트 데이터 Attention 가중치 결합
all_attention_weights = np.concatenate([train_attention_weights, test_attention_weights], axis=0)
all_sequences = np.concatenate([X_train_seq, X_test_seq], axis=0)
all_labels = np.concatenate([y_train_enc, y_test_enc], axis=0)



# Attention 가중치 및 시퀀스 길이 동기화
ad_word_attention_scores = {}
non_ad_word_attention_scores = {}

for i, (sequence, weights) in enumerate(zip(all_sequences, all_attention_weights)):
    label = all_labels[i]
    actual_length = min(len(sequence), len(weights))  # 시퀀스와 가중치 길이 동기화
    for j in range(actual_length):
        word_id = sequence[j]
        if word_id != 0:  # 패딩 값은 제외
            if label == 1:  # 광고 데이터
                ad_word_attention_scores[word_id] = ad_word_attention_scores.get(word_id, 0) + weights[j]
            else:  # 일반 데이터
                non_ad_word_attention_scores[word_id] = non_ad_word_attention_scores.get(word_id, 0) + weights[j]

# 상위 20 단어 출력
sorted_ad_attention = sorted(ad_word_attention_scores.items(), key=lambda x: x[1], reverse=True)
sorted_non_ad_attention = sorted(non_ad_word_attention_scores.items(), key=lambda x: x[1], reverse=True)

top_ad_words = [(tokenizer.index_word.get(word_id, ""), score) for word_id, score in sorted_ad_attention[:150]]
top_non_ad_words = [(tokenizer.index_word.get(word_id, ""), score) for word_id, score in sorted_non_ad_attention[:70]]

print("\nTop 20 Important Words for 광고성 게시글:")
for word, score in top_ad_words:
    if word:  # 빈 단어는 제외
        print(f"{word}: {score:.4f}")

print("\nTop 20 Important Words for 일반 게시글:")
for word, score in top_non_ad_words:
    if word:  # 빈 단어는 제외
        print(f"{word}: {score:.4f}")





<<<<<<< HEAD
##### 테스트 데이터 분석 #####
=======
#####  테스트 데이터 분석 #####
>>>>>>> 38090bfa5ab13032f2de3f8db8f553cc81f6b3d0
results = []
for idx, (original_text, tokenized_text, true_label, proba) in enumerate(zip(sample_original_texts, sample_texts, sample_labels, predictions)):
    prediction = np.argmax(proba)
    predicted_label = "광고성 게시글" if prediction == 1 else "일반 게시글"
    actual_label = "광고성 게시글" if true_label == 1 else "일반 게시글"
    proba_str = f"[광고성 {proba[1] * 100:.2f}%, 일반 {proba[0] * 100:.2f}%]"

    # 토큰화된 텍스트를 단어로 복원
    processed_text = ' '.join(tokenized_text)

    # 결과 출력
    print(f"원본 텍스트: {original_text}\n\n전처리 텍스트: {processed_text}\n"
          f"실제 라벨: {actual_label}\n예측 라벨: {predicted_label}\n확률: {proba_str}\n{'ㅡ' * 50}")

    # 결과 저장
    results.append({
        "원본 텍스트": original_text,
        "전처리 텍스트": processed_text,
        "실제 라벨": actual_label,
        "예측 라벨": predicted_label,
        "확률": proba_str
    })





##### 상위 50개 광고성 키워드 추출 #####
top_150_ad_words = [(word, float(score)) for word, score in top_ad_words[:500] if word]

# 딕셔너리 형태로 변환
ad_dictionary = {
    "광고성_단어사전": {
        "description": "광고성 게시글 분류에 사용되는 주요 키워드와 가중치",
        "total_words": len(top_150_ad_words),
        "words": dict(top_150_ad_words)
    }
}

# JSON 파일로 저장
with open('data/6.훈련결과/광고성_단어사전.json', 'w', encoding='utf-8') as f:
    json.dump(ad_dictionary, f, ensure_ascii=False, indent=2)

<<<<<<< HEAD
print("광고성 단어사전이 '광고성_단어사전.json' 파일로 저장되었습니다.")
=======
print("광고성 단어사전이 '광고성_단어사전.json' 파일로 저장되었습니다.")

# 상위 150개 키워드와 가중치 출력
# print("\n상위 150개 광고성 키워드 및 가중치:")
# for word, score in top_150_ad_words:
#     print(f"{word}: {score:.4f}")


>>>>>>> 38090bfa5ab13032f2de3f8db8f553cc81f6b3d0
