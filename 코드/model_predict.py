# AiServer, 서버코드
from fastapi import FastAPI
from fastapi import Request
from pydantic import BaseModel
import tensorflow as tf
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import re
import emoji
import MeCab
import json
import pandas as pd
from fastapi.staticfiles import StaticFiles


# FastAPI 인스턴스 생성
app = FastAPI()

# images 폴더 마운트
app.mount("/images", StaticFiles(directory="images"), name="images")

# 데이터 로드 및 분할
def load_test_data():
    # 테스트용 데이터 로드
    ad_test_data = pd.read_excel('data/7.테스트/광고_테스트.xlsx')
    non_ad_test_data = pd.read_excel('data/7.테스트/일반_테스트.xlsx')

    # 데이터 병합
    test_data = pd.concat([ad_test_data, non_ad_test_data], ignore_index=True)

    # 데이터 전처리
    test_data = test_data[['post_texts', 'label']].dropna()
    test_data['original_text'] = test_data['post_texts']
    test_data['post_texts'] = test_data['post_texts'].astype(str)
    test_data['post_texts'] = test_data['post_texts'].apply(preprocess_text).apply(tokenize_text)

    return test_data

# Attention 레이어 정의
class Attention(Layer):
    def __init__(self, hidden_dim, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.W_k = self.add_weight(
            name='weight_matrix', shape=(input_shape[-1], self.hidden_dim),
            initializer='random_normal', trainable=True
        )
        self.b_k = self.add_weight(
            name='bias_vector', shape=(self.hidden_dim,),
            initializer='zeros', trainable=True
        )
        self.u_s = self.add_weight(
            name='context_vector', shape=(self.hidden_dim, 1),
            initializer='random_normal', trainable=True
        )
        super(Attention, self).build(input_shape)

    def call(self, inputs, return_attention=False):
        u_k = K.tanh(K.dot(inputs, self.W_k) + self.b_k)
        score = K.dot(u_k, self.u_s)
        score = K.squeeze(score, axis=-1)
        attention_weights = K.softmax(score, axis=1)
        context_vector = K.batch_dot(attention_weights, inputs, axes=[1, 1])

        if return_attention:
            return context_vector, attention_weights
        return context_vector

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


def mask_attention_weights(attention_weights, sequence):
    """
    Attention 가중치에서 패딩된 부분을 0으로 설정
    :param attention_weights: Attention 가중치 (배열)
    :param sequence: 입력 시퀀스 (패딩된 정수 배열)
    :return: 패딩 제거 후 Attention 가중치
    """
    attention_weights = attention_weights.squeeze()  # 1차원으로 변환
    mask = sequence != 0  # 패딩 여부를 마스크로 생성
    attention_weights = attention_weights * mask  # 패딩된 위치의 Attention 값을 0으로 설정
    return attention_weights




# 기존 모델 로드
model = load_model("data/6.훈련결과/model.h5", custom_objects={"Attention": Attention})

# Attention 모델 정의
attention_model = tf.keras.models.Model(inputs=model.input, outputs=[model.output, model.get_layer("attention").output])
print("Attention model created successfully.")

# 기존 토크나이저 로드
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(open("data/6.훈련결과/tokenizer.json", "r", encoding="utf-8").read())

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
    '반한','치킨','편하', '강아지', '닭', '볼', '램','룩','사이','블랙','전','티','템','애',
    '싸','믿','밀','셀','럽','구','선','뭐', '쉽', '나왔', '영','무','덤','fff','롬','먹스','셔','쿠','쥬', '든', '틱','셔서', '피', '올', '첫',
    '네', '베', '킨', '베스', '니깐', '라베라', '시연', '룬', 'mlbb', 'nail', 'getregrammer', '권', '따', 's', '따', '재', '커리', '쉴', 'ROCFIT', '여', '엠',
    '왕', '칭', 'h', 'k', '수노', '베베', "무아애", "cm","CM", '화', '브', 'mlbb', '노연', '용하', 'd', 'bitly', 'huggieshappyhug', '제니', '옐로', '소희', '마시',
    '로토토', '얇', '노즈', 'MLBB', 'mlbb', 'midowatches', 'ndmvopt', '헤', '율', '느냐', 'ssoh', 'm', '피너클', '텐', '웨', '피펫', '퐁', 'jieun', '리', '타월', '꿍', '밀키',
    '히피',

     '부산','서면','신도림역', "dior", '제주시', '가디건', '스커트', '윤소원', '캔디', '의사', '팬츠', '레깅스', 'ddanziya', 'ppppsssh', '그린', '장난감', '치맥', '오븐',
     '샌들', '원목', '낙낙', '실내복', '디자인', '귀여운', '용사', '코팅', '책', '미소', '타임리스', '스트릿', '몬', '하기스', '선보이', '스트링',
     '네일', '머리띠', '즈', '네이처', 'moneycoon', '골프', 'nodress', '식기', '소파', 'sofa', '솔솔', '감성', '투', '퍼플', '팡팡', '쌩', '디엘', 'n', '변덕',
     '말차', '퀸', '돗', '신사', '패딩', '기억력', '조이', '눈물', '컷', '핏', '색상', '12', '트리', '파스타', '브라', '가족', '라운', '돼지갈비', '카리카', 'marys', '콤', '아우',
     '펫', '치', '날씨', '생맥주', '갈비', '렌즈', '외식', '마스크', '떡볶이',


    #광고 관련 키워드
    # '광고', '광고입니다', '제품제공', '협찬', '협찬리뷰', '협찬광고', '협찬제품',
    # '내돈내산', '내돈내산후기', '제품리뷰', '리뷰', '찐리뷰', '찐리뷰어', '리뷰그램', '찐후기',
    # '광고아님', '협찬아님', '후기',
    #
    # # 해시태그 포함 버전
    # '#광고', '#광고입니다', '#제품제공', '#협찬', '#협찬리뷰', '#협찬광고', '#협찬제품',
    # '#내돈내산', '#내돈내산후기', '#제품리뷰', '#리뷰', '#찐리뷰', '#찐리뷰어', '#리뷰그램',
    # '#찐후기', '#광고아님', '#협찬아님', '#후기',
]


# 텍스트 전처리 함수 (이모지 및 특수문자 제거)
def preprocess_text(text):
    if isinstance(text, str):
        text = emoji.replace_emoji(text, replace='')  # 이모지 제거
        text = re.sub(r'[^가-힣0-9a-zA-Z%\s]', '', text)  # 특수문자 제거
        text = re.sub(r'[\n\r]+', ' ', text).strip()  # 줄바꿈 및 공백 제거
        return text
    return ""

def split_custom_tokens(text):
    # URL 및 프로토콜 키워드 분리
    url_keywords = ["https", "http", "ftp", "www", "com"]
    url_pattern = r"(" + "|".join(url_keywords) + r")"  # 키워드 리스트를 정규식 패턴으로 변환
    text = re.sub(url_pattern, r" \1 ", text)  # 키워드 앞뒤로 공백 추가

    # 한글 키워드 분리
    korean_keywords = ["프로필", "링크", "협찬", "이벤트", "문의", "오픈", "가성비", "카톡"]
    korean_pattern = r"(" + "|".join(korean_keywords) + r")"  # 한글 키워드  -> r"(프로필|링크|협찬|이벤트|...)"
    text = re.sub(korean_pattern, r" \1 ", text)  # 키워드 앞뒤로 공백 추가

    # 영어/숫자 키워드 분리
    english_keywords = ["official", "repost", "010", "02", "055", "031", "000", "00",]
    english_pattern = r"(" + "|".join(english_keywords) + r")"  # 영어/숫자 키워드
    text = re.sub(english_pattern, r" \1 ", text)  # 키워드 앞뒤로 공백 추가

    # 중복 공백 제거 및 정리
    text = re.sub(r"\s+", " ", text).strip()  # 여러 공백을 하나로 줄이고, 양끝 공백 제거
    return text


# 토큰화 함수
def tokenize_text(text):
    if isinstance(text, str):
        text = split_custom_tokens(text)
        text = re.sub(r"[\n\r]+", " ", text).strip()
        tokens = mecab.parse(text).splitlines()[:-1]  # EOS 제외
        return [token.split('\t')[0] for token in tokens if token.split('\t')[0] not in korean_stopwords]
    return []

# 데이터 입력 모델
class PredictionRequest(BaseModel):
    text: str


templates = Jinja2Templates(directory="templates")

## JSON 파일 로드 및 예측 결과 반환
@app.get("/", response_class=HTMLResponse)
async def read_predictions(request: Request):
    # 테스트 데이터 로드
    test_data = load_test_data()

    # 예측을 위한 데이터 준비
    sample_texts = test_data['post_texts'].tolist()
    sample_labels = test_data['label'].tolist()
    sample_original_texts = test_data['original_text'].tolist()

    # 시퀀스 변환
    sample_sequences = tokenizer.texts_to_sequences(sample_texts)
    sample_padded = tf.keras.preprocessing.sequence.pad_sequences(
        sample_sequences,
        maxlen=250,
        padding='post'
    )

    predictions = []

    # 예측 수행
    for idx, (original_text, text, true_label, sequence, padded_seq) in enumerate(
            zip(sample_original_texts, sample_texts, sample_labels, sample_sequences, sample_padded)):

        # images 경로 설정 (1부터 10까지의 images)
        image_path = f"/images/{idx + 1}.jpg"

        # 예측 및 Attention 가중치 추출
        prediction, attention_data = attention_model.predict(padded_seq.reshape(1, -1))

        if isinstance(attention_data, tuple):
            attention_weights = attention_data[1]
        else:
            attention_weights = attention_data

        attention_weights = attention_weights.squeeze()

        # 중요 단어 추출
        word_importance = []
        # 시퀀스 길이를 attention_weights 길이로 제한
        sequence_length = min(len(sequence), len(attention_weights))

        for i in range(sequence_length):
            word_id = sequence[i]
            if word_id != 0:  # 패딩 제외
                word = tokenizer.index_word.get(word_id, "")
                if word and word in ad_words_dict:
                    score = float(attention_weights[i])
                    dictionary_score = float(ad_words_dict[word])
                    word_importance.append({
                        "word": word,
                        "score": score,
                        "dictionary_score": dictionary_score
                    })

        if word_importance:
            word_importance = sorted(word_importance, key=lambda x: x["score"], reverse=True)[:15]
            important_words = ", ".join([f"{w['word']} ({w['score']:.2%})" for w in word_importance])
        else:
            important_words = "광고성 단어가 발견되지 않았습니다"

        actual_label = "광고성 게시글" if true_label == 1 else "일반 게시글"

        predictions.append({
            "원본 텍스트": original_text,
            "실제 라벨": actual_label,
            "확률": f"광고성 {prediction[0][1] * 100:.2f}%",
            "중요 단어": important_words,
            "images": image_path
        })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "predictions": predictions
    })

# 광고성 단어사전 로드 (전역 변수로 선언)
with open('data/6.훈련결과/광고성_단어사전.json', 'r', encoding='utf-8') as f:
    ad_dictionary = json.load(f)
    ad_words_dict = ad_dictionary['광고성_단어사전']['words']


@app.post("/predict")
async def predict(request: PredictionRequest):
    # 1. 입력 데이터 확인
    input_text = request.text
    print("Step 1: Input Text:\n", input_text)

    # 2. 텍스트 전처리 (model_train과 동일한 방식)
    preprocessed_text = preprocess_text(input_text)
    print("\nStep 2: preprocessed_text Text:\n", preprocessed_text)

    tokenized_text = tokenize_text(preprocessed_text)
    print("\nStep 3: Tokenized Text:\n", tokenized_text)

    # 3. 시퀀스 변환 (문자열로 합치지 않고 직접 토큰화된 텍스트 사용)
    tokenized_sequence = tokenizer.texts_to_sequences([tokenized_text])  # " ".join() 제거
    print("\nStep 4: Tokenized Sequence:\n", tokenized_sequence)

    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_sequence,
        maxlen=250,
        padding='post'
    )
    print("\nStep 5: Padded Sequence Shape:", padded_sequence.shape)
    #print("padded_sequence: ", padded_sequence)

    # 5. 모델 예측 및 Attention 가중치 추출
    prediction, attention_data = attention_model.predict(padded_sequence)
    print("\nStep 6: Raw Prediction Output:", prediction)
    #print("Attention Data Type:", type(attention_data))
    #print("Attention Data Content:", attention_data) #context_vector, attention_weights 반환


    # Attention 가중치가 tuple로 반환된 경우 처리
    if isinstance(attention_data, tuple):
        attention_weights = attention_data[1]  # attention_weights 추출
    else:
        attention_weights = attention_data  # 단일 값일 경우 그대로 사용

    #print("Attention Weights Shape:", attention_weights.shape)
    #print("\nAttention Weights:", attention_weights)

    # Attention 가중치 후처리
    attention_weights = mask_attention_weights(attention_weights, padded_sequence[0])
    #print("Step : Processed Attention Weights:\n", attention_weights)

    attention_scores = attention_weights.squeeze()  # Attention 가중치 (sequence_length,) 1차원 배열로 바꿔주는 역할
    #print("\nattention_scores:", attention_scores)

    # 6. 확률 계산
    ad_probability = float(prediction[0][1] * 100)
    non_ad_probability = float(prediction[0][0] * 100)
    print(f"\nStep 7: Ad Probability: {ad_probability:.2f}%, Non-Ad Probability: {non_ad_probability:.2f}%")

    # 7. 라벨 결정
    label = "광고성 게시글" if ad_probability > 50 else "일반 게시글"
    print("\nStep 8: Label:", label)

    # 8. Attention 가중치를 사용하여 중요 단어 추출
    # attention_scores = attention_weights[0].squeeze()  # (sequence_length,)로 변환
    # sequence = tokenized_text[0]  # 정수 인덱스 시퀀스
    # word_importance = []

    # 중요 단어 추출 부분
    word_importance = []
    sequence = padded_sequence[0]

    for i, word_id in enumerate(sequence):
        if word_id != 0:  # 패딩 제외
            word = tokenizer.index_word.get(word_id, "")
            if word and word in ad_words_dict:  # 단어가 존재하고 광고성 단어사전에 있는 경우만
                score = float(attention_scores[i])
                dictionary_score = float(ad_words_dict[word])
                word_importance.append({
                    "word": word,
                    "score": score,
                    "dictionary_score": dictionary_score
                })

    if word_importance:  # 광고성 단어가 하나라도 있는 경우
        #
        word_importance = sorted(word_importance, key=lambda x: x["score"], reverse=True)[:15]
    else:
        # 광고성 단어가 없는 경우
        word_importance = [{"word": "광고성 단어가 발견되지 않았습니다", "score": 0, "dictionary_score": 0}]

    print("\nStep 9: Important Words:\n", word_importance)

    # 최종 결과 반환
    result = {
        "text": input_text,
        "label": label,
        "ad_probability": ad_probability,
        "non_ad_probability": non_ad_probability,
        "important_keywords": word_importance,
    }
    return result





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)